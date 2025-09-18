#!/usr/bin/env python3
"""
(v24) XMLTV Guide Scroller → HLS stream (themes, muted blocks, random music per track)

Key updates vs v23:
- If there's more than ONE audio file in ./music, we now create a randomized
  ffconcat playlist and feed it to ffmpeg. This picks a random track **each time the
  previous one ends**, forever. (Single-file behavior unchanged: it loops cleanly.)
- Everything else remains: date/timezone in header's first cell, NVENC support,
  cache/live modes, logo rendering, cache cleanup, theme-based muted block colors.
"""
from __future__ import annotations

import argparse, io, os, sys, time, math, random, threading, subprocess, http.server, socketserver, re
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Tuple, Optional as _Optional
try:
    from zoneinfo import ZoneInfo
except Exception:
    try:
        from backports.zoneinfo import ZoneInfo  # type: ignore
    except Exception:
        ZoneInfo = None  # type: ignore

import requests
from xml.etree import ElementTree as ET
from urllib.parse import urljoin, urlparse
from PIL import Image, ImageDraw, ImageFont

# --- Theme support (same as v23, with auto-muted block colors) ----------------
def _hex_to_rgb(s: str):
    s = s.strip()
    if s.startswith("#"): s = s[1:]
    if len(s) == 3: s = "".join([c*2 for c in s])
    if len(s) != 6: raise ValueError(f"Bad color hex: {s}")
    return tuple(int(s[i:i+2], 16) for i in (0,2,4))

def _mix(c1, c2, t: float):
    t = max(0.0, min(1.0, float(t)))
    return tuple(int(round(c1[i]*(1.0 - t) + c2[i]*t)) for i in range(3))

def _luma(rgb):
    r,g,b = [v/255.0 for v in rgb]
    return 0.2126*r + 0.7152*g + 0.0722*b

def _best_text_on(bg, light=(235,242,250), dark=(16,16,16)):
    return dark if _luma(bg) > 0.6 else light

THEMES = {
    "classic": {
        "bg": "#060A0E","header_bg": "#0C1620","header_border": "#283C50",
        "grid_bg": "#0A121A","row_sep": "#16222E",
        "text_primary": "#EBF2FA","text_muted": "#B4C0CC","accent": "#007AFF",
        "block_colors": []  # auto-derive muted palette
    },
    "midnight": {
        "bg": "#000000","header_bg": "#0B0B14","header_border": "#1F2833",
        "grid_bg": "#0E0E18","row_sep": "#1B2430",
        "text_primary": "#F5F7FA","text_muted": "#AEB6C2","accent": "#00D1FF",
        "block_colors": []
    },
    "retro_blue": {
        "bg": "#0A1630","header_bg": "#0F1E3F","header_border": "#1E3A8A",
        "grid_bg": "#0C1A36","row_sep": "#1C2E5A",
        "text_primary": "#E6EEF9","text_muted": "#A5B4CF","accent": "#3B82F6",
        "block_colors": []
    },
    "amber": {
        "bg": "#1A1200","header_bg": "#221A00","header_border": "#735B00",
        "grid_bg": "#1D1500","row_sep": "#4D3C00",
        "text_primary": "#FFF7E6","text_muted": "#E5D3A6","accent": "#FFB300",
        "block_colors": []
    },
    "mono_light": {
        "bg": "#F3F4F6","header_bg": "#E5E7EB","header_border": "#9CA3AF",
        "grid_bg": "#FFFFFF","row_sep": "#D1D5DB",
        "text_primary": "#111827","text_muted": "#4B5563","accent": "#2563EB",
        "block_colors": []
    }
}

def _derive_block_palette(theme_colors: dict):
    acc   = theme_colors['accent']
    grid  = theme_colors['grid_bg']
    head  = theme_colors['header_bg']
    row   = theme_colors['row_sep']
    blends = [
        _mix(acc, grid, 0.80),
        _mix(acc, grid, 0.85),
        _mix(acc, grid, 0.90),
        _mix(acc, head, 0.80),
        _mix(acc, head, 0.88),
        _mix(acc, row,  0.75),
        _mix(acc, row,  0.82),
        _mix(acc, grid, 0.78),
    ]
    uniq = []
    for c in blends:
        if c not in uniq:
            uniq.append(c)
    return uniq[:6] if len(uniq) >= 6 else (uniq + [uniq[-1]]*(6-len(uniq)))

def _resolve_theme(name: str | None, theme_file: str | None):
    base = THEMES["classic"].copy()
    if name: base = THEMES.get(name, base).copy()
    if theme_file and os.path.exists(theme_file):
        import json as _json
        with open(theme_file, "r", encoding="utf-8") as fp:
            user = _json.load(fp) or {}
        base.update(user)
    out = {}
    for k,v in base.items():
        if k == "block_colors": continue
        out[k] = _hex_to_rgb(v) if isinstance(v, str) else tuple(v)
    bc = base.get("block_colors")
    if bc:
        out["block_colors"] = [(_hex_to_rgb(c) if isinstance(c, str) else tuple(v2)) for c in bc]
    else:
        out["block_colors"] = _derive_block_palette(out)
    return out

# --- Helpers -----------------------------------------------------------------
def resolve_resource(base: str, src: _Optional[str]) -> _Optional[str]:
    if not src: return None
    s = src.strip()
    if re.match(r'^(https?|file)://', s) or os.path.isabs(s): return s
    if base and re.match(r'^https?://', base):
        try: return urljoin(base, s)
        except Exception: return s
    base_dir = os.path.dirname(base) if base else '.'
    return os.path.normpath(os.path.join(base_dir, s))

def parse_xmltv_time(ts: str) -> datetime:
    ts = ts.strip()
    m = re.match(r"^(\d{4})(\d{2})(\{0\})(\d{2})(\d{2})(?:\s*([+-]\d{4}|Z))?$".replace("{0}", r"\d{2}"), ts)
    if not m:
        # Fallback: full YYYYmmddHHMMSS(+ZZZZ)
        m2 = re.match(r"^(\d{4})(\d{2})(\d{2})(\d{2})(\d{2})(\d{2})(?:\s*([+-]\d{4}|Z))?$", ts)
        if not m2: raise ValueError(f"Unrecognized XMLTV time: {ts}")
        y, mo, d, h, mi, s, off = m2.groups()
    else:
        y, mo, d, h, mi, off = m.groups()
        s = "00"
    dt = datetime(int(y), int(mo), int(d), int(h), int(mi), int(s))
    if off is None or off == 'Z':
        tz = timezone.utc
    else:
        sign = 1 if off.startswith('+') else -1
        oh = int(off[1:3]); om = int(off[3:5])
        tz = timezone(sign * timedelta(hours=oh, minutes=om))
    return dt.replace(tzinfo=tz)

def floor_to_half_hour(dt: datetime, tz: ZoneInfo) -> datetime:
    dt = dt.astimezone(tz)
    minute = 0 if dt.minute < 30 else 30
    return dt.replace(minute=minute, second=0, microsecond=0)

# --- Music helpers ------------------------------------------------------------
def _build_music_shuffle_concat(files:list[str], repeats:int=400, work_dir:str="hls_out") -> str:
    """
    Build a long randomized ffconcat playlist from `files`. Each repeat shuffles the order.
    A few hundred repeats gives many hours of unique playback.
    """
    os.makedirs(work_dir, exist_ok=True)
    out_path = os.path.join(work_dir, "music_playlist.ffconcat")
    norm = []
    for p in files:
        ap = os.path.abspath(p).replace("\\", "/")
        ap = ap.replace("'", r"'\\''")  # escape single quotes
        norm.append(ap)
    import random as _random
    with open(out_path, "w", encoding="utf-8") as fp:
        fp.write("ffconcat version 1.0\n")
        for _ in range(max(1, repeats)):
            _list = norm[:]
            _random.shuffle(_list)
            for ap in _list:
                fp.write(f"file '{ap}'\n")
    return out_path

# --- Models & Loader ----------------------------------------------------------
class Programme:
    __slots__ = ("start","stop","title","subtitle")
    def __init__(self, start: datetime, stop: datetime, title: str, subtitle: str | None):
        self.start, self.stop, self.title, self.subtitle = start, stop, title, subtitle or ""

class Channel:
    __slots__ = ("id","name","number","icon")
    def __init__(self, id: str, name: str, number: _Optional[str], icon: _Optional[str]=None):
        self.id, self.name, self.number, self.icon = id, name, number or "", icon

class EPG:
    def __init__(self, tz: ZoneInfo):
        self.tz = tz
        self.channels: List[Channel] = []
        self.programmes_by_channel: Dict[str, List[Programme]] = {}
    def programmes_in_window(self, cid: str, start: datetime, end: datetime) -> List[Programme]:
        out = []
        for p in self.programmes_by_channel.get(cid, []):
            if not (p.stop <= start or p.start >= end):
                out.append(p)
        return out

class XMLTVLoader(threading.Thread):
    def __init__(self, xmltv_url: str, tz: ZoneInfo, refresh_minutes: int=15):
        super().__init__(daemon=True)
        self.xmltv_url, self.tz, self.refresh_minutes = xmltv_url, tz, max(1, refresh_minutes)
        self.latest_epg: EPG | None = None
        self._err: str | None = None
        self._stop = threading.Event()
    @property
    def error(self): return self._err
    def stop(self): self._stop.set()
    def run(self):
        while not self._stop.is_set():
            try:
                self.latest_epg = self._fetch_and_parse(); self._err=None
            except Exception as e:
                self._err = f"EPG load error: {e}"
            for _ in range(self.refresh_minutes*60):
                if self._stop.is_set(): return
                time.sleep(1)
    def _fetch_and_parse(self) -> EPG:
        if re.match(r"^https?://", self.xmltv_url, re.I):
            r = requests.get(self.xmltv_url, timeout=60); r.raise_for_status(); data = r.content
        else:
            with open(self.xmltv_url, 'rb') as f: data = f.read()
        root = ET.fromstring(data)
        epg = EPG(self.tz)
        i=1
        for ch in root.findall('channel'):
            cid = ch.get('id') or f"ch{i}"
            disp = ch.findtext('display-name') or cid
            number = None
            lcn = ch.findtext('lcn')
            if lcn and lcn.strip():
                number = lcn.strip()
            else:
                m = re.match(r"^(\d+(?:\.\d+)?[A-Za-z]?)\s*[-–:]\s*(.+)$", disp)
                if m: number, disp = m.group(1), (m.group(2).strip() or disp)
                else:
                    m2 = re.match(r"^(\d+(?:\.\d+)?)(?:\s+)(.+)$", disp)
                    if m2: number, disp = m2.group(1), (m2.group(2).strip() or disp)
            icon_url=None
            ic = ch.find('icon')
            if ic is not None and ic.get('src'):
                icon_url = resolve_resource(self.xmltv_url, ic.get('src'))
            epg.channels.append(Channel(cid, disp.strip(), number, icon_url)); i+=1
        if not epg.channels: raise ValueError("No <channel> entries found in XMLTV")
        ids = {c.id for c in epg.channels}
        pb = {c.id: [] for c in epg.channels}
        for pr in root.findall('programme'):
            cid = pr.get('channel')
            if cid not in ids: continue
            start = parse_xmltv_time(pr.get('start')); stop_attr = pr.get('stop')
            if not stop_attr: continue
            stop = parse_xmltv_time(stop_attr)
            start = start.astimezone(self.tz); stop = stop.astimezone(self.tz)
            title = (pr.findtext('title') or '').strip() or 'Untitled'
            subtitle = (pr.findtext('sub-title') or '').strip()
            pb[cid].append(Programme(start, stop, title, subtitle))
        for cid in pb: pb[cid].sort(key=lambda p:p.start)
        epg.programmes_by_channel = pb
        return epg

# --- HTTP server --------------------------------------------------------------
class QuietHTTPRequestHandler(http.server.SimpleHTTPRequestHandler):
    def log_message(self, *args, **kwargs): pass
def start_http_server(directory: str, port: int) -> threading.Thread:
    handler = lambda *a, **kw: QuietHTTPRequestHandler(*a, directory=directory, **kw)
    httpd = socketserver.TCPServer(("0.0.0.0", port), handler)
    t = threading.Thread(target=httpd.serve_forever, daemon=True); t.start(); return t

# --- Logos --------------------------------------------------------------------
LOGO_CACHE: Dict[Tuple[str,int,int], Image.Image] = {}
def _fetch_logo_source(url_or_path: str) -> Image.Image | None:
    try:
        if not url_or_path: return None
        if re.match(r'^https?://', url_or_path):
            r = requests.get(url_or_path, timeout=15); r.raise_for_status()
            return Image.open(io.BytesIO(r.content)).convert('RGBA')
        if url_or_path.startswith('file://'):
            p = urlparse(url_or_path); path = os.path.abspath(os.path.join(p.netloc, p.path))
            return Image.open(path).convert('RGBA')
        if os.path.exists(url_or_path): return Image.open(url_or_path).convert('RGBA')
    except Exception: return None
    return None
def get_logo(resolved_url: str | None, max_w: int, max_h: int) -> Image.Image | None:
    if not resolved_url: return None
    key = (resolved_url, int(max_w), int(max_h))
    if key in LOGO_CACHE: return LOGO_CACHE[key]
    base = _fetch_logo_source(resolved_url); 
    if base is None: return None
    im = base.copy(); im.thumbnail((int(max_w), int(max_h)), Image.LANCZOS)
    LOGO_CACHE[key] = im; return im

# --- Renderer -----------------------------------------------------------------
class GuideRenderer:
    def __init__(self, width:int, height:int, tz:ZoneInfo, hours:float,
                 row_height:int=56, header_height:int=84, left_col_w:int=360,
                 scroll_px_per_sec:float=24.0, font_path:str|None=None,
                 show_now_line:bool=True, show_clock:bool=True,
                 logos:bool=True, logo_max_w:int=96, logo_max_h:int=40, logo_gap:int=8,
                 theme: dict | None = None):
        self.W,self.H,self.tz,self.hours = width,height,tz,hours
        self.row_h,self.header_h,self.left_w = row_height,header_height,left_col_w
        self.grid_w = self.W - self.left_w
        self.scroll_px_per_sec = scroll_px_per_sec
        self.show_now_line, self.show_clock = show_now_line, show_clock
        self.logos, self.logo_max_w, self.logo_max_h, self.logo_gap = logos, logo_max_w, logo_max_h, logo_gap
        try:
            if font_path and os.path.exists(font_path):
                self.font_lg = ImageFont.truetype(font_path, 28)
                self.font_md = ImageFont.truetype(font_path, 22)
                self.font_sm = ImageFont.truetype(font_path, 18)
            else:
                self.font_lg = ImageFont.load_default(); self.font_md = ImageFont.load_default(); self.font_sm = ImageFont.load_default()
        except Exception:
            self.font_lg = ImageFont.load_default(); self.font_md = ImageFont.load_default(); self.font_sm = ImageFont.load_default()
        t = theme or _resolve_theme('classic', None)
        self.bg=t['bg']; self.header_bg=t['header_bg']; self.header_border=t['header_border']
        self.grid_bg=t['grid_bg']; self.row_sep=t['row_sep']
        self.text_primary=t['text_primary']; self.text_muted=t['text_muted']; self.accent=t['accent']
        self.block_colors=t['block_colors']
    def _time_slots(self, now:datetime)->Tuple[datetime,List[datetime]]:
        anchor = floor_to_half_hour(now, self.tz); num = int(self.hours*2)
        return anchor, [anchor+timedelta(minutes=30*i) for i in range(num+1)]
    def _draw_text_clipped(self, draw:ImageDraw.ImageDraw, text:str, xy:Tuple[int,int], font, max_w:int, fill):
        t=text; bbox=draw.textbbox(xy,t,font=font)
        if bbox[2]-bbox[0] <= max_w: draw.text(xy,t,font=font,fill=fill); return
        ell='…'; lo,hi=0,len(t)
        while lo<hi:
            mid=(lo+hi)//2; tt=t[:mid]+ell; bb=draw.textbbox(xy,tt,font=font)
            if bb[2]-bb[0] <= max_w: lo=mid+1
            else: hi=mid
        tt=t[:max(0,lo-1)]+ell; draw.text(xy,tt,font=font,fill=fill)
    def _render_core(self, epg:EPG|None, err:str|None, now:datetime, elapsed:float)->Image.Image:
        anchor, slots = self._time_slots(now)
        win_start, win_end = anchor, anchor+timedelta(hours=self.hours)
        win_dur = (win_end-win_start).total_seconds()
        im = Image.new('RGB',(self.W,self.H),self.bg); draw=ImageDraw.Draw(im)
        # Header
        draw.rectangle([0,0,self.W,self.header_h], fill=self.header_bg)
        draw.line([0,self.header_h-1,self.W,self.header_h-1], fill=self.header_border, width=1)
        for i in range(len(slots)):
            x = self.left_w + int(self.grid_w*(i/(len(slots)-1)))
            draw.line([x,0,x,self.header_h], fill=self.header_border, width=1)
        for i, ts in enumerate(slots[:-1]):
            label = ts.strftime('%-I:%M %p') if os.name!='nt' else ts.strftime('%I:%M %p').lstrip('0')
            x = self.left_w + int(self.grid_w*(i/(len(slots)-1))) + 8
            draw.text((x,18), label, font=self.font_lg, fill=self.text_primary)
        # Date/TZ in header left cell
        date_label = now.strftime('%a %b %d, %Y')
        tzname = self.tz.key if hasattr(self.tz,'key') else now.tzname()
        self._draw_text_clipped(draw, date_label, (12,14), self.font_lg, self.left_w-24, self.text_primary)
        self._draw_text_clipped(draw, tzname, (12,42), self.font_md, self.left_w-24, self.text_muted)
        # Grid areas
        draw.rectangle([0,self.header_h,self.left_w,self.H], fill=self.grid_bg)
        draw.rectangle([self.left_w,self.header_h,self.W,self.H], fill=self.bg)
        draw.line([self.left_w,0,self.left_w,self.header_h], fill=self.header_border, width=1)
        # Rows
        rows_visible = max(1, (self.H - self.header_h)//self.row_h + 1)
        for r in range(rows_visible):
            y = self.header_h + r*self.row_h
            draw.line([0,y,self.W,y], fill=self.row_sep)
        draw.line([self.left_w,self.header_h,self.left_w,self.H], fill=self.row_sep)
        if epg is None:
            msg = err or "Loading EPG…"; tb=draw.textbbox((0,0), msg, font=self.font_lg)
            draw.text(((self.W-tb[2])//2, (self.H-tb[3])//2), msg, font=self.font_lg, fill=self.text_primary)
            return im
        # Scrolling
        scroll_px = (elapsed*self.scroll_px_per_sec) % (max(1,len(epg.channels))*self.row_h)
        start_row = int(scroll_px//self.row_h); y_offset = int(scroll_px%self.row_h)
        base_y = self.header_h
        for r in range(rows_visible+1):
            idx = (start_row + r) % len(epg.channels)
            y_top = base_y + r*self.row_h - y_offset
            if y_top >= self.H: break
            ch = epg.channels[idx]; left_pad=12; name_y=y_top+8
            logo_img=None; text_max_w=self.left_w-24
            if self.logos and getattr(ch,'icon',None):
                li=get_logo(ch.icon,self.logo_max_w,self.logo_max_h)
                if li is not None:
                    logo_img=li; text_max_w=max(20, text_max_w - (logo_img.width + self.logo_gap))
            if ch.number:
                self._draw_text_clipped(draw, ch.number, (left_pad,name_y), self.font_lg, text_max_w, self.text_primary)
                self._draw_text_clipped(draw, ch.name, (left_pad,name_y+28), self.font_md, text_max_w, self.text_muted)
            else:
                self._draw_text_clipped(draw, ch.name, (left_pad,name_y+10), self.font_lg, text_max_w, self.text_primary)
            if logo_img is not None:
                lx=self.left_w - logo_img.width - 12; ly=y_top + (self.row_h - logo_img.height)//2
                im.paste(logo_img,(lx,ly),logo_img)
            progs = epg.programmes_in_window(ch.id, win_start, win_end)
            for i,p in enumerate(progs):
                xs = self.left_w + int(self.grid_w * ((max(p.start,win_start)-win_start).total_seconds()/win_dur))
                xe = self.left_w + int(self.grid_w * ((min(p.stop,win_end)-win_start).total_seconds()/win_dur))
                if xe<=xs: continue
                x0=xs+1; x1=xe-2; y0=y_top+4; y1=y_top+self.row_h-6
                if x1<=x0+1 or y1<=y0: continue
                block_color = self.block_colors[i%len(self.block_colors)]
                ImageDraw.Draw(im).rectangle([x0,y0,x1,y1], fill=block_color)
                title=p.title + (f" — {p.subtitle}" if p.subtitle else "")
                tx=x0+10; ty=y0+6; maxw=(x1-x0)-20
                if maxw>20: self._draw_text_clipped(draw, title, (tx,ty), self.font_md, maxw, _best_text_on(block_color))
        if self.show_now_line:
            now_x = self.left_w + int(self.grid_w * ((now-win_start).total_seconds()/win_dur))
            if self.left_w <= now_x <= self.W:
                draw.line([now_x, self.header_h, now_x, self.H], fill=self.accent, width=2)
        if self.show_clock:
            footer = now.strftime('%I:%M:%S %p').lstrip('0') + '  •  Old-School Guide'
            tb=draw.textbbox((0,0), footer, font=self.font_md)
            draw.rectangle([0,self.H-tb[3]-12,self.W,self.H], fill=self.header_bg)
            draw.text((12,self.H - tb[3] - 8), footer, font=self.font_md, fill=self.text_muted)
        return im
    def render_frame(self, epg:EPG|None, err:str|None, t0:float)->Image.Image:
        now=datetime.now(self.tz); elapsed=time.perf_counter()-t0; return self._render_core(epg,err,now,elapsed)
    def render_frame_at(self, epg:EPG|None, err:str|None, anchor:datetime, t_sec:float)->Image.Image:
        now=anchor+timedelta(seconds=t_sec); elapsed=t_sec; return self._render_core(epg,err,now,elapsed)

# --- Strips for cache mode ----------------------------------------------------
def draw_header_layer(renderer:'GuideRenderer', anchor:datetime)->Image.Image:
    now=anchor; _,slots=renderer._time_slots(now)
    im=Image.new('RGB',(renderer.W,renderer.header_h),renderer.header_bg); draw=ImageDraw.Draw(im)
    draw.line([0,renderer.header_h-1,renderer.W,renderer.header_h-1], fill=renderer.header_border, width=1)
    for i in range(len(slots)):
        x = renderer.left_w + int(renderer.grid_w * (i/(len(slots)-1)))
        draw.line([x,0,x,renderer.header_h], fill=renderer.header_border, width=1)
    for i, ts in enumerate(slots[:-1]):
        label = ts.strftime('%-I:%M %p') if os.name!='nt' else ts.strftime('%I:%M %p').lstrip('0')
        x = renderer.left_w + int(renderer.grid_w * (i/(len(slots)-1))) + 8
        draw.text((x,18), label, font=renderer.font_lg, fill=renderer.text_primary)
    date_label = now.strftime('%a %b %d, %Y')
    tzname = renderer.tz.key if hasattr(renderer.tz,'key') else now.tzname()
    renderer._draw_text_clipped(draw, date_label, (12,14), renderer.font_lg, renderer.left_w-24, renderer.text_primary)
    renderer._draw_text_clipped(draw, tzname, (12,42), renderer.font_md, renderer.left_w-24, renderer.text_muted)
    return im

def build_scrolling_strip(epg:'EPG', renderer:'GuideRenderer', anchor:datetime)->Image.Image:
    win_start=anchor; win_end=anchor+timedelta(hours=renderer.hours); win_dur=(win_end-win_start).total_seconds()
    n=max(1,len(epg.channels)); strip_h=n*renderer.row_h
    im=Image.new('RGB',(renderer.W,strip_h),renderer.bg); draw=ImageDraw.Draw(im)
    for idx in range(n):
        y_top=idx*renderer.row_h
        ImageDraw.Draw(im).rectangle([0,y_top,renderer.left_w,y_top+renderer.row_h], fill=renderer.grid_bg)
        draw.line([0,y_top,renderer.W,y_top], fill=renderer.row_sep)
    draw.line([renderer.left_w,0,renderer.left_w,strip_h], fill=renderer.row_sep)
    for idx,ch in enumerate(epg.channels):
        y_top=idx*renderer.row_h; left_pad=12; name_y=y_top+8
        logo_img=None; text_max_w=renderer.left_w-24
        if renderer.logos and getattr(ch,'icon',None):
            li=get_logo(ch.icon,renderer.logo_max_w,renderer.logo_max_h)
            if li is not None: logo_img=li; text_max_w=max(20, text_max_w - (logo_img.width + renderer.logo_gap))
        if ch.number:
            renderer._draw_text_clipped(draw, ch.number, (left_pad,name_y), renderer.font_lg, text_max_w, renderer.text_primary)
            renderer._draw_text_clipped(draw, ch.name, (left_pad,name_y+28), renderer.font_md, text_max_w, renderer.text_muted)
        else:
            renderer._draw_text_clipped(draw, ch.name, (left_pad,name_y+10), renderer.font_lg, text_max_w, renderer.text_primary)
        if logo_img is not None:
            lx=renderer.left_w - logo_img.width - 12; ly=y_top + (renderer.row_h - logo_img.height)//2
            im.paste(logo_img,(lx,ly),logo_img)
        progs=epg.programmes_in_window(ch.id,win_start,win_end)
        for i,p in enumerate(progs):
            xs = renderer.left_w + int(renderer.grid_w * ((max(p.start,win_start)-win_start).total_seconds()/win_dur))
            xe = renderer.left_w + int(renderer.grid_w * ((min(p.stop,win_end)-win_start).total_seconds()/win_dur))
            if xe<=xs: continue
            x0=xs+1; x1=xe-2; y0=y_top+4; y1=y_top+renderer.row_h-6
            if x1<=x0+1 or y1<=y0: continue
            block_color = renderer.block_colors[i%len(renderer.block_colors)]
            ImageDraw.Draw(im).rectangle([x0,y0,x1,y1], fill=block_color)
            title=p.title + (f" — {p.subtitle}" if p.subtitle else "")
            tx=x0+10; ty=y0+6; maxw=(x1-x0)-20
            if maxw>20: renderer._draw_text_clipped(draw, title, (tx,ty), renderer.font_md, maxw, _best_text_on(block_color))
    return im

# --- FFmpeg helpers -----------------------------------------------------------
def compute_loop_seconds(epg:EPG, renderer:'GuideRenderer')->float:
    n=max(1,len(epg.channels)); return (n*renderer.row_h)/max(1e-6, renderer.scroll_px_per_sec)

def encode_block_to_mp4(renderer:'GuideRenderer', epg:'EPG', anchor:datetime, loop_seconds:float,
                        width:int, height:int, fps:int, out_mp4:str, vcodec:str='libx264',
                        nvenc_preset:str|None=None, rc:str|None=None, bitrate:str='2500k',
                        maxrate:str|None=None, bufsize:str|None=None, profile:str|None=None,
                        pix_fmt:str='yuv420p'):
    os.makedirs(os.path.dirname(out_mp4) or '.', exist_ok=True)
    tmp_mp4 = out_mp4 + '.tmp'
    frame_count = int(math.ceil(max(0.1, float(loop_seconds)) * fps))
    header_img = draw_header_layer(renderer, anchor)
    strip = build_scrolling_strip(epg, renderer, anchor); strip_h=strip.height
    view_h = height - renderer.header_h
    repeats = max(2, math.ceil((view_h + strip_h) / strip_h))
    tiled_h = strip_h * repeats
    tiled = Image.new('RGB', (width, tiled_h), renderer.bg)
    y=0
    for _ in range(repeats):
        tiled.paste(strip,(0,y)); y+=strip_h
    cmd = [
        'ffmpeg','-y','-hide_banner','-loglevel','warning',
        '-f','rawvideo','-pix_fmt','rgb24','-video_size',f'{width}x{height}','-framerate',str(fps),'-i','-',
        '-f','lavfi','-i','anullsrc=r=48000:cl=stereo',
    ]
    if vcodec.lower() in ('h264_nvenc','hevc_nvenc'):
        cmd += ['-c:v',vcodec]
        if nvenc_preset: cmd += ['-preset', nvenc_preset]
        if rc: cmd += ['-rc', rc]
        cmd += ['-b:v', bitrate]
        if maxrate: cmd += ['-maxrate', maxrate]
        if bufsize: cmd += ['-bufsize', bufsize]
        if profile: cmd += ['-profile:v', profile]
        if pix_fmt: cmd += ['-pix_fmt', pix_fmt]
        cmd += ['-g', str(fps*2), '-force_key_frames', 'expr:gte(t,n_forced*2)']
    else:
        cmd += ['-c:v','libx264','-preset','veryfast']
        cmd += ['-pix_fmt',pix_fmt,'-profile:v',(profile or 'high'),'-b:v',bitrate]
        cmd += ['-g',str(fps*2),'-keyint_min',str(fps*2),'-sc_threshold','0','-force_key_frames','expr:gte(t,n_forced*2)']
    cmd += ['-c:a','aac','-b:a','128k','-shortest','-movflags','+faststart','-t',f"{frame_count/fps:.6f}",'-f','mp4',tmp_mp4]
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    for n in range(frame_count):
        t_sec = n / fps
        yoff = int((t_sec * renderer.scroll_px_per_sec) % strip_h)
        crop = tiled.crop((0, yoff, width, yoff + view_h))
        frame = Image.new('RGB', (width, height), renderer.bg)
        frame.paste(header_img,(0,0))
        frame.paste(crop,(0,renderer.header_h))
        try: p.stdin.write(frame.tobytes())
        except (BrokenPipeError,OSError): break
    try:
        if p.stdin: p.stdin.close()
    except Exception: pass
    ret=p.wait()
    if ret!=0 or not os.path.exists(tmp_mp4):
        raise RuntimeError(f"ffmpeg encode failed (code {ret}) for {out_mp4}")
    os.replace(tmp_mp4,out_mp4)

def start_hls_from_mp4_loop_reencode(mp4_path:str, out_dir:str, playlist:str,
    vcodec:str, nvenc_preset:str|None, rc:str|None, bitrate:str,
    maxrate:str|None, bufsize:str|None, profile:str|None, pix_fmt:str, fps:int,
    verbose:bool=False, music_file:str|None=None, music_vol:float=0.25) -> subprocess.Popen:
    os.makedirs(out_dir, exist_ok=True); out_path=os.path.join(out_dir, playlist)
    loglevel='info' if verbose else 'warning'
    cmd=['ffmpeg','-hide_banner','-loglevel',loglevel,
         '-thread_queue_size','1024','-re','-stream_loop','-1','-i',mp4_path,
         '-fflags','+genpts']
    if music_file:
        if music_file.lower().endswith('.ffconcat'):
            cmd += ['-f','concat','-safe','0','-i', music_file]
        else:
            cmd += ['-thread_queue_size','1024','-re','-stream_loop','-1','-i', music_file]
    if vcodec.lower() in ('h264_nvenc','hevc_nvenc'):
        cmd += ['-c:v',vcodec]
        if nvenc_preset: cmd += ['-preset', nvenc_preset]
        if rc: cmd += ['-rc', rc]
        cmd += ['-b:v', bitrate]
        if maxrate: cmd += ['-maxrate', maxrate]
        if bufsize: cmd += ['-bufsize', bufsize]
        if profile: cmd += ['-profile:v', profile]
        if pix_fmt: cmd += ['-pix_fmt', pix_fmt]
        cmd += ['-g', str(fps*2), '-force_key_frames', 'expr:gte(t,n_forced*2)']
    else:
        cmd += ['-c:v','libx264','-preset','veryfast','-tune','zerolatency']
        cmd += ['-pix_fmt',pix_fmt,'-profile:v',(profile or 'high'),'-b:v',bitrate]
        cmd += ['-g',str(fps*2),'-keyint_min',str(fps*2),'-sc_threshold','0','-force_key_frames','expr:gte(t,n_forced*2)']
    if music_file:
        cmd += ['-map','0:v:0','-map','1:a:0','-filter:a',f'volume={music_vol}','-c:a','aac','-b:a','128k']
    else:
        cmd += ['-c:a','aac','-b:a','128k']
    cmd += ['-fps_mode:v','vfr']
    cmd += ['-f','hls','-hls_time','2','-hls_list_size','60',
            '-hls_flags','delete_segments+independent_segments+temp_file',
            '-hls_playlist_type','event', out_path]
    return subprocess.Popen(cmd)

def start_hls_from_mp4_loop_copy(mp4_path:str, out_dir:str, playlist:str,
    verbose:bool=False, music_file:str|None=None, music_vol:float=0.25) -> subprocess.Popen:
    os.makedirs(out_dir, exist_ok=True); out_path=os.path.join(out_dir, playlist)
    loglevel='info' if verbose else 'warning'
    cmd=['ffmpeg','-hide_banner','-loglevel',loglevel]
    if music_file:
        if music_file.lower().endswith('.ffconcat'):
            cmd += ['-thread_queue_size','1024','-re','-stream_loop','-1','-i', mp4_path,
                    '-f','concat','-safe','0','-i', music_file,
                    '-map','0:v:0','-c:v','copy',
                    '-map','1:a:0','-filter:a',f'volume={music_vol}','-c:a','aac']
        else:
            cmd += ['-thread_queue_size','1024','-re','-stream_loop','-1','-i', mp4_path,
                    '-thread_queue_size','1024','-re','-stream_loop','-1','-i', music_file,
                    '-map','0:v:0','-c:v','copy',
                    '-map','1:a:0','-filter:a',f'volume={music_vol}','-c:a','aac']
    else:
        cmd += ['-re','-stream_loop','-1','-i', mp4_path, '-c:v','copy','-c:a','copy']
    cmd += ['-f','hls','-hls_time','2','-hls_list_size','60',
            '-hls_flags','delete_segments+independent_segments+temp_file',
            '-hls_playlist_type','event', out_path]
    return subprocess.Popen(cmd)

# --- Cache cleanup ------------------------------------------------------------
def clean_cache_dir(path:str)->int:
    removed=0
    try:
        for n in os.listdir(path):
            if n.endswith('.lock') or n.endswith('.tmp'):
                try: os.remove(os.path.join(path,n)); removed+=1
                except Exception: pass
    except FileNotFoundError: pass
    return removed

def _parse_block_dt(name:str, tz:ZoneInfo)->datetime|None:
    m=re.match(r'^(\d{8}T\d{4})\.mp4$', name); 
    if not m: return None
    try:
        dt=datetime.strptime(m.group(1),'%Y%m%dT%H%M'); return dt.replace(tzinfo=tz)
    except Exception: return None

def purge_old_cache_blocks(cache_dir:str, current_anchor:datetime, tz:ZoneInfo, grace_sec:int=300)->int:
    removed=0
    try:
        cutoff=current_anchor - timedelta(seconds=max(0,grace_sec))
        for name in os.listdir(cache_dir):
            if not name.endswith('.mp4'): continue
            dt=_parse_block_dt(name,tz)
            if dt is None: continue
            if dt < cutoff:
                try: os.remove(os.path.join(cache_dir,name)); removed+=1
                except Exception: pass
    except FileNotFoundError: pass
    return removed

# --- Music scanning -----------------------------------------------------------
def _scan_music_dir(music_dir:str)->list[str]:
    try:
        exts=('.mp3','.m4a','.aac','.flac','.ogg','.wav','.opus'); files=[]
        if music_dir and os.path.isdir(music_dir):
            for root,_,fnames in os.walk(music_dir):
                for n in fnames:
                    if n.lower().endswith(exts): files.append(os.path.join(root,n))
        return files
    except Exception: return []

# --- Main ---------------------------------------------------------------------
def main():
    ap=argparse.ArgumentParser(description='XMLTV → Old-school scrolling guide → HLS stream')
    ap.add_argument('--xmltv', required=True)
    ap.add_argument('--tz', default=os.environ.get('TZ') or 'America/Chicago')
    ap.add_argument('--hours', type=float, default=3.0)
    ap.add_argument('--res', default='1280x720')
    ap.add_argument('--fps', type=int, default=30)
    ap.add_argument('--bitrate', default='2500k')
    ap.add_argument('--vcodec', default='libx264')
    ap.add_argument('--nvenc-preset', default='p5')
    ap.add_argument('--rc', default=None)
    ap.add_argument('--maxrate', default=None)
    ap.add_argument('--bufsize', default=None)
    ap.add_argument('--profile', default=None)
    ap.add_argument('--pix-fmt', dest='pix_fmt', default='yuv420p')
    ap.add_argument('--hls-dir', default='hls_out')
    ap.add_argument('--http-port', type=int, default=8000)
    ap.add_argument('--row-height', type=int, default=56)
    ap.add_argument('--scroll-pps', type=float, default=24.0)
    ap.add_argument('--font', default=None)
    ap.add_argument('--refresh-min', type=int, default=15)
    ap.add_argument('--no-logos', action='store_true')
    ap.add_argument('--logo-max-w', type=int, default=96)
    ap.add_argument('--logo-max-h', type=int, default=40)
    ap.add_argument('--left-col-w', type=int, default=360)
    ap.add_argument('--theme', default='classic')
    ap.add_argument('--theme-file', default=None)
    ap.add_argument('--mode', choices=['live','cache'], default='cache')
    ap.add_argument('--cache-dir', default='cache_blocks')
    ap.add_argument('--block-seconds', type=int, default=1800)
    ap.add_argument('--precache-blocks', type=int, default=6)
    ap.add_argument('--cache-grace-sec', type=int, default=300)
    ap.add_argument('--packager-copy', action='store_true')
    ap.add_argument('--hls-verbose', action='store_true')
    ap.add_argument('--music-dir', default='music')
    ap.add_argument('--no-music', action='store_true')
    ap.add_argument('--music-volume', type=float, default=0.25)

    args=ap.parse_args()
    if ZoneInfo is None:
        print("ZoneInfo not available. Install Python 3.9+ or: pip install backports.zoneinfo tzdata"); sys.exit(2)
    m=re.match(r'^(\d+)x(\d+)$', args.res)
    if not m: print('Invalid --res; use WxH'); sys.exit(2)
    W,H=int(m.group(1)), int(m.group(2))
    try: tz=ZoneInfo(args.tz)
    except Exception as e:
        print(f"Failed to load timezone '{args.tz}'. Try: pip install tzdata\nError: {e}"); sys.exit(2)

    theme_cfg = _resolve_theme(args.theme, args.theme_file)

    # Music discovery
    MUSIC_FILES=[]; SELECTED_MUSIC=None; MUSIC_PLAYLIST=None
    if not args.no_music and args.music_dir:
        MUSIC_FILES=_scan_music_dir(args.music_dir)
        if MUSIC_FILES:
            if len(MUSIC_FILES) == 1:
                SELECTED_MUSIC = MUSIC_FILES[0]
                print(f"[music] single file: {os.path.basename(SELECTED_MUSIC)} (looping)")
            else:
                MUSIC_PLAYLIST = _build_music_shuffle_concat(MUSIC_FILES, repeats=400, work_dir=args.hls_dir)
                print(f"[music] playlist with {len(MUSIC_FILES)} tracks → randomized forever")
        else:
            print("[music] no music files found; continuing without music")
    else:
        print("[music] disabled")

    loader=XMLTVLoader(args.xmltv, tz, refresh_minutes=args.refresh_min); loader.start()
    os.makedirs(args.hls_dir, exist_ok=True); start_http_server(args.hls_dir, args.http_port)
    renderer=GuideRenderer(W,H,tz,args.hours,row_height=args.row_height,header_height=84,left_col_w=args.left_col_w,
                           scroll_px_per_sec=args.scroll_pps,font_path=args.font,
                           show_now_line=(args.mode=='live'), show_clock=(args.mode=='live'),
                           logos=(not args.no_logos), logo_max_w=args.logo_max_w, logo_max_h=args.logo_max_h,
                           theme=theme_cfg)

    print("===============================================================")
    print("XMLTV Guide Scroller → HLS")
    print(f"Serving HLS on: http://localhost:{args.http_port}/guide.m3u8")
    print(f"Video encoder: {args.vcodec}")
    print("Open that URL in VLC or any HLS-compatible player.")
    print("Press Ctrl+C to stop.")
    print("===============================================================")

    if args.mode=='cache':
        try:
            os.makedirs(args.cache_dir, exist_ok=True)
            n=clean_cache_dir(args.cache_dir)
            if n: print(f"[cache] removed {n} stale .lock/.tmp files from {args.cache_dir}")
            now_dt=datetime.now(tz); anchor_now=floor_to_half_hour(now_dt,tz)
            rm=purge_old_cache_blocks(args.cache_dir, anchor_now, tz, args.cache_grace_sec)
            if rm: print(f"[cache] purged {rm} old cache blocks from {args.cache_dir}")
        except Exception as e:
            print(f"[cache] cleanup warning: {e}")

    if args.mode=='live':
        ff=_launch_live_hls(W,H,args,renderer,loader,
                            music_file=(MUSIC_PLAYLIST or SELECTED_MUSIC),
                            music_vol=args.music_volume)
        try: ff.wait()
        except KeyboardInterrupt: pass
        finally:
            try:
                if ff and ff.stdin: ff.stdin.close()
            except Exception: pass
            try:
                if ff: ff.terminate()
            except Exception: pass
            try: loader.stop()
            except Exception: pass
        return

    while loader.latest_epg is None and loader.error is None: time.sleep(0.5)
    if loader.latest_epg is None and loader.error: print(loader.error); sys.exit(1)

    def block_name(dt:datetime)->str: return dt.strftime('%Y%m%dT%H%M')
    def ensure_block(anchor_dt:datetime):
        epg=loader.latest_epg; mp4=os.path.join(args.cache_dir, block_name(anchor_dt)+'.mp4')
        if os.path.exists(mp4): return mp4
        lock_path=mp4+'.lock'
        while True:
            try: fd=os.open(lock_path, os.O_CREAT|os.O_EXCL|os.O_WRONLY); os.close(fd); break
            except FileExistsError:
                if os.path.exists(mp4): return mp4
                time.sleep(0.2)
        try:
            if not os.path.exists(mp4):
                print(f"[cache] encoding block {anchor_dt} → {mp4}")
                loop_secs=compute_loop_seconds(epg,renderer)
                print(f"[cache] loop_seconds≈{loop_secs:.2f}s, frames≈{int(math.ceil(loop_secs*args.fps))}")
                encode_block_to_mp4(renderer, epg, anchor_dt, loop_secs, W,H,args.fps, mp4,
                                    vcodec=args.vcodec, nvenc_preset=args.nvenc_preset, rc=args.rc,
                                    bitrate=args.bitrate, maxrate=args.maxrate, bufsize=args.bufsize,
                                    profile=args.profile, pix_fmt=args.pix_fmt)
            return mp4
        finally:
            try: os.remove(lock_path)
            except Exception: pass

    stop_precache=threading.Event()
    def precache_loop():
        while not stop_precache.is_set():
            now=datetime.now(tz); anchor=floor_to_half_hour(now,tz)
            for i in range(max(1,args.precache_blocks)):
                a=anchor+timedelta(minutes=30*i)
                try: ensure_block(a)
                except Exception as e: print(f"[cache] failed to encode {a}: {e}")
            try:
                rm=purge_old_cache_blocks(args.cache_dir, anchor, tz, args.cache_grace_sec)
                if rm: print(f"[cache] purged {rm} old cache blocks")
            except Exception as e:
                print(f"[cache] purge warning: {e}")
            for _ in range(5*60):
                if stop_precache.is_set(): return
                time.sleep(1)
    os.makedirs(args.cache_dir, exist_ok=True)
    threading.Thread(target=precache_loop, daemon=True).start()

    streamer=None
    try:
        while True:
            now=datetime.now(tz); anchor=floor_to_half_hour(now,tz)
            mp4=ensure_block(anchor)
            music_input = MUSIC_PLAYLIST or SELECTED_MUSIC
            if streamer:
                try: streamer.terminate()
                except Exception: pass
            if args.packager_copy:
                streamer = start_hls_from_mp4_loop_copy(mp4, args.hls_dir, 'guide.m3u8',
                                                        verbose=args.hls_verbose,
                                                        music_file=music_input, music_vol=args.music_volume)
            else:
                streamer = start_hls_from_mp4_loop_reencode(mp4, args.hls_dir, 'guide.m3u8',
                                        vcodec=args.vcodec, nvenc_preset=args.nvenc_preset, rc=args.rc,
                                        bitrate=args.bitrate, maxrate=args.maxrate, bufsize=args.bufsize,
                                        profile=args.profile, pix_fmt=args.pix_fmt, fps=args.fps,
                                        verbose=args.hls_verbose, music_file=music_input, music_vol=args.music_volume)
            try:
                rm=purge_old_cache_blocks(args.cache_dir, anchor, tz, args.cache_grace_sec)
                if rm: print(f"[cache] purged {rm} old cache blocks")
            except Exception as e:
                print(f"[cache] purge warning: {e}")
            next_boundary=anchor+timedelta(seconds=args.block_seconds); last_purge=time.time()
            while datetime.now(tz) < next_boundary:
                time.sleep(1)
                if streamer and (streamer.poll() is not None):
                    print("[hls] packager exited early; restarting…")
                    if args.packager_copy:
                        streamer = start_hls_from_mp4_loop_copy(mp4, args.hls_dir, 'guide.m3u8', verbose=args.hls_verbose,
                                                                music_file=music_input, music_vol=args.music_volume)
                    else:
                        streamer = start_hls_from_mp4_loop_reencode(mp4, args.hls_dir, 'guide.m3u8',
                                            vcodec=args.vcodec, nvenc_preset=args.nvenc_preset, rc=args.rc,
                                            bitrate=args.bitrate, maxrate=args.maxrate, bufsize=args.bufsize,
                                            profile=args.profile, pix_fmt=args.pix_fmt, fps=args.fps,
                                            verbose=args.hls_verbose, music_file=music_input, music_vol=args.music_volume)
                if time.time()-last_purge >= 60:
                    try:
                        rm=purge_old_cache_blocks(args.cache_dir, anchor, tz, args.cache_grace_sec)
                        if rm: print(f"[cache] purged {rm} old cache blocks")
                    except Exception as e:
                        print(f"[cache] purge warning: {e}")
                    last_purge=time.time()
    except KeyboardInterrupt:
        pass
    finally:
        try:
            if streamer: streamer.terminate()
        except Exception: pass
        try: loader.stop()
        except Exception: pass

# --- Live (on-the-fly) --------------------------------------------------------
def _launch_live_hls(W:int,H:int,args,renderer:GuideRenderer,loader:XMLTVLoader,
                     music_file:str|None=None, music_vol:float=0.25):
    out_path=os.path.join(args.hls_dir,'guide.m3u8')
    cmd=['ffmpeg','-hide_banner','-loglevel','warning',
         '-f','rawvideo','-pix_fmt','rgb24','-video_size',f'{W}x{H}','-framerate',str(args.fps),'-i','-',
         '-f','lavfi','-i','anullsrc=r=48000:cl=stereo']
    if args.vcodec.lower() in ('h264_nvenc','hevc_nvenc'):
        cmd+=['-c:v',args.vcodec]
        if args.nvenc_preset: cmd+=['-preset',args.nvenc_preset]
        if args.rc: cmd+=['-rc',args.rc]
        cmd+=['-b:v',args.bitrate]
        if args.maxrate: cmd+=['-maxrate',args.maxrate]
        if args.bufsize: cmd+=['-bufsize',args.bufsize]
        if args.profile: cmd+=['-profile:v',args.profile]
        if args.pix_fmt: cmd+=['-pix_fmt',args.pix_fmt]
        cmd+=['-g',str(args.fps*2),'-force_key_frames','expr:gte(t,n_forced*2)']
    else:
        cmd+=['-c:v','libx264','-preset','veryfast','-tune','zerolatency']
        cmd+=['-pix_fmt',args.pix_fmt,'-profile:v',(args.profile or 'high'),'-b:v',args.bitrate]
        cmd+=['-g',str(args.fps*2),'-keyint_min',str(args.fps*2),'-sc_threshold','0','-force_key_frames','expr:gte(t,n_forced*2)']
    if music_file:
        if music_file.lower().endswith('.ffconcat'):
            cmd+=['-f','concat','-safe','0','-i', music_file,
                  '-map','0:v:0','-map','2:a:0','-filter:a',f'volume={music_vol}','-c:a','aac','-b:a','128k']
        else:
            cmd+=['-thread_queue_size','1024','-re','-stream_loop','-1','-i',music_file,
                  '-map','0:v:0','-map','2:a:0','-filter:a',f'volume={music_vol}','-c:a','aac','-b:a','128k']
    else:
        cmd+=['-map','0:v:0','-map','1:a:0','-c:a','aac','-b:a','128k']
    cmd+=['-f','hls','-hls_time','2','-hls_list_size','60',
          '-hls_flags','delete_segments+independent_segments+temp_file',
          '-hls_playlist_type','event', out_path]
    ff=subprocess.Popen(cmd, stdin=subprocess.PIPE)
    t0=time.perf_counter(); frame_interval=1.0/max(1,args.fps); next_frame=time.perf_counter()
    try:
        while True:
            nowp=time.perf_counter()
            if nowp<next_frame: time.sleep(max(0,next_frame-nowp))
            next_frame+=frame_interval
            epg=loader.latest_epg; err=loader.error
            im=renderer.render_frame(epg, err, t0)
            ff.stdin.write(im.tobytes())
    except (BrokenPipeError,OSError):
        print("ffmpeg pipe closed. Exiting.")
    return ff

if __name__=='__main__':
    main()
