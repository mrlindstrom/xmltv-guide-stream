# XMLTV Guide Scroller → HLS

Old cable-style TV guide renderer that ingests XMLTV, renders a scrolling guide,
and serves an HLS stream. Optional background music via `./music`.

## Requirements
- Python 3.8+
- ffmpeg on PATH (`ffmpeg -version`)
- Windows users may need `tzdata`.

## Install
```bash
python -m venv .venv && . .venv/bin/activate  # (Windows: .venv\Scripts\activate)
pip install -r requirements.txt
