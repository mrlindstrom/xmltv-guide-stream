FROM debian:trixie

# Base deps
RUN apt-get update -y && \
    apt-get install -y --no-install-recommends \
      ffmpeg python3 python3-pillow python3-requests fonts-dejavu-core tzdata && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /

# Your app
COPY xmltv_guide_stream.py /xmltv_guide_stream.py

# Add the entrypoint wrapper
COPY docker-entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

# Web Server port for HLS
EXPOSE 8000

# Vars are intentionally left blank here; compose can set them.
# If they are blank at runtime, the entrypoint will fill defaults.
ENV XMLTV_SRC="" \
    XMLTV_TZ="" \
    XMLTV_HOURS="" \
    XMLTV_RESOLUTION="" \
    XMLTV_PORT="" \
    XMLTV_VCODEC="" \
    XMLTV_NVPRESET="" \
    XMLTV_RC="" \
    XMLTV_BITRATE="" \
    XMLTV_MAXRATE="" \
    XMLTV_BUFSIZE=""

ENTRYPOINT ["/entrypoint.sh"]
