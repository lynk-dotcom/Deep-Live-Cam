FROM python:3.12-slim-bookworm

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    QT_QPA_PLATFORM=offscreen \
    INSIGHTFACE_HOME=/app/models/insightface \
    # --- tunables (override at docker run -e) ---
    TARGET_FPS=12 \
    MAX_WIDTH=640 \
    MAX_HEIGHT=480 \
    EXECUTION_PROVIDER=cpu \
    OMP_NUM_THREADS=4 \
    OPENCV_FFMPEG_CAPTURE_OPTIONS="threads;1"

WORKDIR /app

# --- system deps -----------------------------------------------------------
# Kept: OpenCV/glib runtime, ffmpeg for video I/O, OpenMP for ONNX Runtime,
#       v4l for any local camera passthrough, curl for healthcheck.
# Dropped: full Qt/X11 stack (ui.py is stubbed; not needed for web).
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential cmake git curl wget ca-certificates \
        libglib2.0-0 libgl1 libglx0 \
        libgomp1 \
        libgstreamer1.0-0 libgstreamer-plugins-base1.0-0 \
        libv4l-0 \
        ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# --- python deps (cached separately from app code) -------------------------
COPY requirements.txt /app/requirements.txt
RUN python -m pip install --upgrade pip setuptools wheel \
    && python -m pip install -r /app/requirements.txt

# --- app + non-root user ---------------------------------------------------
COPY . /app
RUN mkdir -p /app/models/insightface \
    && useradd -m -u 1000 -s /bin/bash dlc \
    && chown -R dlc:dlc /app
USER dlc

EXPOSE 7860

HEALTHCHECK --interval=30s --timeout=5s --start-period=90s --retries=3 \
    CMD curl -fsS http://localhost:7860/ >/dev/null || exit 1

CMD ["python", "-u", "web_app.py"]
