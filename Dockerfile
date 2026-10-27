# syntax=docker/dockerfile:1.7
FROM python:3.12-slim-bookworm

# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    QT_QPA_PLATFORM=offscreen \
    INSIGHTFACE_HOME=/app/models/insightface \
    # runtime tunables — override at `docker run -e` or in compose
    TARGET_FPS=12 \
    MAX_WIDTH=640 \
    MAX_HEIGHT=480 \
    EXECUTION_PROVIDER=cpu \
    OMP_NUM_THREADS=4 \
    OPENCV_FFMPEG_CAPTURE_OPTIONS="threads;1" \
    GRADIO_SERVER_NAME=0.0.0.0 \
    GRADIO_SERVER_PORT=7860

WORKDIR /app

# ---------------------------------------------------------------------------
# System packages
# ---------------------------------------------------------------------------
# libegl1 / libgl1-mesa-dri kept as belt-and-suspenders: even though ui.py is
# stubbed at runtime, any stray Qt import will still find its libraries.
RUN apt-get update && apt-get install -y --no-install-recommends \
        # build toolchain (needed for a few wheels: insightface, onnxruntime)
        build-essential \
        cmake \
        git \
        curl \
        wget \
        ca-certificates \
        # OpenCV / GL runtime
        libglib2.0-0 \
        libgl1 \
        libglx0 \
        libegl1 \
        libgl1-mesa-dri \
        # OpenMP (onnxruntime CPU EP)
        libgomp1 \
        # GStreamer (opencv video backend)
        libgstreamer1.0-0 \
        libgstreamer-plugins-base1.0-0 \
        # v4l (optional local camera passthrough)
        libv4l-0 \
        # media
        ffmpeg \
    && rm -rf /var/lib/apt/lists/* /var/cache/apt/archives/*

# ---------------------------------------------------------------------------
# Python dependencies (cached separately from app source)
# ---------------------------------------------------------------------------
COPY requirements.txt /app/requirements.txt
RUN python -m pip install --upgrade pip setuptools wheel \
    && python -m pip install -r /app/requirements.txt

# ---------------------------------------------------------------------------
# Application source + non-root user
# ---------------------------------------------------------------------------
COPY . /app

RUN mkdir -p /app/models/insightface /app/models \
    && useradd -m -u 1000 -s /bin/bash dlc \
    && chown -R dlc:dlc /app
USER dlc

# ---------------------------------------------------------------------------
# Runtime
# ---------------------------------------------------------------------------
EXPOSE 7860

HEALTHCHECK --interval=30s --timeout=5s --start-period=120s --retries=3 \
    CMD curl -fsS http://localhost:7860/ >/dev/null || exit 1

# -u keeps stdout unbuffered so `docker logs` shows preflight + tracebacks
CMD ["python", "-u", "web_app.py"]
