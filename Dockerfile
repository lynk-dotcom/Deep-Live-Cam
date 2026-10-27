FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    QT_QPA_PLATFORM=offscreen \
    INSIGHTFACE_HOME=/app/models/insightface \
    TARGET_FPS=20 \
    MAX_WIDTH=1280 \
    MAX_HEIGHT=720 \
    EXECUTION_PROVIDER=cuda \
    OMP_NUM_THREADS=4

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.10 python3-pip python3.10-dev \
        build-essential cmake git curl wget ca-certificates \
        libglib2.0-0 libgl1 libglx0 libegl1 libgl1-mesa-dri \
        libgomp1 libgstreamer1.0-0 libgstreamer-plugins-base1.0-0 \
        libv4l-0 ffmpeg \
    && rm -rf /var/lib/apt/lists/*

RUN ln -sf /usr/bin/python3.10 /usr/local/bin/python \
    && ln -sf /usr/bin/pip3 /usr/local/bin/pip

WORKDIR /app
COPY requirements.txt /app/requirements.txt
RUN pip install --upgrade pip setuptools wheel \
    && pip install -r /app/requirements.txt \
    && pip install onnxruntime-gpu>=1.18 \
    && pip uninstall -y onnxruntime

COPY . /app
RUN mkdir -p /app/models/insightface \
    && useradd -m -u 1000 -s /bin/bash dlc \
    && chown -R dlc:dlc /app
USER dlc

EXPOSE 7860
HEALTHCHECK --interval=30s --timeout=5s --start-period=120s --retries=3 \
    CMD curl -fsS http://localhost:7860/ >/dev/null || exit 1

CMD ["python", "-u", "web_app.py"]
