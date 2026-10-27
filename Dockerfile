# --- STAGE 1: Build dependencies ---
FROM python:3.12-slim-bookworm AS builder

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    INSIGHTFACE_HOME=/app/models/insightface \
    HOME=/app/models \
    OMP_NUM_THREADS=1 \
    PORT=8080

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential cmake git \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
# Install python packages into a virtualenv
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
RUN pip install --upgrade pip setuptools wheel \
    && pip install -r requirements.txt

# --- STAGE 2: Final slim image ---
FROM python:3.12-slim-bookworm

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    INSIGHTFACE_HOME=/app/models/insightface \
    HOME=/app/models \
    OMP_NUM_THREADS=1 \
    PORT=8080

WORKDIR /app

# ONLY install runtime libraries (No build-essential, no cmake)
RUN apt-get update && apt-get install -y --no-install-recommends \
        libglib2.0-0 libgl1 libglx0 libegl1 libgl1-mesa-dri \
        libgomp1 libgstreamer1.0-0 libgstreamer-plugins-base1.0-0 \
        libv4l-0 ffmpeg curl \
    && rm -rf /var/lib/apt/lists/*

# Copy the virtualenv from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy app code
COPY . .

RUN mkdir -p /app/models/insightface /app/faces \
    && useradd -m -u 1000 -s /bin/bash dlc \
    && chown -R dlc:dlc /app
USER dlc

EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=5s --start-period=120s --retries=3 \
    CMD curl -fsS http://localhost:8080/ >/dev/null || exit 1

CMD ["python", "-u", "web_app.py"]
