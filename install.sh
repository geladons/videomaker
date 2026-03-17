#!/usr/bin/env bash
set -euo pipefail

APP_DIR="/workspace/videomaker"
VENV_DIR="$APP_DIR/.venv"
MODELS_DIR="/models"
WHISPER_DIR="$MODELS_DIR/whisper"
PIPER_DIR="$MODELS_DIR/piper"

export DEBIAN_FRONTEND=noninteractive

apt-get update -y
apt-get install -y --no-install-recommends \
  ffmpeg \
  python3-pip \
  git \
  build-essential \
  pkg-config \
  curl \
  wget \
  libsndfile1 \
  libomp5 \
  espeak-ng \
  fonts-inter \
  software-properties-common \
  ca-certificates \
  gnupg \
  libssl-dev \
  zlib1g-dev \
  libbz2-dev \
  libreadline-dev \
  libsqlite3-dev \
  libffi-dev \
  liblzma-dev \
  tk-dev

ensure_python311() {
  if command -v python3.11 >/dev/null 2>&1; then
    return 0
  fi

  if apt-get install -y --no-install-recommends python3.11 python3.11-venv python3.11-dev; then
    return 0
  fi

  echo "python3.11 not found in apt repo, trying deadsnakes PPA..."
  add-apt-repository ppa:deadsnakes/ppa -y || true
  apt-get update -y
  if apt-get install -y --no-install-recommends python3.11 python3.11-venv python3.11-dev; then
    return 0
  fi

  echo "python3.11 still unavailable, building from source..."
  PY_VER="3.11.9"
  PY_PREFIX="/opt/python3.11"
  if [ ! -x "${PY_PREFIX}/bin/python3.11" ]; then
    curl -fL "https://www.python.org/ftp/python/${PY_VER}/Python-${PY_VER}.tgz" -o /tmp/python311.tgz
    tar -xzf /tmp/python311.tgz -C /tmp
    rm -f /tmp/python311.tgz
    pushd /tmp/Python-${PY_VER}
    ./configure --prefix="${PY_PREFIX}" --with-ensurepip=install
    make -j"$(nproc)"
    make install
    popd
    rm -rf /tmp/Python-${PY_VER}
  fi
  if [ -x "${PY_PREFIX}/bin/python3.11" ]; then
    ln -sf "${PY_PREFIX}/bin/python3.11" /usr/local/bin/python3.11
  fi
}

ensure_python311

mkdir -p "$MODELS_DIR" "$WHISPER_DIR" "$PIPER_DIR"

rm -rf "$VENV_DIR"
python3.11 -m venv "$VENV_DIR"
"$VENV_DIR/bin/pip" install --upgrade pip wheel
"$VENV_DIR/bin/pip" install -r "$APP_DIR/requirements.txt"

# Install Coqui TTS (Python 3.11 compatible)
echo "Installing Coqui TTS..."
"$VENV_DIR/bin/pip" install TTS==0.22.0 || {
  echo "Coqui TTS install failed. Check network or PyPI mirrors.";
  exit 1;
}


# Install Piper binary (CLI) so we don't depend on pip wheels
PIPER_VERSION="1.2.0"
ARCH="$(uname -m)"
case "$ARCH" in
  x86_64|amd64)
    PIPER_ARCH="amd64"
    ;;
  aarch64|arm64)
    PIPER_ARCH="arm64"
    ;;
  armv7l|armv6l)
    PIPER_ARCH="armhf"
    ;;
  *)
    echo "Unsupported architecture for Piper: $ARCH"
    exit 1
    ;;
esac

PIPER_URL="https://github.com/rhasspy/piper/releases/download/v${PIPER_VERSION}/piper_${PIPER_ARCH}.tar.gz"
PIPER_FALLBACK_URL="https://ghproxy.com/https://github.com/rhasspy/piper/releases/download/v${PIPER_VERSION}/piper_${PIPER_ARCH}.tar.gz"
PIPER_INSTALL_DIR="/opt"
mkdir -p "$PIPER_INSTALL_DIR"
PIPER_TAR="/tmp/piper_${PIPER_ARCH}.tar.gz"
if ! curl -fL "$PIPER_URL" -o "$PIPER_TAR"; then
  echo "Direct GitHub download failed, trying proxy..."
  curl -fL "$PIPER_FALLBACK_URL" -o "$PIPER_TAR"
fi
tar -xz -C "$PIPER_INSTALL_DIR" -f "$PIPER_TAR"
rm -f "$PIPER_TAR"
if [ -f "/opt/piper/piper" ]; then
  ln -sf /opt/piper/piper /usr/local/bin/piper
fi
if ! command -v piper >/dev/null 2>&1; then
  echo "Piper binary not found after install."
  exit 1
fi

# Download Faster-Whisper model (multilingual small) to /models/whisper/small
"$VENV_DIR/bin/python" - <<'PY'
from huggingface_hub import snapshot_download
import os

model_id = "Systran/faster-whisper-small"
local_dir = "/models/whisper/small"
os.makedirs(local_dir, exist_ok=True)
print(f"Downloading {model_id} to {local_dir}...")
snapshot_download(repo_id=model_id, local_dir=local_dir, local_dir_use_symlinks=False)
print("Whisper model download complete.")
PY

# Download default Piper voice (English)
PIPER_VOICE_NAME="en_US-lessac-medium"
PIPER_VOICE_BASE="https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium"

curl -L -o "$PIPER_DIR/${PIPER_VOICE_NAME}.onnx" "$PIPER_VOICE_BASE/${PIPER_VOICE_NAME}.onnx"
curl -L -o "$PIPER_DIR/${PIPER_VOICE_NAME}.onnx.json" "$PIPER_VOICE_BASE/${PIPER_VOICE_NAME}.onnx.json"
if [ ! -s "$PIPER_DIR/${PIPER_VOICE_NAME}.onnx" ] || [ ! -s "$PIPER_DIR/${PIPER_VOICE_NAME}.onnx.json" ]; then
  echo "Failed to download Piper voice model files."
  exit 1
fi

# Install systemd service
SERVICE_PATH="/etc/systemd/system/videogen.service"
cat <<'SERVICE' > "$SERVICE_PATH"
[Unit]
Description=Video Generator Service
After=network.target

[Service]
Type=simple
WorkingDirectory=/workspace/videomaker
Environment=PYTHONUNBUFFERED=1
Environment=MODELS_DIR=/models
ExecStart=/workspace/videomaker/.venv/bin/uvicorn main:app --host 0.0.0.0 --port 8065 --workers 1
Restart=on-failure
RestartSec=2

[Install]
WantedBy=multi-user.target
SERVICE

systemctl daemon-reload
systemctl enable --now videogen.service

echo "Installation complete. Service is running on port 8065."
