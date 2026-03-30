#!/usr/bin/env bash
set -euo pipefail

if ! command -v apt-get >/dev/null 2>&1; then
  echo "This bootstrap script expects an apt-based image." >&2
  exit 1
fi

if [[ "$(id -u)" -ne 0 ]]; then
  echo "bootstrap_host.sh must be run as root on the JarvisLabs instance." >&2
  exit 1
fi

apt-get update

CACHE_PACKAGE="redis-server"
if apt-cache show valkey >/dev/null 2>&1; then
  CACHE_PACKAGE="valkey"
fi

apt-get install -y \
  curl \
  ffmpeg \
  libgl1 \
  poppler-utils \
  python3-venv \
  postgresql \
  postgresql-client \
  "$CACHE_PACKAGE"

echo "Host dependencies installed (cache package: $CACHE_PACKAGE)."
