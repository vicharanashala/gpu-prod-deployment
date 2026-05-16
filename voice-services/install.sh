#!/usr/bin/env bash
# System packages for voice-services (STT webm/mp3 → WAV via ffmpeg, HTTPS to Sarvam).
set -euo pipefail

if [[ "${EUID:-0}" -ne 0 ]] && ! command -v sudo >/dev/null 2>&1; then
  echo "Run as root or install sudo to use this script." >&2
  exit 1
fi

run() {
  if [[ "${EUID:-0}" -eq 0 ]]; then
    "$@"
  else
    sudo "$@"
  fi
}

install_apt() {
  run apt-get update -y
  run apt-get install -y --no-install-recommends \
    ffmpeg \
    ca-certificates \
    curl
}

install_dnf() {
  run dnf install -y ffmpeg ca-certificates curl
}

install_apk() {
  run apk add --no-cache ffmpeg ca-certificates curl
}

if command -v apt-get >/dev/null 2>&1; then
  install_apt
elif command -v dnf >/dev/null 2>&1; then
  install_dnf
elif command -v apk >/dev/null 2>&1; then
  install_apk
elif command -v brew >/dev/null 2>&1; then
  echo "Using Homebrew (macOS). Install ffmpeg if missing:"
  brew install ffmpeg
else
  echo "Unsupported OS: install ffmpeg and ca-certificates manually, then re-run." >&2
  exit 1
fi

command -v ffmpeg >/dev/null 2>&1 && ffmpeg -version | head -1
echo "System dependencies OK."
