#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
command -v ffmpeg >/dev/null || { echo 'FFmpeg를 먼저 설치해 주세요.'; exit 1; }
command -v ffprobe >/dev/null || { echo 'FFprobe를 먼저 설치해 주세요.'; exit 1; }
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m uvicorn app:app --host 127.0.0.1 --port 8000
