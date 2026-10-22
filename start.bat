@echo off
cd /d "%~dp0"
where ffmpeg >nul 2>nul
if errorlevel 1 (echo Please install FFmpeg and add it to PATH. & pause & exit /b 1)
where ffprobe >nul 2>nul
if errorlevel 1 (echo Please install FFprobe and add it to PATH. & pause & exit /b 1)
py -3 -m venv .venv
if errorlevel 1 exit /b 1
.venv\Scripts\python -m pip install -r requirements.txt
if errorlevel 1 exit /b 1
.venv\Scripts\python -m uvicorn app:app --host 127.0.0.1 --port 8000
pause
