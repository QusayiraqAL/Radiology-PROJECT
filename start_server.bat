@echo off
chcp 65001 >nul
set PYTHONUTF8=1
title AI Radiology Hub - API Server
cd /d "%~dp0api"
echo ============================================
echo   AI Radiology Hub - Real Prediction API
echo   http://127.0.0.1:8000
echo ============================================
"%~dp0api\venv\Scripts\python.exe" -m uvicorn main:app --host 127.0.0.1 --port 8000
pause
