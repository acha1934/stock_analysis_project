@echo off
title Stock Agent Launcher
echo ============================================
echo         Stock Agent - Startup
echo ============================================
echo.

cd /d "%~dp0application"

echo [1/2] Starting FastAPI backend on port 8000...
start "Stock Agent - Backend (keep open)" cmd /k "python -m uvicorn main:app --port 8000"

echo     Waiting 20 seconds for backend to initialize...
timeout /t 20 /nobreak > nul

echo [2/2] Starting Streamlit frontend on port 8501...
start "Stock Agent - Frontend (keep open)" cmd /k "python -m streamlit run app.py --server.port 8501"

echo     Waiting 8 seconds for frontend to initialize...
timeout /t 8 /nobreak > nul

echo.
echo ============================================
echo  Both servers are running!
echo  Backend:  http://localhost:8000
echo  Frontend: http://localhost:8501
echo ============================================
echo.
echo  Opening browser...
start http://localhost:8501

echo  You can close this window.
echo  Keep the other two terminal windows OPEN.
timeout /t 5 /nobreak > nul
