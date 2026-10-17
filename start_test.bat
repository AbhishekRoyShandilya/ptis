@echo off
echo.
echo  ████████╗███████╗███████╗████████╗
echo  ╚══██╔══╝██╔════╝██╔════╝╚══██╔══╝
echo     ██║   █████╗  ███████╗   ██║
echo     ██║   ██╔══╝  ╚════██║   ██║
echo     ██║   ███████╗███████║   ██║
echo     ╚═╝   ╚══════╝╚══════╝   ╚═╝
echo.
echo  [TEST MODE] — DB: ptis_test  Port: 8001
echo  Frontend: run start_test_frontend.bat in a second terminal
echo  To stop:  Ctrl+C
echo.
echo  WARNING: This is the TEST environment.
echo  Your real trades are in the PROD environment (start_prod.bat).
echo.

set PTIS_ENV=test
rem ── Proactive partner: watches your positions + top watchlist symbols, auto-alerts on money moves.
rem     Runtime flags (read via os.getenv, NOT pydantic Settings — must NOT go in .env.test).
rem     Set PTIS_PROACTIVE_LOOP=false to disable.
set PTIS_PROACTIVE_LOOP=true
set PTIS_PROACTIVE_MAX=10
set PTIS_PROACTIVE_INTERVAL=300
.venv\Scripts\uvicorn.exe api.main:app --reload --host 127.0.0.1 --port 8001
