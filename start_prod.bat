@echo off
echo.
echo  ██████╗ ████████╗██╗███████╗
echo  ██╔══██╗╚══██╔══╝██║██╔════╝
echo  ██████╔╝   ██║   ██║███████╗
echo  ██╔═══╝    ██║   ██║╚════██║
echo  ██║        ██║   ██║███████║
echo  ╚═╝        ╚═╝   ╚═╝╚══════╝
echo.
echo  [PRODUCTION MODE] — DB: ptis  Port: 8000
echo  Frontend: run start_prod_frontend.bat in a second terminal
echo  To stop:  Ctrl+C
echo.
echo  This is the REAL environment with your actual trades.
echo.

set PTIS_ENV=prod
.venv\Scripts\uvicorn.exe api.main:app --reload --host 127.0.0.1 --port 8000
