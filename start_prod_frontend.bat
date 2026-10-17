@echo off
echo.
echo  [PROD FRONTEND] — Connecting to http://localhost:8000
echo  Opens on http://localhost:5173
echo.
cd frontend
npm run dev -- --port 5173
