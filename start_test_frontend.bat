@echo off
echo.
echo  [TEST FRONTEND] — Connecting to http://localhost:8001
echo  Opens on http://localhost:5174
echo.
cd frontend
npm run dev -- --mode test --port 5174
