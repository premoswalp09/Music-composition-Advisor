@echo off
REM Start Backend Server
start cmd /k "cd music-composition-advisor-backend && npm start"

REM Start Frontend Server
start cmd /k "cd music-composition-advisor-frontend && npm start"

REM Wait for user input to close
pause
