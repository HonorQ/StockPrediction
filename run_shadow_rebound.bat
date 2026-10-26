@echo off
chcp 65001 >nul
cd /d C:\Users\EDY\Desktop\StockPrediction

REM 72.5 oversold-rebound shadow tracker: daily close snapshot + picks + forward returns
REM Called daily 16:40 by scheduled task StockPrediction_ShadowRebound
REM Independent from production pipeline. Idempotent (delete+insert per day).
REM NOTE: this file MUST keep CRLF line endings. LF makes cmd.exe split the
REM redirect line apart and the task exits with code 255.

echo [%date% %time%] Shadow rebound track started...

python -u scripts\shadow_rebound_track.py >> logs\shadow_rebound.log 2>&1
set RC=%ERRORLEVEL%

echo [%date% %time%] Shadow rebound track done with exit code %RC%
echo.

REM Pass --no-pause when invoked by the scheduler
if /i not "%~1"=="--no-pause" pause

exit /b %RC%
