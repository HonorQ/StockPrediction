@echo off
chcp 65001 >nul
cd /d C:\Users\EDY\Desktop\StockPrediction

REM Kline backfill retry: refills the klines cache missed by fetch_1year_data
REM Called every 4 hours by scheduled task StockPrediction_BackfillKlines
REM Idempotent: cached stocks are skipped; bails out within 30s if baostock down
REM NOTE: this file MUST keep CRLF line endings. LF makes cmd.exe split the
REM redirect line apart and the task exits with code 255.

echo [%date% %time%] Kline backfill started...

python -u scripts\backfill_missing_klines.py >> logs\backfill_klines.log 2>&1
set RC=%ERRORLEVEL%

echo [%date% %time%] Kline backfill done with exit code %RC% and cached file count:
dir /b cache\klines\1year_*.pkl 2>nul | find /c /v ""

echo.
REM Pass --no-pause when invoked by the scheduler: pause would hang an
REM unattended shell until the execution time limit kills the task
if /i not "%~1"=="--no-pause" pause

REM Return python's exit code instead of letting find's code leak out
exit /b %RC%
