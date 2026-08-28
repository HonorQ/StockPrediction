@echo off
chcp 65001 >nul
cd /d C:\Users\EDY\Desktop\StockPrediction

echo [%date% %time%] 开始每周监控...

echo.
echo === 1. 运行每周IC监控 ===
python scripts\weekly_ic_monitor.py

echo.
echo === 2. 运行每周健康检查 ===
python scripts\weekly_health_check.py

echo.
echo [%date% %time%] 每周监控完成!
echo.
pause
