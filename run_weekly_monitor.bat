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
REM 计划任务调用时传 --no-pause: 无人交互时 pause 会挂死直到任务超时被强制结束
if /i not "%~1"=="--no-pause" pause
