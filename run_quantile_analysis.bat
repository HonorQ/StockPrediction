@echo off
chcp 65001 >nul
cd /d C:\Users\EDY\Desktop\StockPrediction

echo [%date% %time%] 开始十分位分析...

echo.
echo === 运行十分位分析 ===
python scripts\quantile_analysis.py

echo.
echo [%date% %time%] 十分位分析完成!
echo.
REM 计划任务调用时传 --no-pause: 无人交互时 pause 会挂死直到任务超时被强制结束
if /i not "%~1"=="--no-pause" pause
