@echo off
chcp 65001 >nul
cd /d C:\Users\EDY\Desktop\StockPrediction

echo [%date% %time%] 开始月度回测验证...

echo.
echo === 运行月度回测 ===
python scripts\monthly_backtest.py --top-n 30 --hold 5

echo.
echo [%date% %time%] 月度回测完成!
echo.
REM 计划任务调用时传 --no-pause: 无人交互时 pause 会挂死直到任务超时被强制结束
if /i not "%~1"=="--no-pause" pause
