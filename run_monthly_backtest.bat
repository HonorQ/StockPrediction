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
pause
