@echo off
chcp 65001 >nul
cd /d C:\Users\EDY\Desktop\StockPrediction

echo [%date% %time%] 开始盘前模式...

python run_daily.py --premarket

echo.
echo [%date% %time%] 盘前模式完成!
