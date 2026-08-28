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
pause
