@echo off
cd /d C:\Users\EDY\Desktop\StockPrediction
set NO_PROXY=*
echo [%date% %time%] pipeline driver start >> logs\pipeline_driver.log
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\research_pipeline.ps1 >> logs\pipeline_driver.log 2>&1
echo [%date% %time%] pipeline driver end >> logs\pipeline_driver.log
