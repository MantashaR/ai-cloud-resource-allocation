@echo off
title AI Cloud Resource Allocation
cd /d "%~dp0"
echo.
echo  ==========================================================
echo   AI-Driven Cloud Resource Allocation
echo   Installing requirements (first run only, needs internet)
echo  ==========================================================
python -m pip install -q -r requirements.txt
echo.
echo  Starting dashboard... your browser will open automatically.
echo  Keep this window open. Close it to stop the app.
echo.
python -m streamlit run app.py --server.headless false --browser.gatherUsageStats false
pause
