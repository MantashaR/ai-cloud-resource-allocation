@echo off
title AI Cloud Resource Allocation (Simple)
cd /d "%~dp0"
echo.
echo  Installing requirements (first run only, needs internet)...
python -m pip install -q -r requirements.txt
echo.
echo  Starting dashboard... your browser will open automatically.
echo  Keep this window open. Close it to stop the app.
echo.
python -m streamlit run app_v2.py --server.port 8502 --server.headless false --browser.gatherUsageStats false
pause
