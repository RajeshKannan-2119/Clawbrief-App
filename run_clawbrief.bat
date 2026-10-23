@echo off
title ClawBrief Personal AI Launcher
cd /d "%~dp0"
echo Starting ClawBrief Streamlit Dashboard...
python -m streamlit run clawbrief_app.py
if %errorlevel% neq 0 (
    py -m streamlit run clawbrief_app.py
)
pause
