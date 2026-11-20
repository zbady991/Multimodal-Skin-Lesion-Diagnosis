@echo off
echo ====================================================
echo Starting Multimodal Skin Lesion Diagnosis Live Demo
echo ====================================================
cd /d "%~dp0\.."
call .venv\Scripts\activate.bat
python demo\app.py
pause
