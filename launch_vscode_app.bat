@echo off
setlocal
cd /d "%~dp0"
set PYTHONPATH=%PYTHONPATH%;%CD%
python ui/vscode_app.py
pause