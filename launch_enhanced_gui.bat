@echo off
REM Launcher script for the Enhanced AI Scientist GUI
REM This script launches the enhanced GUI with all features

echo ========================================
echo AI Scientist - Enhanced GUI Launcher
echo ========================================
echo.
echo Starting the enhanced workflow GUI...
echo.

cd /d "%~dp0"
python -m ui.enhanced_gui

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo ERROR: Failed to launch the GUI
    echo Please make sure all dependencies are installed:
    echo   pip install -r requirements.txt
    echo.
    pause
)
