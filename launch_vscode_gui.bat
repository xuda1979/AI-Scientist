@echo off
REM VSCode-Style GUI Launcher
echo ========================================
echo AI Scientist - VSCode Style Interface
echo ========================================
echo.
echo Starting modern VSCode-style GUI...
echo.

cd /d "%~dp0"
python -m ui.vscode_style_gui

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo ERROR: Failed to launch GUI
    echo.
    pause
)
