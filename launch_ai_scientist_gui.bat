@echo off
echo AI Scientist VS Code GUI Launcher
echo ==================================
echo.

cd /d "%~dp0ai-scientist-vscode-gui"

if not exist "node_modules" (
    echo Installing dependencies...
    call npm install
    if errorlevel 1 (
        echo Failed to install dependencies
        pause
        exit /b 1
    )
)

echo Compiling TypeScript...
call npm run compile
if errorlevel 1 (
    echo Failed to compile
    pause
    exit /b 1
)

echo.
echo Starting AI Scientist GUI...
echo.
call npm start
