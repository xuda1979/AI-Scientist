#!/bin/bash

echo "AI Scientist VS Code GUI Launcher"
echo "=================================="
echo ""

cd "$(dirname "$0")/ai-scientist-vscode-gui"

if [ ! -d "node_modules" ]; then
    echo "Installing dependencies..."
    npm install
    if [ $? -ne 0 ]; then
        echo "Failed to install dependencies"
        exit 1
    fi
fi

echo "Compiling TypeScript..."
npm run compile
if [ $? -ne 0 ]; then
    echo "Failed to compile"
    exit 1
fi

echo ""
echo "Starting AI Scientist GUI..."
echo ""
npm start
