#!/bin/bash
# Launcher script for the Enhanced AI Scientist GUI
# This script launches the enhanced GUI with all features

echo "========================================"
echo "AI Scientist - Enhanced GUI Launcher"
echo "========================================"
echo ""
echo "Starting the enhanced workflow GUI..."
echo ""

cd "$(dirname "$0")"
python -m ui.enhanced_gui

if [ $? -ne 0 ]; then
    echo ""
    echo "ERROR: Failed to launch the GUI"
    echo "Please make sure all dependencies are installed:"
    echo "  pip install -r requirements.txt"
    echo ""
    read -p "Press Enter to exit..."
fi
