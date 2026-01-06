#!/usr/bin/env python3
"""
Launch the AI Scientist Research GUI.

This GUI provides:
- Auto Mode: Fully autonomous research agent that completes the entire workflow
- Interactive Mode: User can input prompts and control iterations
- Iteration Control: Specify the number of research/revision iterations
- Real-time Progress: See agent progress and outputs in real-time
- Chat Interface: Interactive chat with AI research assistant
"""
import sys
from pathlib import Path

# Ensure the parent directory is in the path
sys.path.insert(0, str(Path(__file__).parent))

from ui.vscode_style_gui import main

if __name__ == "__main__":
    main()
