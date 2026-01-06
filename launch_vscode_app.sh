#!/bin/bash
cd "$(dirname "$0")"
export PYTHONPATH=$PYTHONPATH:$(pwd)
python3 ui/vscode_app.py