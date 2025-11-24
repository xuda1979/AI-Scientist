#!/usr/bin/env python3
import argparse
import sys
import subprocess
import shlex
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description="Wrapper to regenerate all tables and figures.")
    parser.add_argument("--emit-figure-index", action="store_true", help="Emit a JSON index of generated figures.")
    parser.add_argument("--demo-mode", action="store_true", help="Run in demo mode.")
    args = parser.parse_args()

    cmd = [sys.executable, "simulation.py"]
    if args.emit_figure_index:
        cmd.append("--emit-figure-index")
    if args.demo_mode:
        cmd.append("--demo-mode")

    print(f"+ {' '.join(shlex.quote(c) for c in cmd)}")
    subprocess.check_call(cmd)
    print("Done: generated datasets.")

if __name__ == "__main__":
    main()
