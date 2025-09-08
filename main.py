#!/usr/bin/env python3
"""
SciResearch Workflow - Main Entry Point

This is the main entry point that delegates to the working workflow implementation.
"""
import sys
import subprocess
from pathlib import Path


def main() -> None:
    """Main entry point that delegates to the working workflow."""
    # Get the path to the working workflow script
    workflow_script = Path(__file__).parent / "sciresearch_workflow.py"
    
    # Pass all arguments to the working workflow
    cmd = [sys.executable, str(workflow_script)] + sys.argv[1:]
    
    # Run the working workflow
    try:
        result = subprocess.run(cmd, check=False)
        sys.exit(result.returncode)
    except KeyboardInterrupt:
        print("\nWorkflow interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"Error running workflow: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
