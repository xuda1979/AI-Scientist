#!/usr/bin/env python3
"""
Example script demonstrating the --science-only flag.

This simplified mode:
1. Reads the existing paper
2. Sends a simple prompt: "Improve the scientific content of the paper. Then output git diff code."
3. Displays the response
4. Extracts and saves any git diff code

No validation, no iterations, no quality checks - just fast scientific improvements.
"""

from pathlib import Path
from sciresearch_workflow import run_workflow


def main():
    """Run science-only mode on the Access Point Selection paper."""
    
    output_dir = Path("output/Access_Point_Selection_Precoding")
    
    print("="*80)
    print("SCIENCE-ONLY MODE - Fast Scientific Content Improvement")
    print("="*80)
    print(f"Paper: {output_dir / 'paper.tex'}")
    print(f"Model: gpt-4o")
    print(f"Mode: science_only=True (simplified workflow)")
    print("="*80)
    print()
    print("This mode will:")
    print("  1. Read the current paper")
    print("  2. Send simple prompt: 'Improve the scientific content'")
    print("  3. Request git diff output")
    print("  4. Display and save the response")
    print()
    print("No validation, no iterations, no quality checks.")
    print("="*80)
    print()
    
    try:
        project_dir = run_workflow(
            topic="Cell-Free Massive MIMO",
            field="Wireless Communications", 
            question="QUBO optimization for access point selection",
            output_dir=output_dir,
            model="gpt-4o",
            science_only=True,  # Enable science-only mode
            modify_existing=True,
        )
        
        print("\n" + "="*80)
        print("SCIENCE-ONLY MODE COMPLETED")
        print("="*80)
        print(f"Results saved to: {project_dir}")
        print(f"Response file: {project_dir / 'science_only_response.txt'}")
        
        # Check if diff was extracted
        diff_file = project_dir / "science_only.diff"
        if diff_file.exists():
            print(f"Diff file: {diff_file}")
            print("\nTo apply the diff:")
            print(f"  cd {project_dir}")
            print(f"  git apply science_only.diff")
        
        print("="*80)
        
    except Exception as e:
        print(f"\n{'='*80}")
        print(f"ERROR in science-only mode")
        print(f"{'='*80}")
        print(f"{e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
