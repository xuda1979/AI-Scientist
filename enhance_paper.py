#!/usr/bin/env python3
"""
Script to enhance the Access Point Selection Precoding paper using the workflow.

Usage:
    python enhance_paper.py              # Full enhancement workflow
    python enhance_paper.py --science-only  # Simple mode: improve content + git diff
"""
import sys
from pathlib import Path
from sciresearch_workflow import run_workflow

def main():
    """Enhance the existing paper with comprehensive improvements."""
    
    # Check for --science-only flag
    science_only = "--science-only" in sys.argv
    
    output_dir = Path("output/Access_Point_Selection_Precoding")
    
    if science_only:
        print("="*80)
        print("SCIENCE-ONLY MODE - Fast Scientific Improvement")
        print("="*80)
        print(f"Output directory: {output_dir}")
        print(f"Model: gpt-4o")
        print(f"Mode: science_only=True")
        print("="*80)
        print()
        print("Simplified workflow:")
        print("  1. Read current paper.tex")
        print("  2. Send prompt: 'Improve the scientific content'")
        print("  3. Request git diff output")
        print("  4. Save response and extracted diff")
        print()
        print("No validation, no iterations, no quality checks - just fast improvements!")
        print("="*80)
        print()
        
        try:
            project_dir = run_workflow(
                topic="Cell-Free Massive MIMO",
                field="Wireless Communications",
                question="QUBO optimization for AP selection",
                output_dir=output_dir,
                model="gpt-4o",
                science_only=True,
                modify_existing=True,
            )
            
            print("\n" + "="*80)
            print("SCIENCE-ONLY MODE COMPLETED")
            print("="*80)
            print(f"Response saved to: {project_dir / 'science_only_response.txt'}")
            
            diff_file = project_dir / "science_only.diff"
            if diff_file.exists():
                print(f"Diff extracted to: {diff_file}")
                print("\nTo apply the diff:")
                print(f"  cd {project_dir}")
                print(f"  git apply science_only.diff")
            
            print("="*80)
            
        except Exception as e:
            print(f"\n{'='*80}")
            print(f"ERROR: Science-only mode failed")
            print(f"{'='*80}")
            print(f"{e}")
            import traceback
            traceback.print_exc()
        
        return
    
    # FULL ENHANCEMENT MODE (original behavior)
    print("="*80)
    print("ENHANCING PAPER: Cell-Free Massive MIMO via Ising")
    print("="*80)
    print(f"Output directory: {output_dir}")
    print(f"Model: gpt-4o (latest available)")
    print(f"Max iterations: 5")
    print(f"Mode: modify_existing=True")
    print("="*80)
    print()
    
    # User prompt with specific requirements
    user_prompt = """
Transform this short paper into a comprehensive, publication-ready research paper:

CRITICAL REQUIREMENTS:
1. ADD 15-20 AUTHENTIC REFERENCES: Include real published papers in cell-free MIMO, 
   QUBO optimization, Ising machines, and wireless communications. Use \\begin{filecontents*}{refs.bib}
   at the TOP of paper.tex with proper BibTeX entries.

2. EXPAND TO 5000-8000 WORDS: Add detailed content to each section:
   - Introduction: 800-1000 words with motivation and contributions
   - Related Work: 1000-1500 words with critical comparison of 15+ papers
   - System Model: 800-1000 words with detailed problem formulation
   - QUBO Formulation: 1000-1200 words with mathematical derivations
   - Solution Method: 600-800 words on CIM implementation
   - Simulation Results: 1000-1500 words with detailed analysis
   - Conclusion: 400-600 words

3. ADD MATHEMATICAL RIGOR:
   - Detailed derivations of QUBO formulation
   - Complexity analysis with Big-O notation
   - Convergence analysis or theoretical guarantees
   - Step-by-step proofs where applicable

4. ENHANCE SIMULATION SECTION:
   - Add tables showing numerical results
   - Include multiple scenarios and parameter variations
   - Provide statistical analysis and confidence intervals
   - Compare with baseline methods

5. MAINTAIN TECHNICAL QUALITY:
   - Keep all existing equations and enhance them
   - Add more mathematical formulations
   - Ensure all claims are properly cited
   - Use professional academic writing style
"""
    
    # Run the workflow
    try:
        project_dir = run_workflow(
            topic="Cell-Free Massive MIMO with Ising-Based Optimization",
            field="Wireless Communications",
            question="How can we jointly optimize access point selection and precoding in cell-free massive MIMO using QUBO formulations solved by coherent Ising machines?",
            output_dir=output_dir,
            model="gpt-4o",
            max_iterations=5,
            modify_existing=True,
            user_prompt=user_prompt,
        )
        
        print("\n" + "="*80)
        print("WORKFLOW COMPLETED SUCCESSFULLY")
        print("="*80)
        print(f"Enhanced paper saved to: {project_dir / 'paper.tex'}")
        print(f"PDF available at: {project_dir / 'paper.pdf'}")
        print("="*80)
        
    except Exception as e:
        print(f"\n{'='*80}")
        print(f"ERROR: Workflow failed")
        print(f"{'='*80}")
        print(f"{e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
