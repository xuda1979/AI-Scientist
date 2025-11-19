#!/usr/bin/env python3
"""
Script to fix papers that have no references after many rounds of modification.
This addresses the root cause: content protection rejecting changes and diff mode issues.
"""
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from sciresearch_workflow import run_workflow, WorkflowConfig

def fix_paper_with_references(paper_dir: str, model: str = "gpt-4o", max_iterations: int = 3):
    """
    Fix a paper that has missing or incomplete references.
    
    This uses special settings to ensure changes are actually applied:
    - Disables content protection temporarily
    - Uses explicit instructions about references
    - Ensures LaTeX auto-fix is enabled
    """
    output_dir = Path(paper_dir)
    
    if not output_dir.exists():
        print(f"Error: Directory {paper_dir} does not exist!")
        return
    
    # Create a custom configuration that ensures changes are applied
    config = WorkflowConfig()
    
    # CRITICAL FIX #1: Disable content protection temporarily
    # This was preventing revisions from being applied
    config.content_protection = False
    config.enable_content_protection = False
    
    # CRITICAL FIX #2: Enable LaTeX auto-fix
    config.latex_auto_fix = True
    
    # CRITICAL FIX #3: Lower quality threshold to allow iterations
    config.quality_threshold = 0.5
    
    # CRITICAL FIX #4: Enable reference validation
    config.reference_validation = True
    
    # CRITICAL FIX #5: Set model
    config.default_model = model
    config.review_model = model
    config.revision_model = model
    
    print("="*80)
    print("🔧 PAPER FIX MODE - Adding References and Improving Content")
    print("="*80)
    print(f"📁 Paper directory: {paper_dir}")
    print(f"🤖 Model: {model}")
    print(f"🔄 Max iterations: {max_iterations}")
    print(f"⚙️  Content protection: DISABLED (to ensure changes apply)")
    print(f"⚙️  LaTeX auto-fix: ENABLED")
    print(f"⚙️  Reference validation: ENABLED")
    print("="*80)
    print()
    
    # Check current paper state
    paper_path = output_dir / "paper.tex"
    if paper_path.exists():
        content = paper_path.read_text(encoding="utf-8", errors="ignore")
        import re
        ref_count = len(re.findall(r'\\bibitem\{|@\w+\{', content))
        word_count = len(re.findall(r'\b\w+\b', content)) // 2
        print(f"📊 CURRENT PAPER STATE:")
        print(f"   - Approximate word count: {word_count}")
        print(f"   - Reference count: {ref_count}")
        print()
        
        if ref_count < 10:
            print(f"⚠️  WARNING: Only {ref_count} references found!")
            print(f"   This script will add comprehensive references.\n")
    
    # User prompt that emphasizes references
    user_prompt = """
CRITICAL REQUIREMENTS FOR THIS REVISION:

1. REFERENCES (TOP PRIORITY):
   - Add AT LEAST 20 high-quality, authentic references
   - Use recent publications (2018-2025) mixed with foundational works
   - Include proper bibliographic details (authors, title, journal, year, DOI)
   - Embed references using filecontents or thebibliography environment
   - Cite references throughout the paper where appropriate
   - Ensure all citations have corresponding bibliography entries

2. CONTENT QUALITY:
   - Expand methodology section with technical details
   - Add experimental validation and results
   - Include proper mathematical formulations
   - Add figures or tables if simulation data is available

3. STRUCTURE:
   - Ensure all standard sections are present
   - Follow academic paper conventions
   - Use proper LaTeX formatting

DO NOT reduce content length or remove existing material.
DO NOT use placeholder or fake references.
ALL references must be real, published works.
"""
    
    try:
        # Run the workflow with custom config
        result = run_workflow(
            topic="",  # Will be extracted from existing paper
            field="",  # Will be extracted from existing paper
            question="",  # Will be extracted from existing paper
            output_dir=output_dir,
            model=model,
            request_timeout=600,  # 10 minute timeout
            max_iterations=max_iterations,
            modify_existing=True,
            strict_singletons=False,
            quality_threshold=config.quality_threshold,
            check_references=True,
            validate_figures=True,
            user_prompt=user_prompt,
            config=config,
            enable_ideation=False,  # Don't re-ideate for existing papers
        )
        
        print("\n" + "="*80)
        print("✅ PAPER FIX COMPLETED SUCCESSFULLY!")
        print("="*80)
        print(f"📁 Result directory: {result}")
        
        # Check final state
        if paper_path.exists():
            content = paper_path.read_text(encoding="utf-8", errors="ignore")
            ref_count = len(re.findall(r'\\bibitem\{|@\w+\{', content))
            word_count = len(re.findall(r'\b\w+\b', content)) // 2
            print(f"\n📊 FINAL PAPER STATE:")
            print(f"   - Approximate word count: {word_count}")
            print(f"   - Reference count: {ref_count}")
            
            if ref_count >= 15:
                print(f"\n✅ SUCCESS: Paper now has {ref_count} references (target achieved)!")
            else:
                print(f"\n⚠️  NOTE: Paper has {ref_count} references (target: 15+)")
                print(f"   You may want to run this script again to add more references.")
        
        print("="*80)
        
    except Exception as e:
        print("\n" + "="*80)
        print(f"❌ ERROR: {e}")
        print("="*80)
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Fix papers with missing references")
    parser.add_argument("paper_dir", help="Path to paper directory (e.g., output/Access_Point_Selection_Precoding)")
    parser.add_argument("--model", default="gpt-4o", help="AI model to use (default: gpt-4o)")
    parser.add_argument("--max-iterations", type=int, default=3, help="Maximum iterations (default: 3)")
    
    args = parser.parse_args()
    
    fix_paper_with_references(args.paper_dir, args.model, args.max_iterations)
