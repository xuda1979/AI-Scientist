"""
Standalone script to run science-only mode on the Access Point Selection paper.
This bypasses all validation and just asks GPT-4o to improve scientific content.
"""
from pathlib import Path
from sciresearch_workflow import run_workflow

if __name__ == "__main__":
    print("=" * 80)
    print("SCIENCE-ONLY MODE - Access Point Selection Paper")
    print("=" * 80)
    print("\nThis will:")
    print("1. Send paper to GPT-4o with simple prompt: 'Improve the scientific content'")
    print("2. Get back improved content as git diff")
    print("3. Save to science_only_response.txt and science_only.diff")
    print("\nPlease wait 30-60 seconds for API response...")
    print("=" * 80)
    print()
    
    run_workflow(
        topic='Cell-Free Massive MIMO',
        field='Wireless Communications',
        question='QUBO optimization',
        output_dir=Path('output/Access_Point_Selection_Precoding'),
        model='gpt-4o',
        science_only=True,
        modify_existing=True
    )
    
    print("\n" + "=" * 80)
    print("SCIENCE-ONLY MODE COMPLETE")
    print("=" * 80)
    print("\nCheck output files:")
    print("- output/Access_Point_Selection_Precoding/science_only_response.txt")
    print("- output/Access_Point_Selection_Precoding/science_only.diff")
    print("\nTo apply the changes, run:")
    print("  cd output/Access_Point_Selection_Precoding")
    print("  git apply science_only.diff")
    print("=" * 80)
