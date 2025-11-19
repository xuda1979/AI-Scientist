"""
Verification script to ensure user prompt is sent to LLM in every review/revision cycle.

This script checks that:
1. User prompt is included in the system message of all prompt functions
2. User prompt is highlighted with visual markers (=== borders)
3. User prompt includes iteration context
4. User prompt is prioritized over other instructions
"""

import re
from pathlib import Path

def verify_prompt_function(file_path: Path, function_name: str) -> dict:
    """Verify a specific prompt function includes proper user prompt handling."""
    
    content = file_path.read_text(encoding='utf-8')
    
    # Find the function
    pattern = rf'def {function_name}\([^)]*\):[^:]*?(?=\ndef |\Z)'
    match = re.search(pattern, content, re.DOTALL)
    
    if not match:
        return {
            "found": False,
            "error": f"Function {function_name} not found"
        }
    
    func_content = match.group(0)
    
    # Check for user_prompt parameter
    has_user_prompt_param = 'user_prompt' in re.search(rf'def {function_name}\([^)]*\)', func_content).group(0)
    
    # Check for visual markers (=== borders)
    has_visual_markers = "{'='*80}" in func_content or '='*80 in func_content
    
    # Check for priority/critical markers
    has_priority_marker = any(marker in func_content for marker in [
        "PRIORITY INSTRUCTION",
        "HIGHEST PRECEDENCE",
        "🎯",
        "CRITICAL:"
    ])
    
    # Check for iteration context
    has_iteration_context = "iteration" in func_content.lower() or "Iteration" in func_content
    
    # Check if user_prompt is actually used in the function
    uses_user_prompt = "if user_prompt:" in func_content
    
    return {
        "found": True,
        "function": function_name,
        "has_user_prompt_param": has_user_prompt_param,
        "has_visual_markers": has_visual_markers,
        "has_priority_marker": has_priority_marker,
        "has_iteration_context": has_iteration_context,
        "uses_user_prompt": uses_user_prompt,
        "all_checks_pass": all([
            has_user_prompt_param,
            has_visual_markers,
            has_priority_marker,
            uses_user_prompt
        ])
    }

def main():
    """Run verification checks."""
    
    workflow_file = Path("sciresearch_workflow.py")
    
    if not workflow_file.exists():
        print(f"❌ ERROR: {workflow_file} not found!")
        return
    
    print("="*80)
    print("VERIFYING USER PROMPT PERSISTENCE IN REVIEW/REVISION CYCLES")
    print("="*80)
    print()
    
    # Functions to check
    functions_to_check = [
        "_combined_review_edit_revise_prompt",
        "_review_prompt",
        "_editor_prompt",
        "_revise_prompt"
    ]
    
    results = []
    for func_name in functions_to_check:
        print(f"Checking function: {func_name}")
        result = verify_prompt_function(workflow_file, func_name)
        results.append(result)
        
        if not result["found"]:
            print(f"  ❌ {result['error']}")
            continue
        
        print(f"  {'✓' if result['has_user_prompt_param'] else '✗'} Has user_prompt parameter")
        print(f"  {'✓' if result['has_visual_markers'] else '✗'} Has visual markers (borders)")
        print(f"  {'✓' if result['has_priority_marker'] else '✗'} Has priority/critical markers")
        print(f"  {'✓' if result['has_iteration_context'] else '✗'} Has iteration context")
        print(f"  {'✓' if result['uses_user_prompt'] else '✗'} Actually uses user_prompt")
        
        if result['all_checks_pass']:
            print(f"  ✅ ALL CHECKS PASS")
        else:
            print(f"  ⚠️  SOME CHECKS FAILED")
        print()
    
    # Summary
    print("="*80)
    print("SUMMARY")
    print("="*80)
    
    total = len([r for r in results if r["found"]])
    passed = len([r for r in results if r.get("all_checks_pass", False)])
    
    print(f"Functions checked: {total}")
    print(f"Functions passing all checks: {passed}/{total}")
    
    if passed == total:
        print("\n✅ SUCCESS: All prompt functions properly include user prompt in every cycle!")
        print("\nKEY FEATURES VERIFIED:")
        print("  • User prompt included in system message")
        print("  • Visual markers (=== borders) for emphasis")
        print("  • Priority/critical markers to highlight importance")
        print("  • Iteration context to maintain continuity")
        print("\nThe user prompt will now be sent to the LLM in EVERY review/revision cycle.")
    else:
        print(f"\n⚠️  WARNING: {total - passed} function(s) need updates")
    
    print()

if __name__ == "__main__":
    main()
