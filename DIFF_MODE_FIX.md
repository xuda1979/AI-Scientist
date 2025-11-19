# Diff Mode Fix: Preventing Truncated Papers

## Problem Identified

When using `--modify-existing` with long papers, pages were missing from the generated output. 

**Root Cause:**
1. The system requested **diff format** from the LLM when modifying existing papers
2. Some LLMs (like gpt-5-pro) **ignored the diff instruction** and returned **full LaTeX content** instead
3. The LLM **hit its output token limit** (typically 4K-16K tokens) before completing the entire paper
4. The **truncated LaTeX** was used as-is, resulting in missing pages/sections

## Solution Implemented

Modified `sciresearch_workflow.py` to:

### 1. **File-Based Diff Mode Detection** (Lines 1791-1801)

**Before:**
```python
use_diff_mode = not is_initial_draft
```

**After:**
```python
# Check if paper.tex actually exists
paper_tex_path = project_dir / "paper.tex"
use_diff_mode = paper_tex_path.exists()

if use_diff_mode:
    print(f"       Using DIFF mode (paper.tex exists)")
else:
    print(f"       Using FULL mode (paper.tex does not exist)")
```

**Benefits:**
- More reliable detection based on actual file existence
- Clear logging of which mode is being used
- Follows the principle: "unless there is no paper.tex file, we always use git diff"

### 2. **Truncation Detection & Protection** (Lines 1851-1868)

**Before:**
```python
else:
    # Fallback: If no diff detected in diff mode, assume full content
    print(f"       ⚠ Expected diff format but got full content, using as-is")
    candidate = candidate_response
```

**After:**
```python
else:
    # Fallback: If no diff detected in diff mode, LLM returned full content
    print(f"       ⚠ WARNING: Expected diff format but got full content!")
    print(f"       ⚠ This may be truncated due to LLM output token limits.")
    print(f"       ⚠ Response length: {len(candidate_response)} chars")
    
    # Check if response looks complete (has \end{document})
    if "\\end{document}" in candidate_response:
        print(f"       → Full content appears complete, using as-is")
        candidate = candidate_response
    else:
        print(f"       → Full content appears TRUNCATED (no \\end{{document}})")
        print(f"       → Keeping original paper to avoid data loss")
        candidate = current_tex  # Don't use truncated content!
```

**Benefits:**
- **Detects truncation** by checking for `\end{document}`
- **Prevents data loss** by rejecting truncated papers
- **Clear warnings** help users understand what's happening
- **Falls back to original** rather than corrupting the paper

## How It Works Now

### Scenario 1: New Paper (no paper.tex)
```
1. paper.tex doesn't exist → use_diff_mode = False
2. System requests FULL paper from LLM
3. LLM generates complete paper
4. Paper is saved
```

### Scenario 2: Existing Paper + LLM Obeys Diff Request ✅
```
1. paper.tex exists → use_diff_mode = True
2. System requests DIFF/PATCH from LLM
3. LLM returns unified diff
4. System applies diff to existing paper
5. Modified paper is saved
```

### Scenario 3: Existing Paper + LLM Ignores Diff Request (FIXED!) 🔧
```
1. paper.tex exists → use_diff_mode = True
2. System requests DIFF/PATCH from LLM
3. LLM ignores instruction, returns full LaTeX
4. System detects it's not a diff format
5. ⚠️ WARNING: Expected diff format but got full content
6. Check if content is complete:
   - Has \end{document}? → Use as-is ✓
   - No \end{document}? → TRUNCATED! Keep original ✓
```

## Testing the Fix

Run your command again:
```powershell
python main.py --modify-existing --output-dir .\output\black_hole\ --model gpt-5-pro --max-iterations 1 --enable-pdf-review
```

**What to look for in the output:**
- ✅ `Using DIFF mode (paper.tex exists)` - Confirms diff mode is active
- ✅ `Diff format detected, applying patch...` - LLM cooperated with diff request
- ⚠️ `Expected diff format but got full content!` - LLM returned full content instead
  - Then: `Full content appears complete` ✓ or
  - Then: `Full content appears TRUNCATED - Keeping original` ✓

## Why This Works

1. **Diff mode avoids token limits**: A diff/patch is typically much smaller than the full paper, so LLMs can complete it within their output token limit

2. **Truncation detection prevents data loss**: By checking for `\end{document}`, we ensure we never save an incomplete paper

3. **Graceful fallback**: If the LLM can't follow diff instructions but returns complete content, we still use it

4. **Clear diagnostics**: Detailed logging helps you understand what's happening and why

## Additional Recommendations

### For Better Diff Mode Compliance

If you continue to see "Expected diff format but got full content" warnings, you can:

1. **Use a different model** that better follows diff instructions (e.g., gpt-4-turbo, claude-3-opus)

2. **Check the diff_utils.py** to ensure the diff detection regex is comprehensive

3. **Increase request_timeout** to give the LLM more time to think

### For Very Long Papers

If your paper is extremely long (>100 pages), consider:

1. **Breaking into multiple iterations** with smaller changes per iteration
2. **Using --max-iterations 3** or higher to make incremental improvements
3. **Manually specifying changes** with more focused user prompts

## Verification

To verify the fix is working:

1. Check that paper.tex is complete with `\end{document}`
2. Count pages in the PDF: should match or exceed original
3. Check terminal output for truncation warnings
4. Compare file sizes before/after revision

## Summary

✅ **Fixed**: Papers are no longer truncated when LLMs return full content instead of diffs  
✅ **Protected**: System detects truncation and preserves original paper  
✅ **Improved**: Clear logging and diagnostics  
✅ **Reliable**: File-based diff mode detection instead of flag-based  

The system now intelligently handles all scenarios and protects your papers from data loss.
