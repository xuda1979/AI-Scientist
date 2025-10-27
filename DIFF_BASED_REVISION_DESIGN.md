# Critical Design Flaw - Full Paper Regeneration Issue

## Problem Identified

The AI Scientist workflow has a **CRITICAL DESIGN FLAW** where it asks the LLM to regenerate the **ENTIRE PAPER** on every modification iteration, instead of generating targeted diffs/patches.

### Current Broken Behavior

**Location:** `sciresearch_workflow.py` line 3426  
**Function:** `_revise_prompt()`

```python
sys_prompt = (
    "You are the paper author making revisions based on peer review. Your goal is to address ALL reviewer concerns "
    "while maintaining scientific integrity and clarity. Produce a COMPLETE revised LaTeX file.\n\n"  # ❌ THIS IS THE PROBLEM
```

### Why This Is Catastrophic

1. **Token Waste**: Regenerating entire papers (10K+ lines) for minor changes
2. **Truncation**: Papers get cut off mid-section when hitting token limits
3. **Data Loss**: All backups are truncated versions from failed full regenerations  
4. **Cost**: Massively expensive - generating 50K+ tokens to change a few lines
5. **Reliability**: High failure rate due to model context/output limits

### Example Impact

- **Initial paper**: 771 lines
- **Minor review feedback**: "Add more statistical tests"
- **Current approach**: Regenerate all 771 lines (hits 16000 token limit at line 770) ❌
- **Correct approach**: Generate 5-line diff to add tests ✅

## Solution Design

### 1. Diff-Based Revision System

The workflow should work like git:
- **First iteration**: Generate complete paper (necessary)
- **All subsequent iterations**: Generate **ONLY diffs/patches**
- Apply diffs automatically to existing file

### 2. Implementation Plan

#### Step 1: Modify `_revise_prompt()` to request diffs

```python
def _revise_prompt(..., is_initial_draft: bool = False):
    if is_initial_draft:
        # Original behavior for initial generation
        sys_prompt = "Produce a COMPLETE revised LaTeX file..."
    else:
        # NEW: Request diffs only
        sys_prompt = (
            "You are the paper author making revisions based on peer review. "
            "Your goal is to address reviewer concerns with TARGETED CHANGES.\n\n"
            
            "OUTPUT FORMAT - GIT-STYLE DIFFS ONLY:\n"
            "Produce unified diff format patches that can be applied to the existing paper.\n"
            "Format:\n"
            "```diff\n"
            "--- a/paper.tex\n"
            "+++ b/paper.tex\n"
            "@@ -120,7 +120,9 @@\n"
            " existing line 1\n"
            " existing line 2\n"
            "-line to remove\n"
            "+new line to add\n"
            "+another new line\n"
            " existing line 3\n"
            "```\n\n"
            
            "RULES:\n"
            "- Generate ONLY the minimal diff needed to address review feedback\n"
            "- Include sufficient context lines (±3-5 lines) for accurate patching\n"
            "- Multiple diffs allowed for changes in different sections\n"
            "- Preserve all existing content unless specifically removing it\n"
            "- DO NOT regenerate the entire paper\n"
        )
```

#### Step 2: Create diff application function

```python
def _apply_diff_to_paper(original_content: str, diff_text: str) -> str:
    """
    Apply git-style unified diff to paper content.
    
    Args:
        original_content: Current paper.tex content
        diff_text: Unified diff patch from LLM
    
    Returns:
        Patched paper content
    """
    import re
    from difflib import unified_diff
    
    # Parse diff blocks
    diff_blocks = re.findall(
        r'```diff\n(.*?)```',
        diff_text,
        re.DOTALL
    )
    
    if not diff_blocks:
        # Fallback: If no diff format, treat as full content
        print("⚠ Warning: No diff format detected, using full content")
        return diff_text
    
    modified_content = original_content
    
    for diff_block in diff_blocks:
        # Apply each diff block using patch-like logic
        modified_content = _apply_single_diff(modified_content, diff_block)
    
    return modified_content

def _apply_single_diff(content: str, diff: str) -> str:
    """Apply a single unified diff block."""
    lines = content.split('\n')
    diff_lines = diff.split('\n')
    
    # Parse hunk header: @@ -120,7 +120,9 @@
    hunk_match = re.search(r'@@ -(\d+),(\d+) \+(\d+),(\d+) @@', diff)
    if not hunk_match:
        return content
    
    old_start = int(hunk_match.group(1)) - 1  # 0-indexed
    
    # Extract changes
    removals = []
    additions = []
    context = []
    
    for line in diff_lines:
        if line.startswith('-') and not line.startswith('---'):
            removals.append(line[1:])
        elif line.startswith('+') and not line.startswith('+++'):
            additions.append(line[1:])
        elif line.startswith(' '):
            context.append(line[1:])
    
    # Apply changes
    # ... (implement proper patch application logic)
    
    return '\n'.join(lines)
```

#### Step 3: Modify revision workflow

```python
def _generate_best_revision_candidate(..., is_initial_draft: bool = False):
    """Generate revision with diff-based approach."""
    
    if is_initial_draft:
        # First iteration: generate complete paper
        prompt = _initial_draft_prompt(...)
        revised_content = _universal_chat(prompt, ...)
    else:
        # Subsequent iterations: generate diffs only
        prompt = _revise_prompt(..., is_initial_draft=False)
        diff_text = _universal_chat(prompt, ...)
        
        # Apply diffs to existing content
        revised_content = _apply_diff_to_paper(current_tex, diff_text)
    
    return revised_content
```

### 3. Benefits

✅ **Massive token savings**: 100-1000x reduction in output tokens  
✅ **No truncation**: Diffs are tiny (< 1000 tokens vs 50000 tokens)  
✅ **Faster**: Less generation time, quicker iterations  
✅ **Cheaper**: Dramatically reduced API costs  
✅ **More reliable**: No mid-paper cutoffs  
✅ **Better quality**: Model focuses on specific improvements  
✅ **Git-compatible**: Standard diff format for version control  

### 4. Migration Path

1. ✅ Remove token limits (already done)
2. ⚠️ Implement diff-based revision system (TO DO)
3. ⚠️ Add fallback to full regeneration if diff fails (TO DO)
4. ⚠️ Test with existing papers (TO DO)
5. ⚠️ Update documentation (TO DO)

## Immediate Actions Required

1. **Implement diff-based revision** - This is the correct long-term fix
2. **Keep token limit removal** - Still beneficial for initial generation
3. **Regenerate deliberative_compute** - Using fixed workflow

## Files to Modify

- `sciresearch_workflow.py`:
  - `_revise_prompt()` - Add diff output mode
  - `_generate_best_revision_candidate()` - Add diff application
  - New functions: `_apply_diff_to_paper()`, `_apply_single_diff()`

## Estimated Impact

- **Before**: 50K tokens output per revision × 10 iterations = 500K tokens
- **After**: 2K tokens output per revision × 10 iterations = 20K tokens
- **Savings**: 96% reduction in output tokens, 25x cost reduction

## Priority

**CRITICAL - HIGH PRIORITY**

This should be implemented immediately as it affects:
- System reliability
- Cost efficiency
- User experience
- Paper quality

---

**Date**: October 26, 2025  
**Status**: Design complete, implementation pending
