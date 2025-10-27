# Code Execution Delivery Fix

## Problem Statement

When LLMs have code execution capabilities (like running Python on their server side), they might execute `simulation.py` to generate results but fail to deliver the actual code file changes back to us. This creates a critical issue:

1. **Missing Code Files**: Even if the LLM runs code and gets results, we need the actual code files saved locally for:
   - Version control (git tracking)
   - Reproducibility (re-running experiments)
   - Code review and validation
   - Local execution and debugging

2. **Previous Implementation Gap**: The diff-based revision system only prompted for `paper.tex` changes and didn't explicitly request `simulation.py` or other code file diffs.

## Solution Implemented

### 1. Updated Diff Prompt Template (`utils/diff_utils.py`)

Enhanced `create_diff_prompt_suffix()` to:

```python
- Explicitly request diffs for ALL files including simulation.py
- Add clear warning: "Even if you have code execution capabilities and run simulations on your server side, you MUST STILL provide the complete code file changes in diff format"
- Include example of simulation.py diff format
- Emphasize mandatory delivery: "MANDATORY: Deliver all code file changes even if you executed code on your server"
```

**Key additions to prompt:**
```
CRITICAL: Even if you have code execution capabilities and run simulations on your server side, 
you MUST STILL provide the complete code file changes in diff format so we can save them locally.

If you need to modify simulation.py or any other code files, provide diffs for them as well:

```diff
--- a/simulation.py
+++ b/simulation.py
@@ -45,6 +45,8 @@
 def run_experiment(params):
     # Existing code
+    # Add validation step
+    validate_parameters(params)
     results = []
```

IMPORTANT FOR CODE EXECUTION:
Even if you have the ability to run Python code on your server and generate results,
you MUST STILL provide the complete simulation.py file changes in diff format.
We need the code files saved locally for reproducibility and version control.
Do not assume we can access code you ran on your server - deliver all code changes explicitly.
```

### 2. Multi-File Diff Handling (`utils/diff_utils.py`)

Added new functions to handle diffs for multiple files:

#### `extract_file_diffs(llm_output: str) -> dict`
- Parses LLM output to identify diffs for different files
- Returns dictionary: `{'paper.tex': 'diff...', 'simulation.py': 'diff...'}`
- Handles multiple file header formats:
  - `--- a/paper.tex` / `+++ b/paper.tex`
  - `--- simulation.py` / `+++ simulation.py`

#### `apply_diffs_to_files(file_contents: dict, llm_output: str) -> Tuple[dict, bool, str]`
- Applies diffs to multiple files simultaneously
- Takes dictionary of current file contents
- Returns dictionary of modified file contents
- Provides detailed success/error messages per file

#### Updated `is_diff_format(content: str) -> bool`
- Enhanced to detect diffs for any file (not just `paper.tex`)
- Added patterns: `--- a/`, `--- simulation.py`, `+++ simulation.py`

### 3. Updated Review/Revision Workflow (`workflow_steps/review_revision.py`)

Modified diff application logic to handle multiple files:

**Before:**
```python
# Only handled paper.tex
from utils.diff_utils import is_diff_format, apply_diff_to_content

if use_diff_mode and is_diff_format(revised):
    revised_content, success, msg = apply_diff_to_content(current_tex, revised)
    if success:
        revised = revised_content
```

**After:**
```python
# Handles both paper.tex and simulation.py
from utils.diff_utils import is_diff_format, apply_diffs_to_files

if use_diff_mode and is_diff_format(revised):
    # Prepare file contents dictionary
    file_contents = {'paper.tex': current_tex}
    
    # Add simulation.py if it exists
    sim_path = project_dir / "simulation.py"
    if sim_path.exists():
        file_contents['simulation.py'] = sim_path.read_text(encoding='utf-8', errors='ignore')
    
    # Apply diffs to all files
    modified_files, success, msg = apply_diffs_to_files(file_contents, revised)
    
    if success and 'paper.tex' in modified_files:
        revised = modified_files['paper.tex']
        print(f"    ✓ Diff applied to paper.tex: {msg}")
        
        # Apply simulation.py changes if present
        if 'simulation.py' in modified_files:
            sim_path.write_text(modified_files['simulation.py'], encoding='utf-8')
            print(f"    ✓ Applied diff to simulation.py")
```

## Benefits

1. **Guaranteed Code Delivery**: LLM must provide code file changes regardless of server-side execution
2. **Multi-File Support**: Seamlessly handles diffs for paper.tex, simulation.py, and any other files
3. **Backward Compatible**: Still works with single-file diffs or full content responses
4. **Clear Messaging**: Explicit instructions prevent LLM from omitting code files
5. **Local Reproducibility**: All code changes are saved locally for git tracking and re-execution

## Files Modified

1. **utils/diff_utils.py**
   - Updated `create_diff_prompt_suffix()` with multi-file instructions
   - Added `extract_file_diffs()` function
   - Added `apply_diffs_to_files()` function
   - Updated `is_diff_format()` to detect any file diffs

2. **workflow_steps/review_revision.py**
   - Replaced single-file diff logic with multi-file handling
   - Added simulation.py diff application
   - Enhanced status messages for multi-file updates

## Testing Recommendations

1. **Test with code execution LLMs** (e.g., ChatGPT with code interpreter):
   - Verify they provide simulation.py diffs even after running code
   - Check that both paper.tex and simulation.py diffs are applied correctly

2. **Test with non-code-execution LLMs**:
   - Ensure backward compatibility with single-file diffs
   - Verify graceful fallback to full content mode if needed

3. **Test edge cases**:
   - Multiple diff blocks in single response
   - Mixed format (some files as diffs, some as full content)
   - Missing simulation.py (should not cause errors)

## Example LLM Response (Expected Format)

```diff
--- a/paper.tex
+++ b/paper.tex
@@ -150,5 +150,8 @@
 \section{Results}

-Our experiments show promising results.
+Our experiments demonstrate significant improvements over baseline methods.
+Figure \ref{fig:results} shows the performance comparison across different
+datasets, with our approach achieving 15\% higher accuracy on average.

--- a/simulation.py
+++ b/simulation.py
@@ -23,4 +23,10 @@
 def run_experiments():
     results = []
+    
+    # Add comprehensive logging
+    logging.info(f"Starting experiments with {len(datasets)} datasets")
+    
+    # Run with multiple seeds for statistical significance
     for dataset in datasets:
+        for seed in [42, 123, 456]:
+            results.append(evaluate(dataset, seed))
     return results
```

## Conclusion

This fix ensures that even when LLMs execute code on their servers, they still deliver the complete code file changes in diff format for local storage, version control, and reproducibility. The implementation is backward compatible and provides clear, actionable feedback during the revision process.
