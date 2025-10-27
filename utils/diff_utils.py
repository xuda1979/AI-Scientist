"""
Diff-based paper revision utilities.

This module provides functions to apply git-style unified diffs to LaTeX papers,
enabling targeted revisions instead of full paper regeneration.
"""

import re
from typing import List, Tuple, Optional
from pathlib import Path


def extract_diff_blocks(llm_output: str) -> List[str]:
    """
    Extract all diff blocks from LLM output.
    
    Supports multiple formats:
    - ```diff ... ```
    - --- a/paper.tex ... +++ b/paper.tex ...
    
    Args:
        llm_output: Raw output from LLM
        
    Returns:
        List of diff block strings
    """
    # Try markdown code blocks first
    diff_blocks = re.findall(
        r'```diff\n(.*?)```',
        llm_output,
        re.DOTALL | re.MULTILINE
    )
    
    if diff_blocks:
        return diff_blocks
    
    # Try raw diff format
    if '--- a/paper.tex' in llm_output or '--- paper.tex' in llm_output:
        return [llm_output]
    
    return []


def parse_unified_diff_hunk(hunk_lines: List[str]) -> Optional[Tuple[int, int, List[str], List[str]]]:
    """
    Parse a single unified diff hunk.
    
    Args:
        hunk_lines: Lines of the hunk including header
        
    Returns:
        Tuple of (old_start_line, old_line_count, removed_lines, added_lines) or None
    """
    if not hunk_lines:
        return None
    
    # Parse hunk header: @@ -120,7 +120,9 @@
    header_match = re.match(r'@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@', hunk_lines[0])
    if not header_match:
        return None
    
    old_start = int(header_match.group(1))
    old_count = int(header_match.group(2)) if header_match.group(2) else 1
    
    removed = []
    added = []
    
    for line in hunk_lines[1:]:
        if line.startswith('-') and not line.startswith('---'):
            removed.append(line[1:])
        elif line.startswith('+') and not line.startswith('+++'):
            added.append(line[1:])
    
    return (old_start, old_count, removed, added)


def apply_diff_to_content(original_content: str, diff_text: str) -> Tuple[str, bool, str]:
    """
    Apply unified diff to content.
    
    Args:
        original_content: Original file content
        diff_text: Unified diff to apply
        
    Returns:
        Tuple of (modified_content, success, error_message)
    """
    try:
        diff_blocks = extract_diff_blocks(diff_text)
        
        if not diff_blocks:
            # No diff format found - check if it's full content
            if '\\documentclass' in diff_text and '\\end{document}' in diff_text:
                print("⚠️  No diff markers found - treating as full paper replacement")
                return diff_text, True, "Full content replacement (no diff format)"
            else:
                return original_content, False, "No valid diff blocks or full content found"
        
        modified_content = original_content
        
        for diff_block in diff_blocks:
            result, success, msg = _apply_single_diff_block(modified_content, diff_block)
            if success:
                modified_content = result
            else:
                print(f"⚠️  Failed to apply diff block: {msg}")
                # Continue with other blocks
        
        return modified_content, True, f"Successfully applied {len(diff_blocks)} diff block(s)"
        
    except Exception as e:
        return original_content, False, f"Error applying diff: {str(e)}"


def _apply_single_diff_block(content: str, diff_block: str) -> Tuple[str, bool, str]:
    """
    Apply a single diff block to content.
    
    Uses a simple line-by-line matching approach with context verification.
    
    Args:
        content: Current content
        diff_block: Single diff block to apply
        
    Returns:
        Tuple of (modified_content, success, message)
    """
    lines = content.split('\n')
    diff_lines = diff_block.split('\n')
    
    # Find hunks in the diff
    hunks = []
    current_hunk = []
    
    for line in diff_lines:
        if line.startswith('@@'):
            if current_hunk:
                hunks.append(current_hunk)
            current_hunk = [line]
        elif current_hunk:
            current_hunk.append(line)
    
    if current_hunk:
        hunks.append(current_hunk)
    
    if not hunks:
        return content, False, "No hunks found in diff block"
    
    # Apply hunks in reverse order (to maintain line numbers)
    for hunk in reversed(hunks):
        parse_result = parse_unified_diff_hunk(hunk)
        if not parse_result:
            continue
        
        old_start, old_count, removed, added = parse_result
        
        # Verify context and apply changes
        start_idx = old_start - 1  # Convert to 0-indexed
        
        if start_idx < 0 or start_idx >= len(lines):
            continue
        
        # Find exact match location (with fuzzy matching for slight variations)
        match_idx = _find_best_match(lines, removed, start_idx, window=10)
        
        if match_idx is None:
            print(f"⚠️  Could not find exact match for hunk at line {old_start}")
            continue
        
        # Apply the change
        end_idx = match_idx + len(removed)
        lines[match_idx:end_idx] = added
    
    return '\n'.join(lines), True, "Diff applied successfully"


def _find_best_match(lines: List[str], target: List[str], start_hint: int, window: int = 10) -> Optional[int]:
    """
    Find the best match for target lines in the content.
    
    Args:
        lines: Content lines
        target: Lines to find
        start_hint: Suggested starting position
        window: Search window around hint
        
    Returns:
        Index of best match or None
    """
    if not target:
        return start_hint
    
    # Try exact match at hint first
    if start_hint + len(target) <= len(lines):
        if lines[start_hint:start_hint + len(target)] == target:
            return start_hint
    
    # Search in window around hint
    search_start = max(0, start_hint - window)
    search_end = min(len(lines), start_hint + window)
    
    for i in range(search_start, search_end):
        if i + len(target) > len(lines):
            break
        if lines[i:i + len(target)] == target:
            return i
    
    # Try fuzzy matching (strip whitespace)
    target_stripped = [line.strip() for line in target]
    for i in range(search_start, search_end):
        if i + len(target) > len(lines):
            break
        if [line.strip() for line in lines[i:i + len(target)]] == target_stripped:
            return i
    
    return None


def is_diff_format(content: str) -> bool:
    """
    Check if content is in diff format (vs full paper content).
    
    Args:
        content: Content to check
        
    Returns:
        True if content appears to be a diff
    """
    diff_indicators = [
        '```diff',
        '--- a/',
        '--- paper.tex',
        '--- simulation.py',
        '+++ b/',
        '+++ paper.tex',
        '+++ simulation.py',
        re.compile(r'@@ -\d+,\d+ \+\d+,\d+ @@'),
    ]
    
    for indicator in diff_indicators:
        if isinstance(indicator, str):
            if indicator in content:
                return True
        else:  # regex pattern
            if indicator.search(content):
                return True
    
    return False


def extract_file_diffs(llm_output: str) -> dict:
    """
    Extract diffs for multiple files from LLM output.
    
    Args:
        llm_output: Raw output from LLM containing diffs for multiple files
        
    Returns:
        Dictionary mapping filenames to their diff content
        Example: {'paper.tex': 'diff content...', 'simulation.py': 'diff content...'}
    """
    file_diffs = {}
    
    # Pattern to match file headers in unified diff format
    # Matches: --- a/paper.tex or --- paper.tex or --- a/simulation.py
    file_pattern = re.compile(
        r'(?:^|\n)---\s+(?:a/)?([^\n\s]+)\s*\n\+\+\+\s+(?:b/)?([^\n\s]+)',
        re.MULTILINE
    )
    
    matches = list(file_pattern.finditer(llm_output))
    
    if not matches:
        # No file headers found - might be single file diff in code block
        diff_blocks = extract_diff_blocks(llm_output)
        if diff_blocks:
            # Assume it's for paper.tex if no filename specified
            file_diffs['paper.tex'] = llm_output
        return file_diffs
    
    for i, match in enumerate(matches):
        filename_a = match.group(1).strip()
        filename_b = match.group(2).strip()
        
        # Use the 'b' filename (after modification) as canonical
        filename = filename_b if filename_b else filename_a
        
        # Normalize filename (remove quotes, paths, etc.)
        if '/' in filename:
            filename = filename.split('/')[-1]
        
        # Extract diff content for this file
        start_pos = match.start()
        end_pos = matches[i + 1].start() if i + 1 < len(matches) else len(llm_output)
        
        diff_content = llm_output[start_pos:end_pos].strip()
        file_diffs[filename] = diff_content
    
    return file_diffs


def apply_diffs_to_files(file_contents: dict, llm_output: str) -> Tuple[dict, bool, str]:
    """
    Apply diffs to multiple files.
    
    Args:
        file_contents: Dictionary mapping filenames to their current content
                      Example: {'paper.tex': 'current content...', 'simulation.py': 'current code...'}
        llm_output: Raw LLM output containing diffs
        
    Returns:
        Tuple of (modified_files_dict, success, message)
        modified_files_dict: Dictionary of filenames to modified content
    """
    file_diffs = extract_file_diffs(llm_output)
    
    if not file_diffs:
        return {}, False, "No file diffs found in LLM output"
    
    modified_files = {}
    errors = []
    successes = []
    
    for filename, diff_content in file_diffs.items():
        # Get original content for this file
        original_content = file_contents.get(filename, "")
        
        if not original_content:
            errors.append(f"No original content provided for {filename}")
            continue
        
        # Apply diff to this file
        modified_content, success, msg = apply_diff_to_content(original_content, diff_content)
        
        if success:
            modified_files[filename] = modified_content
            successes.append(f"{filename}: {msg}")
        else:
            errors.append(f"{filename}: {msg}")
    
    overall_success = len(modified_files) > 0
    
    if overall_success:
        status_msg = f"Applied diffs to {len(modified_files)} file(s): {', '.join(modified_files.keys())}"
        if errors:
            status_msg += f" | Errors: {'; '.join(errors)}"
        return modified_files, True, status_msg
    else:
        return {}, False, f"Failed to apply any diffs. Errors: {'; '.join(errors)}"


def create_diff_prompt_suffix() -> str:
    """
    Get the prompt suffix for requesting diff output.
    
    Returns:
        Prompt text requesting diff format
    """
    return """
OUTPUT FORMAT - UNIFIED DIFF ONLY:

Instead of providing the complete revised files, provide unified diffs (patches) that show only the changes needed.

CRITICAL: Even if you have code execution capabilities and run simulations on your server side, 
you MUST STILL provide the complete code file changes in diff format so we can save them locally.

Format your response as:

```diff
--- a/paper.tex
+++ b/paper.tex
@@ -120,7 +120,10 @@
 \\section{Introduction}
 
-Large language models have shown impressive capabilities.
+Large language models have demonstrated remarkable capabilities in various domains,
+including natural language understanding, generation, and reasoning tasks.
+Recent advances have particularly focused on improving their reliability and
+factual accuracy through various prompting and fine-tuning strategies.
 
 \\subsection{Background}
```

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
     for trial in range(params['num_trials']):
```

RULES:
- Provide ONLY the minimal diffs needed to address the review feedback
- Include 3-5 lines of context before and after each change for accurate matching
- Use unified diff format with @@ line numbers
- Multiple diff blocks are allowed for changes in different sections
- Preserve all existing content unless specifically modifying it
- DO NOT regenerate entire files - only show the changes
- If you modify simulation.py or other code files, MUST provide those diffs too
- MANDATORY: Deliver all code file changes even if you executed code on your server

IMPORTANT FOR CODE EXECUTION:
Even if you have the ability to run Python code on your server and generate results,
you MUST STILL provide the complete simulation.py file changes in diff format.
We need the code files saved locally for reproducibility and version control.
Do not assume we can access code you ran on your server - deliver all code changes explicitly.

If changes are extensive across many sections or files, you may provide multiple diff blocks.
"""
