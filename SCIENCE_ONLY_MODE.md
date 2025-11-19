# Science-Only Mode Documentation

## Overview

The `--science-only` flag provides a **simplified, fast workflow** for improving scientific content in papers. It bypasses all validation, quality checks, and iterative refinement, sending a single focused prompt to the LLM.

## Purpose

**When to use science-only mode:**
- You want quick scientific improvements without lengthy iterations
- You trust the LLM to improve content without validation
- You want git diff output to review changes before applying
- You need fast turnaround for minor improvements
- You're prototyping or experimenting with content changes

**When NOT to use science-only mode:**
- You need comprehensive paper generation from scratch
- You want quality validation and enforcement
- You need iterative refinement with multiple rounds
- You require reference validation and content protection
- You want automatic application of changes

## How It Works

### Normal Workflow (without --science-only)
```
1. Load paper ✓
2. Run simulation ✓
3. Compile LaTeX ✓
4. Quality validation (66 checks) ✓
5. Content protection ✓
6. Reference validation ✓
7. Experimental rigor checks ✓
8. Generate review ✓
9. Editorial decision ✓
10. Generate revision ✓
11. Validate revision (regression detection) ✓
12. Apply changes with guardian protection ✓
13. Repeat for max_iterations (default: 5)
```

**Time:** 5-30 minutes per iteration, multiple iterations

### Science-Only Workflow (with --science-only)
```
1. Load paper ✓
2. Send simple prompt to LLM ✓
3. Receive response ✓
4. Extract and save diff ✓
5. Done!
```

**Time:** 30 seconds to 2 minutes (single API call)

## Prompt Sent to LLM

The exact prompt sent in science-only mode is:

```
System: You are a scientific writing assistant. Improve the scientific content of the paper.

User: Improve the scientific content of the paper. Then output git diff code showing the changes.

Current paper:
[paper.tex content]
```

That's it! No complex instructions, no validation requirements, no iteration logic.

## Usage Examples

### Method 1: Command Line Flag

```bash
python enhance_paper.py --science-only
```

### Method 2: Python API

```python
from pathlib import Path
from sciresearch_workflow import run_workflow

run_workflow(
    topic="Your Topic",
    field="Your Field",
    question="Your Question",
    output_dir=Path("output/YourPaper"),
    model="gpt-4o",
    science_only=True,      # ← Enable science-only mode
    modify_existing=True,
)
```

### Method 3: Dedicated Script

```bash
python science_only_mode_example.py
```

## Output Files

Science-only mode creates:

1. **`science_only_response.txt`**
   - Full LLM response
   - Contains improved content and/or git diff

2. **`science_only.diff`** (if diff detected)
   - Extracted git diff code
   - Can be applied with `git apply`

## Applying Changes

Science-only mode does NOT automatically modify your paper. You must review and apply changes manually:

### Option 1: Apply Git Diff

```bash
cd output/YourPaper
git apply science_only.diff
```

### Option 2: Manual Review

```bash
# View the response
cat output/YourPaper/science_only_response.txt

# Manually copy/paste improvements to paper.tex
```

### Option 3: Extract and Apply Selectively

```bash
# Extract specific sections from response
grep -A 20 "\\section{Introduction}" science_only_response.txt

# Apply only parts you want
```

## Comparison: Full vs Science-Only Mode

| Feature | Full Workflow | Science-Only |
|---------|--------------|--------------|
| **Time** | 20-150 minutes | 1-2 minutes |
| **Iterations** | 4-5 (default) | 1 (single call) |
| **Validation** | 66+ quality checks | None |
| **Content Protection** | Multi-layer guardian | None |
| **Reference Validation** | Enforced (15-20 required) | None |
| **Regression Detection** | Rejects bad changes | None |
| **Auto-apply Changes** | Yes (with validation) | No (manual review) |
| **Output Format** | Updated paper.tex | Git diff + response file |
| **Safety** | High (multiple protections) | Low (manual review needed) |
| **Best For** | Production papers | Quick experiments |

## Advanced Usage

### Custom Prompt in Science-Only Mode

Currently, science-only mode uses a fixed simple prompt. To customize:

```python
# Modify sciresearch_workflow.py lines 4310-4320
# Change the prompt content to your needs
```

### Combine with Other Flags

```python
run_workflow(
    topic="Topic",
    field="Field",
    question="Question",
    output_dir=Path("output/paper"),
    model="gpt-4o",
    science_only=True,        # Fast mode
    request_timeout=120,      # Shorter timeout
)
```

### Use Different Models

```python
# Use faster/cheaper model for quick experiments
run_workflow(
    ...,
    model="gpt-3.5-turbo",  # Faster, cheaper
    science_only=True,
)
```

## Error Handling

Science-only mode has minimal error handling:

```python
try:
    run_workflow(..., science_only=True)
except Exception as e:
    print(f"Error: {e}")
    # Response is still saved if LLM call succeeded
```

**Common errors:**
- **No paper.tex found:** Science-only requires existing paper
- **API timeout:** Reduce `request_timeout` or try again
- **No diff in response:** LLM didn't output git diff format
  - Check `science_only_response.txt` for content
  - May need to manually extract improvements

## Limitations

1. **No Quality Guarantees**
   - LLM may introduce errors
   - No validation of scientific accuracy
   - No reference checking

2. **Single Pass Only**
   - No iterative refinement
   - No progressive improvement
   - Quality depends on single LLM call

3. **Manual Review Required**
   - Changes are not auto-applied
   - You must review before using
   - Risk of introducing problems

4. **No Content Protection**
   - Could delete sections
   - Could remove references
   - Could truncate content
   - No regression detection

5. **Format Dependent**
   - Relies on LLM outputting proper git diff
   - May not always generate diff format
   - Might return full content instead

## Best Practices

✅ **DO:**
- Use for quick experiments and prototypes
- Review all changes before applying
- Keep backups of your paper
- Use git version control
- Test with smaller papers first
- Verify scientific accuracy manually

❌ **DON'T:**
- Use for final production papers without review
- Trust the output blindly
- Skip manual validation
- Apply diffs without checking
- Use on papers without backups
- Expect the same quality as full workflow

## Example Session

```bash
$ python enhance_paper.py --science-only

================================================================================
SCIENCE-ONLY MODE - Fast Scientific Improvement
================================================================================
Output directory: output/Access_Point_Selection_Precoding
Model: gpt-4o
Mode: science_only=True
================================================================================

Simplified workflow:
  1. Read current paper.tex
  2. Send prompt: 'Improve the scientific content'
  3. Request git diff output
  4. Save response and extracted diff

No validation, no iterations, no quality checks - just fast improvements!
================================================================================

Sending request to gpt-4o...
Paper size: 3934 characters

================================================================================
SCIENCE-ONLY MODE RESPONSE
================================================================================
[LLM response with improvements and git diff]
================================================================================

Response saved to: output/Access_Point_Selection_Precoding/science_only_response.txt

Diff extracted and saved to: output/Access_Point_Selection_Precoding/science_only.diff

To apply the diff, run:
  cd output/Access_Point_Selection_Precoding
  git apply science_only.diff

================================================================================
SCIENCE-ONLY MODE COMPLETED
================================================================================
```

## Future Enhancements

Potential improvements to science-only mode:

1. **Custom prompts:** Allow user-specified prompts
2. **Auto-apply option:** Add `--apply` flag to auto-apply diff
3. **Diff preview:** Show diff in terminal before saving
4. **Multiple models:** Try multiple models and compare
5. **Focused improvements:** Target specific sections only
6. **Confidence scoring:** Rate LLM's confidence in changes

## Troubleshooting

**Problem:** No diff in output
- **Solution:** Check `science_only_response.txt` - LLM may have returned full content instead

**Problem:** Diff doesn't apply cleanly
- **Solution:** Paper may have changed since diff was generated - review and apply manually

**Problem:** Improvements are low quality
- **Solution:** Use full workflow mode for better results with validation

**Problem:** Process hangs
- **Solution:** Check network connection, reduce timeout, try different model

## Summary

Science-only mode is a **fast, lightweight alternative** to the full workflow:

- ✅ **Speed:** 98% faster (1-2 min vs 20-150 min)
- ✅ **Simplicity:** Single API call, no complex logic
- ✅ **Review-friendly:** Git diff output for easy review
- ⚠️ **Trade-off:** No validation, no safety checks, manual review required

**Use it when:** Speed > Safety
**Avoid it when:** Safety > Speed
