# User Prompt Timeout Extension

**Date**: October 31, 2025  
**Change**: Extended timeout for main user prompt input from 30 seconds to 10 minutes  

---

## Summary

Modified the AI-Scientist workflow to give users **10 minutes (600 seconds)** to enter their custom prompt at the beginning of the workflow, while keeping all other timeout prompts at the original 30 seconds.

---

## Changes Made

### File: `sciresearch_workflow.py`

**Line ~4081**: Main user prompt timeout extended to 10 minutes

```python
# CHANGED: timeout=30 → timeout=600
user_prompt = timeout_input(
    "Enter your custom prompt (or press Enter to skip):", 
    timeout=600,  # 10 minutes
    default=""
).strip()
```

**Context**: This is the main prompt where users provide custom instructions for the research workflow. Users now have 10 minutes to think and compose their prompt before the system uses the default (empty string).

---

## Timeouts Kept at 30 Seconds

The following prompts remain at **30 seconds** as requested:

### 1. Research Area Prompt (Line ~4979)
```python
args.topic = timeout_input(
    "Research area of interest:", 
    timeout=30,  # Kept at 30 seconds
    default="Large Language Models"
).strip()
```

### 2. Field Prompt (Line ~4981)
```python
args.field = timeout_input(
    "Field:", 
    timeout=30,  # Kept at 30 seconds
    default="Computer Science"
).strip()
```

### 3. Research Direction Prompt (Line ~4983)
```python
args.question = timeout_input(
    "General research direction:", 
    timeout=30,  # Kept at 30 seconds
    default="Find revolutionary, impactful, and practical methods?"
).strip()
```

### 4. Content Protection Approval (utils/content_protection.py, Line ~332)
```python
choice = timeout_input(
    "Approve this revision? [y]es/[n]o/[s]how diff:", 
    timeout=30,  # Kept at 30 seconds
    default="y"
).lower().strip()
```

---

## Rationale

### Why 10 minutes for the main user prompt?
- **Composition time**: Users may need time to carefully craft detailed custom prompts
- **Research context**: Users might need to review documentation or previous results
- **Flexibility**: Allows users to step away briefly without losing their session
- **Flag compatibility**: Works seamlessly with `--user-prompt` flag usage

### Why 30 seconds for other prompts?
- **Quick responses**: Topic, field, and question are typically one-word or short answers
- **Default values**: All have sensible defaults that work well for most cases
- **Workflow efficiency**: Prevents unnecessary delays in automated workflows
- **Approval prompts**: Content protection approvals need quick decisions

---

## Usage Examples

### Interactive Mode (10-minute timeout)
```bash
python main.py --modify-existing --output-dir .\output\black_hole\ --max-iterations 1 --model gpt-5-pro
```

When prompted:
```
Enter your custom prompt (or press Enter to skip): [default: ]
```

**You now have 10 minutes to type your prompt!**

Example prompt:
```
Enhance the theoretical rigor by:
1. Adding explicit proofs for key theorems
2. Clarifying all assumptions with mathematical precision
3. Connecting to recent work on quantum information theory
4. Adding experimental validation strategies
```

### With Flag (immediate, no wait)
```bash
python main.py --modify-existing --output-dir .\output\black_hole\ \
  --user-prompt "Fix all LaTeX errors and improve clarity" \
  --max-iterations 1 --model gpt-5-pro
```

When using `--user-prompt` flag, no timeout occurs—the prompt is used immediately.

---

## Timeout Behavior

### Windows (Current System)
- **Character-by-character input**: Uses `msvcrt.kbhit()` for real-time input
- **Backspace support**: Full editing capability
- **Visual feedback**: Shows countdown or waiting indicator
- **Graceful timeout**: After 10 minutes, displays: `"Timeout reached. Using default: "`

### Unix/Linux/Mac
- **Line-based input**: Uses `select.select()` on stdin
- **Standard readline**: Full terminal editing support
- **Clean timeout**: Returns default value after timeout period

---

## Testing

To test the 10-minute timeout:

### Quick Test (won't wait full 10 minutes)
```bash
python main.py --modify-existing --output-dir .\output\black_hole\ --max-iterations 1 --model gpt-5-pro
```

1. Wait for the prompt: `Enter your custom prompt (or press Enter to skip):`
2. **Option A**: Type your prompt immediately and press Enter
3. **Option B**: Wait 10 minutes without typing—system will use empty default

### Verify Timeout Display
The prompt should show:
```
Enter your custom prompt (or press Enter to skip): [default: ]
```

After 10 minutes of no input:
```
Timeout reached. Using default: 
```

---

## Configuration

If you need to adjust the timeout in the future:

### Change Main Prompt Timeout
Edit `sciresearch_workflow.py`, line ~4081:
```python
user_prompt = timeout_input(
    "Enter your custom prompt (or press Enter to skip):", 
    timeout=600,  # Change this value (in seconds)
    default=""
).strip()
```

**Common values**:
- `300` = 5 minutes
- `600` = 10 minutes (current)
- `900` = 15 minutes
- `1200` = 20 minutes

### Change Other Prompts
If needed, edit the `timeout=30` parameter in:
- Line ~4979: Research area
- Line ~4981: Field
- Line ~4983: Research direction
- `utils/content_protection.py` line ~332: Approval prompt

---

## Impact on Workflows

### ✅ Positive Impacts
1. **Better prompts**: Users can compose thoughtful, detailed instructions
2. **Reduced errors**: Less risk of timeout during composition
3. **Multitasking**: Can reference documentation while composing
4. **Interruption handling**: Brief interruptions won't abort the session

### ⚠️ Considerations
1. **Automated scripts**: If running unattended, will wait 10 minutes before proceeding
   - **Solution**: Use `--user-prompt "..."` flag to skip waiting
2. **Quick iterations**: 10 minutes might feel long for simple prompts
   - **Solution**: Just press Enter immediately to skip

---

## Related Files

### Modified Files
- ✅ `sciresearch_workflow.py` - Main prompt timeout changed to 600s
- ✅ `utils/content_protection.py` - Kept at 30s (no change needed)

### Unchanged Files (intentionally kept at 30s)
- `sciresearch_workflow.py` - Topic, field, question prompts
- `utils/content_protection.py` - Approval prompts

---

## Rollback Instructions

If you need to revert to 30-second timeout for the main prompt:

```python
# In sciresearch_workflow.py, line ~4081
# Change back from:
user_prompt = timeout_input("Enter your custom prompt (or press Enter to skip):", timeout=600, default="").strip()

# To original:
user_prompt = timeout_input("Enter your custom prompt (or press Enter to skip):", timeout=30, default="").strip()
```

---

## Conclusion

The main user prompt now provides **10 minutes** for thoughtful composition, while keeping all other prompts responsive with **30-second** timeouts. This balances user flexibility with workflow efficiency.

**Key takeaway**: You can now take your time crafting the perfect research prompt! ⏰
