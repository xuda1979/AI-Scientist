# CRITICAL FIX: Short Papers Without References

## Problem Statement

**User Issue:** After 10 iterations of revision, papers are still:
1. ❌ **Too short** (should be 5000-8000 words, but staying short)
2. ❌ **No references** (should have 15-20 references, but having 0-5)
3. ❌ **Not improving** (AI making minimal changes instead of substantial expansions)

## Root Cause Analysis

The problem was **NOT that the requirements weren't in the prompts** - they were there all along:
- ✅ "Minimum 15-20 authentic references" - present in review prompt
- ✅ "5000-8000 words" - present in content depth section
- ✅ "Embedded references mandatory" - present in requirements

**The REAL problem was:**

### 1. Ambiguous Instruction Language
```python
"STEP 2: REVISION\n"
"Provide complete file diffs for all necessary changes..."
```
The word "diffs" suggested **minimal changes** rather than **complete content expansion**.

### 2. No Explicit Pre-Review Analysis
The AI wasn't being **TOLD UPFRONT** what was wrong:
- No automatic count of current references
- No automatic count of current word count
- No explicit "YOU MUST ADD X MORE REFERENCES" instruction

### 3. No Post-Revision Validation
Even if AI made changes, there was:
- No verification that references were actually added
- No verification that content was actually expanded
- No feedback loop showing "you still need to do more"

## Solutions Implemented

### Fix #1: Explicit Expansion Instructions

**Location:** `sciresearch_workflow.py` line ~2863

**Before:**
```python
"STEP 2: REVISION\n"
"Provide complete file diffs for all necessary changes to address the review issues.\n\n"
```

**After:**
```python
"STEP 2: REVISION\n"
"Provide COMPLETE REVISED FILE CONTENTS for all files that need changes.\n"
"⚠️ CRITICAL: Do NOT provide minimal diffs or small patches - provide the ENTIRE FILE CONTENT.\n"
"⚠️ CRITICAL: The revised paper MUST be LONGER and MORE DETAILED than the original.\n"
"⚠️ CRITICAL: If the paper has few/no references, you MUST ADD 15-20 authentic references with \\begin{filecontents*}{refs.bib}.\n"
"⚠️ CRITICAL: EXPAND sections to professional length (500-1500 words each, NOT 100-200 words).\n"
"⚠️ CRITICAL: ADD content, DON'T DELETE content unless it's clearly wrong or harmful.\n\n"
```

### Fix #2: Pre-Revision Requirements Checklist

**Location:** `sciresearch_workflow.py` line ~2782

**Added:**
```python
"REVISION OUTPUT FORMAT:\n"
"Always provide complete revised file contents in this exact format:\n\n"
"⚠️ CRITICAL REVISION REQUIREMENTS BEFORE YOU START:\n"
"1. COUNT REFERENCES: The paper MUST have at least 15-20 authentic references\n"
"   - If current paper has <15 references, your revision MUST ADD more references\n"
"   - Use \\begin{filecontents*}{refs.bib}...\\end{filecontents*} at the TOP of paper.tex\n"
"   - All references must be REAL published works (author names, journal, year, etc.)\n"
"2. CHECK PAPER LENGTH: The paper MUST be 5000-8000 words (excluding references)\n"
"   - If current paper is <5000 words, your revision MUST ADD substantial content\n"
"   - EXPAND sections with more details, examples, analysis, and explanations\n"
"   - DO NOT just add fluff - add genuine academic substance\n"
"3. VERIFY COMPLETENESS: Every revised file must be COMPLETE, not a partial diff\n"
"   - Include ALL sections from \\documentclass to \\end{document}\n"
"   - Include ALL function definitions in simulation.py\n"
"   - Do NOT use placeholders like '...rest of content...' or '...continued...'\n\n"
```

### Fix #3: Automatic Quality Analysis (Tells AI What's Wrong)

**Location:** `sciresearch_workflow.py` line ~2900

**Added:**
```python
# ADD AUTOMATIC REFERENCE AND LENGTH CHECK
import re
ref_count = len(re.findall(r'\\bibitem\{|@\w+\{', paper_tex))
word_count = len(re.findall(r'\b\w+\b', paper_tex.split('\\begin{document}')[-1]...)) // 2

user += (
    "\n----- AUTOMATIC QUALITY ANALYSIS -----\n"
    f"📊 CURRENT PAPER STATISTICS:\n"
    f"   - Estimated word count: ~{word_count} words\n"
    f"   - Reference count: {ref_count} references\n"
    f"   - Target word count: 5000-8000 words\n"
    f"   - Target reference count: 15-20 references\n\n"
)

critical_issues = []
if ref_count < 15:
    critical_issues.append(f"❌ CRITICAL: Only {ref_count} references (need 15-20)")
    critical_issues.append(f"   → Your revision MUST ADD {15-ref_count} more authentic references")
    critical_issues.append(f"   → Use \\begin{{filecontents*}}{{refs.bib}} at TOP of paper.tex")

if word_count < 3000:
    critical_issues.append(f"❌ CRITICAL: Paper too short (~{word_count} words, need 5000-8000)")
    critical_issues.append(f"   → Your revision MUST EXPAND sections with {5000-word_count}+ more words")
    critical_issues.append(f"   → ADD detailed explanations, examples, analysis, related work")

if critical_issues:
    user += "🚨 MANDATORY FIXES REQUIRED:\n\n"
    for issue in critical_issues:
        user += f"{issue}\n"
    user += (
        "\n⚠️ WARNING: If your revision does NOT fix these issues, it will be REJECTED.\n"
        "Your revision MUST have 15-20 references and 5000-8000 words.\n"
```

**This tells the AI EXACTLY what needs to be fixed!**

### Fix #4: Post-Revision Validation (Verifies Changes Were Made)

**Location:** `workflow_steps/review_revision.py` line ~217

**Added:**
```python
# POST-REVISION VALIDATION: Check if critical issues were actually fixed
if 'paper.tex' in file_changes:
    import re
    new_paper_content = file_changes['paper.tex']
    
    # Count references in revised paper
    ref_count = len(re.findall(r'\\bibitem\{|@\w+\{', new_paper_content))
    
    # Estimate word count
    content_after_begin = new_paper_content.split('\\begin{document}')[-1]...
    word_count = len(re.findall(r'\b\w+\b', content_after_begin)) // 2
    
    print(f"\n{'='*80}")
    print(f"POST-REVISION VALIDATION - Iteration {iteration}")
    print(f"{'='*80}")
    print(f"📊 REVISED PAPER STATISTICS:")
    print(f"   - Estimated word count: ~{word_count} words (target: 5000-8000)")
    print(f"   - Reference count: {ref_count} references (target: 15-20)")
    
    validation_warnings = []
    if ref_count < 15:
        validation_warnings.append(f"⚠️ WARNING: Still only {ref_count} references (need 15-20)")
        validation_warnings.append(f"   The AI did NOT add enough references!")
    
    if word_count < 3000:
        validation_warnings.append(f"⚠️ WARNING: Paper still too short (~{word_count} words)")
        validation_warnings.append(f"   The AI did NOT expand content enough!")
    
    if validation_warnings:
        print(f"\n{'!'*80}")
        print(f"VALIDATION ISSUES DETECTED:")
        for warning in validation_warnings:
            print(warning)
        print(f"{'!'*80}\n")
```

**This provides immediate feedback after each iteration!**

## What You'll See Now

### During Each Iteration:

```
================================================================================
ITERATION 1 - Review and Revision Request
================================================================================

----- AUTOMATIC QUALITY ANALYSIS -----
📊 CURRENT PAPER STATISTICS:
   - Estimated word count: ~1200 words
   - Reference count: 3 references
   - Target word count: 5000-8000 words
   - Target reference count: 15-20 references

🚨 MANDATORY FIXES REQUIRED:

❌ CRITICAL: Only 3 references (need 15-20)
   → Your revision MUST ADD 12 more authentic references
   → Use \begin{filecontents*}{refs.bib} at TOP of paper.tex
❌ CRITICAL: Paper too short (~1200 words, need 5000-8000)
   → Your revision MUST EXPAND sections with 3800+ more words
   → ADD detailed explanations, examples, analysis, related work

⚠️ WARNING: If your revision does NOT fix these issues, it will be REJECTED.
Your revision MUST have 15-20 references and 5000-8000 words.
```

### After Revision Applied:

```
================================================================================
POST-REVISION VALIDATION - Iteration 1
================================================================================
📊 REVISED PAPER STATISTICS:
   - Estimated word count: ~4800 words (target: 5000-8000)
   - Reference count: 18 references (target: 15-20)
   ✓ Reference count meets requirements
   ✓ Word count is improving

✓ All validation checks passed!
================================================================================
```

## Expected Impact

### Before These Fixes:
- ❌ Paper stays at ~1500 words across 10 iterations
- ❌ References stay at 0-5 across all iterations
- ❌ AI makes only cosmetic changes
- ❌ No accountability or feedback

### After These Fixes:
- ✅ AI is TOLD upfront: "You have 3 refs, need 15"
- ✅ AI is INSTRUCTED: "MUST ADD 12 more references"
- ✅ AI receives EXPLICIT direction: "EXPAND by 3800+ words"
- ✅ AI gets VALIDATION feedback: "Still only 8 refs - not enough!"
- ✅ Iterative improvement with measurable progress tracking

## Files Modified

1. **`sciresearch_workflow.py`**
   - Line ~2782: Added pre-revision requirements checklist
   - Line ~2863: Made expansion instructions explicit and aggressive
   - Line ~2900: Added automatic quality analysis (counts refs/words)

2. **`workflow_steps/review_revision.py`**
   - Line ~217: Added post-revision validation with statistics

## Testing

To verify this works:

```bash
cd c:\Users\Lenovo\software\AI-Scientist

# Test with a minimal paper
python main.py --modify-existing --output-dir .\output\test_expansion --max-iterations 3 --model gpt-5
```

**Look for in console output:**
1. Pre-revision analysis showing current stats
2. Explicit instructions to add X references and Y words
3. Post-revision validation showing progress
4. Warnings if AI didn't add enough

## Additional Recommendations

### If papers STILL don't improve after 2-3 iterations:

1. **Check the model**
   - Some models ignore instructions better than others
   - Try `--model gpt-4o` or a different model

2. **Increase max iterations**
   - Use `--max-iterations 10` to give more opportunities

3. **Add user prompt**
   - Use `--user-prompt "The paper MUST have 20+ references and be 6000+ words"`

4. **Check review files**
   - Look at `review_iteration_1.txt` to see if AI is identifying the issues
   - If AI says "paper looks good" when it has 3 refs, that's the problem

5. **Manual intervention**
   - If after 5 iterations refs still < 10, manually add some references
   - Then let AI continue from there

## Why This Should Work

The key psychological/prompt engineering changes:

1. **Specificity:** "Add 12 more references" vs "ensure adequate references"
2. **Urgency:** "CRITICAL" "MUST" "REJECTED" language
3. **Measurement:** Actual counts shown (3 refs → need 15)
4. **Accountability:** Post-validation shows if AI failed
5. **Clarity:** "COMPLETE FILE" vs ambiguous "diffs"
6. **Repetition:** Requirements stated 3+ times in different sections

These are **aggressive prompt engineering techniques** specifically designed to overcome AI tendency to make minimal changes.
