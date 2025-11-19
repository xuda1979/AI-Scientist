# Multi-Layered Reference Emphasis Strategy

## Problem Statement
After 5+ iterations, papers were not accumulating sufficient references (15-20 required) despite having prompts asking for them. The LLM was making other improvements but not prioritizing bibliography additions.

## Root Cause Analysis
1. **Soft warnings only**: System warned about missing references but didn't enforce action
2. **Equal priority**: References were listed alongside other issues without special emphasis
3. **No escalation**: Same message every iteration, no increasing urgency
4. **No tracking**: System didn't remind LLM how many times it had been asked

## Solution: Multi-Layered Escalating Emphasis (No Rejection)

### Layer 1: Iteration-Aware Urgency Levels
**Location**: `sciresearch_workflow.py` line ~2866

```python
urgency_level = "IMPORTANT" if iteration_count <= 2 else "CRITICAL" if iteration_count <= 4 else "URGENT - FINAL WARNING"
```

- Iterations 1-2: "IMPORTANT"
- Iterations 3-4: "CRITICAL"  
- Iterations 5+: "URGENT - FINAL WARNING"

### Layer 2: Visual Escalation with Emoji Markers
**Location**: `sciresearch_workflow.py` line ~2905

```python
ref_emphasis_header = f"{'🔴' * min(iteration_count, 10)} {urgency_level}: BIBLIOGRAPHY REQUIRED {'🔴' * min(iteration_count, 10)}"
```

- Iteration 1: 🔴 IMPORTANT: BIBLIOGRAPHY REQUIRED 🔴
- Iteration 3: 🔴🔴🔴 CRITICAL: BIBLIOGRAPHY REQUIRED 🔴🔴🔴
- Iteration 5: 🔴🔴🔴🔴🔴 URGENT - FINAL WARNING: BIBLIOGRAPHY REQUIRED 🔴🔴🔴🔴🔴

### Layer 3: Detailed BibTeX Example in Prompt
**Location**: `sciresearch_workflow.py` line ~2910

Provides complete example with exact syntax:
```latex
\begin{filecontents*}{refs.bib}
@article{AuthorYear,
  author = {Last, First},
  title = {Paper Title},
  journal = {Journal Name},
  year = {2023},
  volume = {10},
  pages = {1--20}
}
... (add N entries)
\end{filecontents*}
```

### Layer 4: Iteration Counter Reminder
**Location**: `sciresearch_workflow.py` line ~2925

```python
if iteration_count >= 3:
    critical_issues.append(f"\n📚 ITERATION {iteration_count} REMINDER: You have been asked {iteration_count-1} times to add references!")
```

- Shows how many times LLM has been reminded
- Emphasizes that this is a recurring, unresolved issue

### Layer 5: Final Warning for Late Iterations
**Location**: `sciresearch_workflow.py` line ~2930

```python
if iteration_count >= 5:
    critical_issues.append(f"\n⚠️ FINAL WARNING (Iteration {iteration_count}): This is your LAST chance to add references!")
```

### Layer 6: Priority Directive
**Location**: `sciresearch_workflow.py` line ~2943

```python
f"Focus your effort on: {'ADDING REFERENCES' if ref_count < 15 else 'EXPANDING CONTENT'}\n"
```

Explicitly tells LLM what to prioritize this iteration.

### Layer 7: Review Template Priority
**Location**: `sciresearch_workflow.py` line ~3410

In the structured review format:
```
1. **CRITICAL ISSUES & MAJOR WEAKNESSES:**
   - 📚 BIBLIOGRAPHY DEFICIT: If paper has <15 references, this MUST be listed as the FIRST and HIGHEST PRIORITY critical issue
   - 📚 A paper without adequate citations cannot be published - this is a showstopper requiring immediate action
```

Forces the review itself to emphasize references first.

### Layer 8: Mandatory Requirements Section
**Location**: `sciresearch_workflow.py` line ~3280

In the review checklist:
```
5. AUTHENTIC REFERENCES MANDATORY:
   - 🔴 PRIORITY: If paper has fewer than 15 references, this is a CRITICAL deficiency requiring immediate action
   - 🔴 Reference deficit must be addressed BEFORE any other improvements
   - 🔴 A paper without proper citations cannot be published - this is NON-NEGOTIABLE
```

### Layer 9: Post-Revision Visual Alerts
**Location**: `workflow_steps/review_revision.py` line ~266

After applying revision:
```python
if ref_count < 15:
    print(f"\n{'🔴'*40}")
    print(f"📚 REFERENCE DEFICIT ALERT - ITERATION {iteration}")
    print(f"{'🔴'*40}")
    print(f"Current references: {ref_count}")
    print(f"Required references: 15-20")
    print(f"MISSING: {ref_deficit} more references needed!")
```

## Key Design Principles

### ✅ DO:
1. **Escalate gradually**: Start gentle, increase urgency each iteration
2. **Visual markers**: Use emoji/symbols to catch LLM's attention
3. **Concrete examples**: Show exact code/format needed
4. **Track history**: Remind how many times asked
5. **Prioritize explicitly**: Tell LLM what matters most THIS iteration
6. **Multiple touchpoints**: Emphasize in prompt, review template, validation

### ❌ DON'T:
1. **Reject revisions**: Blocking forward progress frustrates workflow
2. **Same message every time**: LLM may tune out repeated identical warnings
3. **Bury in list**: References should be TOP priority when missing
4. **Assume understanding**: Provide complete examples, not just instructions

## Expected Behavior

### Iteration 1-2:
- Polite reminders with examples
- Counted as one of several quality issues

### Iteration 3-4:
- Increased visual emphasis (more 🔴 markers)
- Explicit counter: "You've been asked 2-3 times"
- Priority directive in prompt

### Iteration 5+:
- Maximum urgency: "FINAL WARNING"
- 10 🔴 markers on each side
- Priority #1 statement
- Cannot be ignored

## Testing Recommendations

1. **Run 5-iteration test** on paper with 0 references
2. **Check review files**: Each `review_iteration_N.txt` should show increasing concern about bibliography
3. **Monitor ref_count**: Should increase each iteration (even if slowly)
4. **Validate quality**: References added should be authentic, not placeholders

## Future Enhancements (Optional)

1. **Auto-detect research field**: Suggest specific journals/venues for citations
2. **Extract existing citations**: If paper has `\cite{key}` without bib entry, list missing keys
3. **Embed separate .bib files**: Auto-wrap existing `references.bib` in `filecontents*`
4. **Citation coverage analysis**: Flag sections (Intro, Related Work) with zero citations

## Success Criteria

- ✅ After 3 iterations, paper should have 10+ references
- ✅ After 5 iterations, paper should have 15-20 references
- ✅ References should be authentic (real authors, journals, years)
- ✅ References should be cited in text with `\cite{}`
- ✅ No workflow blocking or rejection (forward progress maintained)
