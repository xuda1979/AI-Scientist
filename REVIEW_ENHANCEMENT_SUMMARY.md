# Review Enhancement Summary

## Changes Made to Require Explicit Strengths and Weaknesses

**Date**: November 1, 2025
**File Modified**: `prompts/templates.py`

### Problem
The previous review prompts did not explicitly require reviewers to list:
- **Strengths** of the paper
- **Weaknesses / Critical Issues** that need addressing

This led to reviews that sometimes focused only on problems or were too generic.

### Solution
Updated both review prompt functions to **REQUIRE** explicit structured reviews:

#### 1. `_review_prompt()` - Standalone Review Function

**Added Required Review Structure:**
```
1. SUMMARY: Brief overview (2-3 sentences)

2. STRENGTHS: List at least 3-5 specific strengths with explanations
   - What does the paper do well?
   - What are the novel contributions?
   - Which methodological choices are sound?
   - Well-written sections?

3. WEAKNESSES / CRITICAL ISSUES: List at least 3-5 specific problems
   - Major flaws or gaps
   - Claims lacking evidence
   - Methodological problems
   - Missing related work
   - Incorrect interpretations
   - Issues preventing publication

4. MINOR ISSUES: Technical corrections, typos, formatting

5. RECOMMENDATION: Accept/Minor Revision/Major Revision/Reject

6. ACTIONABLE FEEDBACK: Specific suggestions for addressing weaknesses
```

**Key Addition:**
```
⚠️ CRITICAL: You MUST explicitly list both STRENGTHS and WEAKNESSES/CRITICAL ISSUES.
Even strong papers have areas for improvement, and weak papers have some redeeming qualities.
Be balanced, specific, and constructive.
```

#### 2. `_combined_review_edit_revise_prompt()` - Combined Review + Revision Function

**Added the same structure** to ensure consistency across both review modes.

### Benefits

1. **More Balanced Reviews**: Forces reviewers to identify both positive and negative aspects
2. **Specific Feedback**: Requires at least 3-5 items in each category with explanations
3. **Actionable**: Clear structure makes it easier to address issues
4. **Consistent**: Both review functions now use the same structured format
5. **Comprehensive**: Ensures critical issues are explicitly called out, not buried in prose

### Impact on Workflow

- Reviews will now always include explicit **STRENGTHS** and **WEAKNESSES** sections
- The AI must provide at least 3-5 items in each category
- Even excellent papers will have constructive suggestions listed
- Critical issues will be clearly separated from minor suggestions
- Recommendations will be based on a balanced assessment

### Testing Recommendation

Run the review workflow on existing papers to verify:
1. Reviews now include explicit STRENGTHS section
2. Reviews now include explicit WEAKNESSES/CRITICAL ISSUES section
3. Each section has at least 3-5 specific items
4. Feedback is constructive and actionable
5. Reviews are balanced (even weak papers get some strengths noted)

### Next Steps

Consider adding:
- Automated validation to ensure reviews include required sections
- Scoring rubric for each review criterion
- Template parsing to extract strengths/weaknesses programmatically
- Quality metrics based on review structure compliance
