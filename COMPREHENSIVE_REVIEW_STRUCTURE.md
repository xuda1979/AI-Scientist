# Comprehensive Review Structure Implementation

**Date**: November 2, 2025  
**Files Modified**: `prompts/templates.py`

## Overview

The review system has been upgraded from a basic 6-section structure to a comprehensive **15-section formal review** structure that mirrors the review processes of top-tier journals like NeurIPS, ICML, ICLR, CVPR, ACL, and major physics/CS journals.

---

## New Formal Review Structure (15 Mandatory Sections)

### 1. **SUMMARY** (2-4 sentences)
- Brief overview of main topic and scope
- Key research question addressed
- Primary contributions claimed

### 2. **NOVELTY & SIGNIFICANCE** (Rating + Justification)
**Rating Scale**: High / Medium / Low
- What is truly novel in this work?
- How does it advance the field beyond prior art?
- What is the significance/impact of contributions?
- Are novelty claims justified?
- Is work incremental or opening new directions?

### 3. **TECHNICAL QUALITY & SOUNDNESS** (Rating + Justification)
**Rating Scale**: Excellent / Good / Fair / Poor
- Are methods scientifically sound?
- Are mathematical derivations correct?
- Are experimental/simulation designs appropriate?
- Are assumptions clearly stated and justified?
- Technical errors or questionable claims?
- Statistical analysis appropriateness

### 4. **CLARITY & PRESENTATION** (Rating + Justification)
**Rating Scale**: Excellent / Good / Fair / Poor
- Paper organization and flow
- Writing clarity and grammar
- Figure/table quality and labeling
- Equation formatting and readability
- Notation consistency
- Concept explanations

### 5. **RELATED WORK & LITERATURE REVIEW** (Rating + Justification)
**Rating Scale**: Comprehensive / Adequate / Incomplete
- Literature review thoroughness
- Key prior works cited?
- Important missing references?
- Positioning relative to prior work clear?
- Fair comparisons with existing methods?

### 6. **EXPERIMENTAL VALIDATION & RESULTS** (Rating + Justification)
**Rating Scale**: Strong / Adequate / Weak / Insufficient
- Experiment/simulation comprehensiveness
- Appropriate baselines and comparisons
- Results presentation convincing?
- Sufficient ablation studies?
- Limitations acknowledged?
- Results support claims?

### 7. **REPRODUCIBILITY** (Rating + Justification)
**Rating Scale**: Fully Reproducible / Partially / Not Reproducible
- Sufficient implementation details?
- Hyperparameters and settings specified?
- Code/data availability mentioned?
- Can results be reproduced?
- Computational requirements specified?

### 8. **STRENGTHS** (4-6 Specific Items)
Each strength must:
- Be specific and concrete (not generic)
- Reference specific sections/equations/results
- Explain WHY it is a strength

**Example Format**:
- ✅ "The theoretical framework in Section 3.2 elegantly unifies X and Y through novel use of Z, providing the first rigorous proof of..."
- ❌ "The paper is well-written" (too generic)

### 9. **WEAKNESSES & CRITICAL ISSUES** (4-6 Specific Items)
Each weakness must:
- Be specific and actionable
- Reference specific sections/claims/results
- Explain the IMPACT of the weakness
- Suggest how it could be addressed
- Be categorized as:
  - **CRITICAL** - blocks publication
  - **MAJOR** - needs fixing before acceptance
  - **MINOR** - should be addressed but not blocking

### 10. **DETAILED COMMENTS & QUESTIONS** (Section-by-Section)
For each major section:
- Specific comments on content, clarity, correctness
- Questions that need addressing
- Suggestions for improvement

**Format**: "Section X.Y: [detailed comment]"

### 11. **MINOR ISSUES** (Technical Corrections)
- Typos and grammatical errors
- Notation inconsistencies
- Formatting issues
- Reference formatting problems
- Figure/table caption improvements

### 12. **ETHICAL CONSIDERATIONS** (If Applicable)
- Ethical concerns with the research?
- Limitations and potential negative impacts discussed?
- Proper credit to prior work?
- Any conflicts of interest apparent?

### 13. **OVERALL RECOMMENDATION** (Choose ONE + Justify)
**Options**:
- ○ **STRONG ACCEPT** - Excellent paper, ready for publication
- ○ **ACCEPT** - Good paper, minor revisions only
- ○ **WEAK ACCEPT** - Acceptable but needs improvements
- ○ **BORDERLINE** - Could go either way, depends on revisions
- ○ **WEAK REJECT** - Has merit but significant issues
- ○ **REJECT** - Major flaws, not suitable for this venue
- ○ **STRONG REJECT** - Fundamentally flawed

Must include 2-3 sentence justification.

### 14. **CONFIDENCE LEVEL** (Choose ONE)
- ○ **EXPERT** - This is my area of expertise
- ○ **HIGH** - Knowledgeable about this topic
- ○ **MEDIUM** - Familiar with related areas
- ○ **LOW** - Outside my main expertise

### 15. **ACTIONABLE NEXT STEPS** (Prioritized List)
Concrete actions in priority order:
1. [Most critical fix with specific details]
2. [Second priority with specific details]
3. [Third priority...]

Each step must be specific about what to do and why.

---

## Key Improvements Over Previous System

### Before (6 Sections):
1. Summary
2. Strengths
3. Weaknesses
4. Minor Issues
5. Recommendation
6. Actionable Feedback

### After (15 Sections):
✅ **Added Rating-Based Assessments** for:
- Novelty & Significance
- Technical Quality
- Clarity & Presentation
- Related Work
- Experimental Validation
- Reproducibility

✅ **Added Structured Evaluations**:
- Detailed section-by-section comments
- Ethical considerations
- Confidence level

✅ **Enhanced Specificity Requirements**:
- Must reference specific sections/equations
- Must categorize weaknesses by severity
- Must provide concrete examples
- Must justify all ratings

✅ **More Granular Recommendations**:
- 7 levels instead of 4
- Required justification
- Explicit confidence declaration

---

## Benefits of the New Structure

### 1. **Comprehensive Coverage**
- Every aspect of the paper is systematically evaluated
- No important dimension is overlooked
- Matches industry-standard review processes

### 2. **Quantifiable Assessments**
- Ratings provide clear benchmarks
- Easy to track improvements across iterations
- Facilitates objective comparison

### 3. **Actionable Feedback**
- Specific, concrete suggestions
- Prioritized action items
- Clear path to improvement

### 4. **Balanced Evaluation**
- Strengths and weaknesses both required
- Multiple perspectives (technical, clarity, novelty, etc.)
- Fair assessment even for weak papers

### 5. **Professional Standard**
- Mirrors top-tier journal review processes
- Prepares papers for real peer review
- Builds credibility of the system

### 6. **Better Author Guidance**
- Clear understanding of what needs improvement
- Specific section references
- Prioritized action list

---

## Implementation Details

### Modified Functions:
1. **`_review_prompt()`** - Standalone review function
2. **`_combined_review_edit_revise_prompt()`** - Combined review+revision workflow

### Enforcement:
- All 15 sections are **MANDATORY**
- Ratings must be provided where requested
- Reviews must include specific examples and references
- Generic or vague feedback is explicitly discouraged

### Quality Control:
The prompt explicitly requires:
- Citing specific sections, equations, page numbers
- Providing concrete examples for all claims
- Balancing criticism with constructive suggestions
- Using professional, respectful language
- Justifying all ratings with evidence

---

## Example Review Snippet

```markdown
## 2. NOVELTY & SIGNIFICANCE
**Rating**: Medium

**Justification**: 
The paper introduces the Horizon Memory Comb (HMC) framework, which 
is a novel combination of existing concepts (quantum combs, black hole 
thermodynamics, and non-Markovian processes). While the specific 
synthesis is new, the individual components (edge modes, JT gravity, 
process tensors) are well-established in the literature.

**Specific Strengths**:
- The dynamic memory dimension linked to Bekenstein-Hawking entropy 
  (Eq. 1.2) is an elegant and novel connection
- The derivation of the Comb Page Theorem (Section 3.2) provides 
  the first rigorous proof in this framework

**Limitations**:
- The connection to existing island formula approaches (Penington 2020) 
  is not sufficiently distinguished
- The novelty relative to recent work on quantum error correction in 
  AdS/CFT (ADH 2015) needs clearer articulation
```

---

## Usage Guidelines

### For Reviewers (AI):
1. **Complete ALL 15 sections** - no exceptions
2. **Provide ratings** where requested
3. **Be specific** - cite equations, sections, figures
4. **Be balanced** - find both strengths and weaknesses
5. **Be constructive** - suggest improvements
6. **Justify claims** - provide evidence for ratings

### For Authors (System):
- Parse review structure to extract ratings
- Track improvements across iterations
- Use ratings for automated quality metrics
- Present feedback in organized format

### For Monitoring:
- Validate that all 15 sections are present
- Check that ratings are provided
- Verify specificity (references to sections/equations)
- Ensure actionable feedback is concrete

---

## Future Enhancements

### Potential Additions:
1. **Numerical Scoring System** - Convert ratings to 1-5 scores
2. **Weighted Recommendations** - Combine section ratings into overall score
3. **Review Quality Metrics** - Assess review comprehensiveness
4. **Automatic Parsing** - Extract ratings and categorize feedback
5. **Comparison Tools** - Track progress across review iterations
6. **Meta-Review** - AI review of the AI review quality

### Integration Opportunities:
- Dashboard visualization of review ratings
- Automatic prioritization of revision tasks
- Quality gates based on review scores
- Trend analysis across multiple papers

---

## Compliance Verification

To verify a review meets the new standards, check:

- [ ] All 15 sections present
- [ ] All ratings provided (High/Medium/Low, Excellent/Good/Fair/Poor, etc.)
- [ ] Strengths list 4-6 specific items
- [ ] Weaknesses list 4-6 specific items with categorization
- [ ] Section-by-section comments provided
- [ ] Recommendation justified (2-3 sentences)
- [ ] Confidence level declared
- [ ] Actionable next steps prioritized and specific
- [ ] References to specific sections/equations throughout
- [ ] Professional, constructive tone maintained

---

## Conclusion

This comprehensive 15-section review structure transforms the review process from a basic checklist into a professional, thorough evaluation that matches industry standards. It ensures every important dimension is evaluated, provides clear ratings and recommendations, and gives authors specific, actionable guidance for improvement.
