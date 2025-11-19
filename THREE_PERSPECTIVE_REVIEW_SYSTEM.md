# Three-Perspective Review System

## Overview

The AI-Scientist review system now implements a **comprehensive three-perspective review framework** that integrates:

1. **Academic Peer Review** - Assesses scientific content and quality
2. **Editorial Review** - Focuses on writing, structure, and presentation  
3. **Technical Review** - Checks mathematical and LaTeX correctness

This ensures every paper receives thorough evaluation covering all aspects of quality that top-tier journals demand.

---

## Three Review Perspectives

### Part A: Academic Peer Review (Scientific Content & Quality)

**Purpose**: Evaluate the scientific merit, novelty, and methodological rigor of the research.

**Sections (1-6)**:
1. **SUMMARY** - Brief overview of the research
2. **NOVELTY & SIGNIFICANCE** - Assessment of contributions (Rate: High/Medium/Low)
3. **METHODOLOGICAL SOUNDNESS** - Rigor of research methods (Rate: Excellent/Good/Fair/Poor)
4. **RESULTS & EVIDENCE** - Strength of empirical support (Rate: Strong/Adequate/Weak/Insufficient)
5. **RELATED WORK & LITERATURE REVIEW** - Completeness of context (Rate: Comprehensive/Adequate/Incomplete)
6. **REPRODUCIBILITY** - Ability to replicate findings (Rate: Fully/Partially/Not Reproducible)

**Key Criteria**:
- Scientific rigor and methodology soundness
- Novel contribution beyond prior art
- Proper literature review with 15-20 authentic references
- Clear research question and appropriate experimental design
- Results interpretation and limitations acknowledgment
- Real simulation data usage (no fabricated numbers)
- Reproducible results documentation

---

### Part B: Editorial Review (Structure, Writing & Presentation)

**Purpose**: Ensure the paper is well-written, clearly organized, and effectively communicates its findings.

**Sections (7-10)**:
7. **ORGANIZATION & STRUCTURE** - Logical flow and coherence (Rate: Excellent/Good/Fair/Poor)
8. **WRITING QUALITY & CLARITY** - Precision and readability (Rate: Excellent/Good/Fair/Poor)
9. **GRAMMAR, STYLE & LANGUAGE** - Correctness and tone (Rate: Excellent/Good/Fair/Poor)
10. **VISUAL PRESENTATION** - Figures and tables quality (Rate: Excellent/Good/Fair/Poor)

**Key Criteria**:
- Paper structure and logical organization
- Writing clarity, grammar, and flow
- Consistent terminology and appropriate academic tone
- Figure/table quality, self-contained visuals with complete captions
- No filename references in paper text
- Effective use of visuals to support narrative

---

### Part C: Technical Review (Mathematical & LaTeX Correctness)

**Purpose**: Verify technical correctness of mathematical content and LaTeX formatting.

**Sections (11-14)**:
11. **MATHEMATICAL CORRECTNESS** - Accuracy of derivations (Rate: Correct/Minor Issues/Major Errors)
12. **EQUATION FORMATTING & NOTATION** - Readability and consistency (Rate: Excellent/Good/Fair/Poor)
13. **LaTeX FORMATTING & COMPILATION** - Build success and formatting (Rate: Perfect/Good/Issues/Broken)
14. **TECHNICAL NOTATION & SYMBOLS** - Consistency and clarity (Rate: Excellent/Good/Fair/Poor)

**Key Criteria**:
- Mathematical derivations correctness
- **Equation overflow prevention** - ALL display equations must fit within text width
  - Use `align`, `multline`, `split`, or `\resizebox` for long equations
  - Never allow equations to overflow margins
- LaTeX compilation success without errors
- Authentic references (no fake citations)
- Single file structure with embedded bibliography
- Proper cross-references (\\ref, \\cite working correctly)
- Notation consistency throughout

---

### Part D: Integrated Assessment (Synthesis of All Three Perspectives)

**Purpose**: Provide holistic evaluation combining insights from all three review types.

**Sections (15-22)**:
15. **STRENGTHS** - 4-6 specific strengths categorized by type
16. **WEAKNESSES & CRITICAL ISSUES** - 4-6 items with type and severity
17. **DETAILED SECTION-BY-SECTION COMMENTS** - Comprehensive feedback
18. **MINOR ISSUES** - Small corrections categorized by type
19. **ETHICAL CONSIDERATIONS** - If applicable
20. **OVERALL RECOMMENDATION** - 7-level scale with justification
21. **CONFIDENCE LEVEL** - Reviewer expertise assessment
22. **PRIORITIZED ACTION ITEMS** - Organized by type and severity

---

## Review Type Categorization System

All feedback items must be tagged with their review type:

- **[ACADEMIC]** - Issues related to scientific content, methodology, novelty, results
- **[EDITORIAL]** - Issues related to writing, structure, clarity, presentation
- **[TECHNICAL]** - Issues related to equations, LaTeX, mathematical correctness

### Examples:

**Strengths**:
```
[ACADEMIC] The novel application of quantum combs to black hole information 
  provides a fresh perspective that advances beyond previous approaches.
  
[EDITORIAL] The paper is exceptionally well-organized with clear section 
  transitions and a compelling narrative that guides readers through complex concepts.
  
[TECHNICAL] All equations are properly formatted using align environments, 
  with consistent notation and no overflow issues.
```

**Weaknesses**:
```
[ACADEMIC - CRITICAL] The experimental validation lacks comparison with 
  baseline methods, making it impossible to assess relative performance.
  
[EDITORIAL - MAJOR] Section 3 contains run-on sentences and unclear 
  explanations that obscure the main contributions.
  
[TECHNICAL - MINOR] Equation 12 uses inconsistent notation for the Hamiltonian 
  (H vs \mathcal{H}).
```

---

## Severity Classification

Weaknesses must be categorized by severity:

- **CRITICAL** - Blocks publication; must be fixed
  - Example: Missing experimental validation, mathematical errors, fabricated data
  
- **MAJOR** - Significant issues that need fixing before acceptance
  - Example: Unclear methodology, inadequate literature review, poor organization
  
- **MINOR** - Small improvements that enhance quality
  - Example: Typos, notation inconsistencies, caption improvements

---

## Rating Scales Used

Different aspects use different rating scales appropriate to what's being assessed:

### Novelty & Significance
- **High** - Groundbreaking, opens new research directions
- **Medium** - Solid contribution with clear advancement
- **Low** - Incremental, limited novelty

### Technical Quality / Clarity / Writing / Organization
- **Excellent** - Exceptional, publication-ready
- **Good** - Solid, minor improvements needed
- **Fair** - Acceptable but needs work
- **Poor** - Significant issues present

### Results & Evidence
- **Strong** - Comprehensive, convincing validation
- **Adequate** - Sufficient but could be strengthened
- **Weak** - Questionable or incomplete
- **Insufficient** - Lacks necessary validation

### Literature Review
- **Comprehensive** - Thorough coverage of relevant work
- **Adequate** - Covers main areas sufficiently
- **Incomplete** - Missing important references

### Reproducibility
- **Fully Reproducible** - Complete details provided
- **Partially Reproducible** - Some details missing
- **Not Reproducible** - Insufficient information

### Mathematical Correctness
- **Correct** - No errors found
- **Minor Issues** - Small notation or formatting problems
- **Major Errors** - Significant mathematical mistakes

### LaTeX Formatting
- **Perfect** - Compiles cleanly, no issues
- **Good** - Compiles with minor warnings
- **Issues** - Formatting problems present
- **Broken** - Does not compile

### Overall Recommendation (7-Level Scale)
1. **STRONG ACCEPT** - Excellent across all three perspectives
2. **ACCEPT** - Good overall, minor improvements needed
3. **WEAK ACCEPT** - Acceptable but needs improvements in 1-2 dimensions
4. **BORDERLINE** - Mixed quality across review dimensions
5. **WEAK REJECT** - Significant issues in multiple dimensions
6. **REJECT** - Major flaws in one or more dimensions
7. **STRONG REJECT** - Fundamentally flawed

### Confidence Level
- **EXPERT** - Expert in all three review dimensions
- **HIGH** - Strong knowledge across all dimensions
- **MEDIUM** - Familiar with most dimensions
- **LOW** - Limited expertise in some dimensions

---

## Complete 22-Section Structure

### Part A: Academic Peer Review
1. Summary
2. Novelty & Significance
3. Methodological Soundness
4. Results & Evidence
5. Related Work & Literature Review
6. Reproducibility

### Part B: Editorial Review
7. Organization & Structure
8. Writing Quality & Clarity
9. Grammar, Style & Language
10. Visual Presentation

### Part C: Technical Review
11. Mathematical Correctness
12. Equation Formatting & Notation
13. LaTeX Formatting & Compilation
14. Technical Notation & Symbols

### Part D: Integrated Assessment
15. Strengths (categorized by type)
16. Weaknesses & Critical Issues (categorized by type and severity)
17. Detailed Section-by-Section Comments (all three perspectives)
18. Minor Issues (categorized by type)
19. Ethical Considerations
20. Overall Recommendation
21. Confidence Level
22. Prioritized Action Items (organized by type and severity)

---

## Action Items Organization

Prioritized action items should be organized by both review type and severity:

```
[CRITICAL - Must Fix]
1. [ACADEMIC] Add baseline comparisons in experimental validation (Section 4)
2. [TECHNICAL] Fix equation overflow in Eq. 15 using align environment

[MAJOR - Should Fix]
1. [EDITORIAL] Restructure Section 3 for better clarity and flow
2. [ACADEMIC] Expand literature review to include recent 2023-2024 papers
3. [TECHNICAL] Fix inconsistent notation for quantum states (\ket vs |)

[MINOR - Nice to Have]
1. [EDITORIAL] Fix typos in abstract and conclusion
2. [TECHNICAL] Add labels to all subfigures for better reference
3. [ACADEMIC] Add brief discussion of negative results in Section 5
```

---

## Benefits of Three-Perspective System

### 1. Comprehensive Coverage
Every important aspect of paper quality is evaluated:
- Scientific merit (Academic)
- Communication effectiveness (Editorial)
- Technical correctness (Technical)

### 2. Clear Responsibility
Reviewers know exactly what to assess in each section:
- Academic perspective: "Is this good science?"
- Editorial perspective: "Is this well-communicated?"
- Technical perspective: "Is this technically correct?"

### 3. Balanced Evaluation
Prevents over-focus on one dimension at expense of others:
- A paper with excellent science but poor writing gets proper feedback
- A well-written but methodologically weak paper is identified
- Technical issues don't get overlooked due to strong content

### 4. Actionable Feedback
Authors know exactly what type of improvements are needed:
- [ACADEMIC] issues → work with domain experts, add experiments
- [EDITORIAL] issues → work with writing coach, restructure
- [TECHNICAL] issues → fix LaTeX, check math, verify equations

### 5. Professional Standards
Matches review processes at top venues:
- **NeurIPS/ICML**: Academic peer review focus with technical rigor
- **JHEP/Physical Review**: Technical review of equations critical
- **Nature/Science**: Editorial review ensures broad accessibility

### 6. Explicit Categorization
Every piece of feedback is tagged with review type, making it clear:
- What kind of issue it is
- Who should address it (scientist, writer, or technical editor)
- How to prioritize fixes

---

## Implementation in AI-Scientist

### File Locations
- **Review prompt**: `prompts/templates.py` → `_review_prompt()`
- **Combined workflow**: `prompts/templates.py` → `_combined_review_edit_revise_prompt()`
- **Review execution**: `workflow_steps/review_revision.py`

### Key Functions

#### `_review_prompt()`
Generates standalone review using 22-section structure organized by three perspectives.

**Input**:
- `paper_tex`: LaTeX content to review
- `sim_summary`: Simulation results summary
- `project_dir`: Project directory path (optional)
- `user_prompt`: Additional user instructions (optional)

**Output**: List of system/user message dicts for LLM

#### `_combined_review_edit_revise_prompt()`
Generates combined review+revision workflow using three-perspective structure.

**Input**:
- Same as `_review_prompt()` plus:
- `latex_errors`: LaTeX compilation errors (if any)
- `iteration_count`: Revision iteration number
- `quality_issues`: List of detected quality problems

**Output**: Prompts for review followed by complete file revision

### Mandatory Requirements

Both prompts enforce:
- ✅ ALL 22 sections are MANDATORY
- ✅ Explicit categorization using [ACADEMIC]/[EDITORIAL]/[TECHNICAL] tags
- ✅ Ratings using specified scales
- ✅ Specific examples and references (section numbers, equation numbers)
- ✅ Balance of criticism with constructive suggestions
- ✅ Professional, respectful language throughout

---

## Example Review Snippet

```markdown
═══════════════════════════════════════════════════════════════
PART A: ACADEMIC PEER REVIEW (Scientific Content & Quality)
═══════════════════════════════════════════════════════════════

1. SUMMARY
This paper proposes "Horizon Memory Combs" (HMC), a non-Markovian quantum 
channel framework to address the black hole information paradox. The authors 
introduce a novel mathematical structure that enables information preservation 
through multi-step correlations at the event horizon.

2. NOVELTY & SIGNIFICANCE
**Rating: MEDIUM**
The application of quantum combs to black hole physics is relatively novel, 
though the underlying quantum comb formalism is well-established. The key 
innovation is showing how non-Markovian dynamics at the horizon can preserve 
information. However, the connection to existing approaches (e.g., Page curve 
analysis, island formula) needs clearer articulation.

═══════════════════════════════════════════════════════════════
PART B: EDITORIAL REVIEW (Structure, Writing & Presentation)
═══════════════════════════════════════════════════════════════

7. ORGANIZATION & STRUCTURE
**Rating: GOOD**
The paper follows a logical progression from motivation → framework → 
results → implications. Section transitions are generally smooth, though 
Section 3's subsection ordering could be improved.

═══════════════════════════════════════════════════════════════
PART C: TECHNICAL REVIEW (Mathematical & LaTeX Correctness)
═══════════════════════════════════════════════════════════════

11. MATHEMATICAL CORRECTNESS
**Rating: CORRECT (with minor notation issues)**
Mathematical derivations appear sound. The proof in Appendix A correctly 
applies quantum comb composition rules. However, Equation 12 uses H for 
Hamiltonian while earlier equations use \mathcal{H}.

12. EQUATION FORMATTING & NOTATION
**Rating: GOOD**
⚠️ CRITICAL ISSUE: Equation 15 overflows the text margin. Recommend breaking 
into multiple lines using align environment:
\begin{align}
S_{\rm BH} &= A/4G_N + \text{[quantum corrections]} \\
           &\quad + \sum_{i=1}^n \text{[memory terms]}
\end{align}

═══════════════════════════════════════════════════════════════
PART D: INTEGRATED ASSESSMENT
═══════════════════════════════════════════════════════════════

15. STRENGTHS
[ACADEMIC] Novel application of quantum combs provides fresh mathematical 
  perspective on information preservation mechanisms.
[ACADEMIC] Clear demonstration that non-Markovian effects can resolve paradox.
[EDITORIAL] Abstract is exceptionally clear and compelling.
[TECHNICAL] LaTeX code is well-organized with consistent macro usage.

16. WEAKNESSES & CRITICAL ISSUES
[ACADEMIC - MAJOR] Missing comparison with Page curve and island formula 
  approaches. How does HMC relate to these established frameworks?
[EDITORIAL - MINOR] Section 3.2 contains dense paragraphs that need breaking.
[TECHNICAL - CRITICAL] Equation 15 overflows margin; must be reformatted.

22. PRIORITIZED ACTION ITEMS
[CRITICAL - Must Fix]
1. [TECHNICAL] Fix equation 15 overflow using align or split environment
2. [ACADEMIC] Add subsection comparing HMC to Page curve/island approaches

[MAJOR - Should Fix]
1. [EDITORIAL] Break dense paragraphs in Section 3.2
2. [ACADEMIC] Expand related work section with 2023-2024 references
3. [TECHNICAL] Unify Hamiltonian notation (H vs \mathcal{H})
```

---

## Compliance Checklist

Use this checklist to verify reviews meet the three-perspective standard:

### Academic Perspective (Part A)
- [ ] Section 1: Summary provided (2-4 sentences)
- [ ] Section 2: Novelty rated (High/Medium/Low) with justification
- [ ] Section 3: Methods assessed (Excellent/Good/Fair/Poor) with examples
- [ ] Section 4: Results evaluated (Strong/Adequate/Weak/Insufficient)
- [ ] Section 5: Literature reviewed (Comprehensive/Adequate/Incomplete)
- [ ] Section 6: Reproducibility rated (Fully/Partially/Not) with specifics

### Editorial Perspective (Part B)
- [ ] Section 7: Organization rated (Excellent/Good/Fair/Poor)
- [ ] Section 8: Writing quality rated (Excellent/Good/Fair/Poor)
- [ ] Section 9: Grammar/style rated (Excellent/Good/Fair/Poor)
- [ ] Section 10: Visuals rated (Excellent/Good/Fair/Poor)

### Technical Perspective (Part C)
- [ ] Section 11: Math correctness assessed (Correct/Minor/Major)
- [ ] Section 12: Equation formatting checked (with overflow verification)
- [ ] Section 13: LaTeX compilation verified (Perfect/Good/Issues/Broken)
- [ ] Section 14: Notation consistency checked (Excellent/Good/Fair/Poor)

### Integrated Assessment (Part D)
- [ ] Section 15: At least 4-6 strengths, all tagged by type
- [ ] Section 16: At least 4-6 weaknesses, tagged by type AND severity
- [ ] Section 17: Section-by-section feedback covering all three perspectives
- [ ] Section 18: Minor issues categorized by type
- [ ] Section 19: Ethical considerations addressed (if applicable)
- [ ] Section 20: Recommendation chosen from 7-level scale with justification
- [ ] Section 21: Confidence level indicated
- [ ] Section 22: Action items organized by type and severity

### General Requirements
- [ ] All feedback tagged with [ACADEMIC]/[EDITORIAL]/[TECHNICAL]
- [ ] All ratings use specified scales
- [ ] Specific references provided (section numbers, equations, line numbers)
- [ ] Constructive suggestions included with criticisms
- [ ] Professional, respectful language throughout

---

## Migration from Previous System

### Old System (15 sections, single perspective)
The previous system had 15 sections but didn't explicitly separate review types:
1. Summary
2. Novelty & Significance
3. Technical Quality & Soundness (mixed academic/technical)
4. Clarity & Presentation (mixed editorial/technical)
5. Related Work
6. Experimental Validation
7. Reproducibility
8. Strengths (not categorized by type)
9. Weaknesses (not categorized by type)
10. Detailed Comments
11. Minor Issues
12. Ethical Considerations
13. Recommendation
14. Confidence
15. Action Items

### New System (22 sections, three explicit perspectives)
Expands to 22 sections with clear separation:
- **Part A (6 sections)**: Pure academic peer review
- **Part B (4 sections)**: Pure editorial review
- **Part C (4 sections)**: Pure technical review
- **Part D (8 sections)**: Integrated assessment with categorization

### Key Improvements
1. **Explicit categorization**: Every feedback item tagged by type
2. **Separation of concerns**: Academic, editorial, technical clearly separated
3. **More granular assessment**: 14 rated dimensions (vs 7 previously)
4. **Better organization**: Grouped by perspective for clarity
5. **Severity classification**: CRITICAL/MAJOR/MINOR for all weaknesses
6. **Type-organized action items**: Easier to assign responsibility

---

## Future Enhancements

Potential improvements to consider:

1. **Automated Categorization Validation**
   - Parser to verify all feedback is properly tagged
   - Check that categorizations are consistent

2. **Perspective-Specific Sub-Reviews**
   - Option to generate separate detailed reports for each perspective
   - Useful for teams with specialized reviewers

3. **Weighted Scoring**
   - Different venues may weight perspectives differently
   - JHEP: High weight on technical correctness
   - Nature: High weight on editorial quality
   - NeurIPS: Balanced across all three

4. **Automated Issue Routing**
   - [ACADEMIC] → Route to domain expert
   - [EDITORIAL] → Route to writing coach
   - [TECHNICAL] → Route to LaTeX expert

5. **Compliance Scoring**
   - Automated check: "Review completeness: 22/22 sections present"
   - Flagging of missing ratings or uncategorized feedback

---

## Conclusion

The three-perspective review system provides comprehensive, well-organized, and actionable feedback that covers:
- ✅ Scientific quality and novelty (Academic)
- ✅ Communication and presentation (Editorial)
- ✅ Technical and mathematical correctness (Technical)

This ensures papers receive thorough evaluation matching the standards of top-tier journals while providing clear, categorized feedback that authors can act upon systematically.
