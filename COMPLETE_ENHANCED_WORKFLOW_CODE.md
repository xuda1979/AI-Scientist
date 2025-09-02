# Complete Enhanced Workflow Code - with Paper Structure Alignment

## New Requirement Added: Paper Structure Alignment

The workflow now includes a 7th critical requirement: **APPROPRIATE STRUCTURE** - ensuring the paper structure aligns with the paper type and field conventions.

## Updated Complete Prompt System

### 1. Initial Draft Prompt (Enhanced)

```python
def _initial_draft_prompt(topic: str, field: str, question: str) -> List[Dict[str, str]]:
    sys_prompt = (
        "You are a meticulous scientist writing a LaTeX paper suitable for a top journal. "
        
        "CRITICAL REQUIREMENTS - NO EXCEPTIONS:\n"
        "1. SINGLE FILE ONLY: Create ONE LaTeX file with NO separate bibliography files\n"
        "2. EMBEDDED REFERENCES: Include ALL references using \\begin{filecontents*}{refs.bib}...\\end{filecontents*} at the TOP of the file\n"
        "3. COMPILABLE: The file must compile successfully with pdflatex\n"
        "4. REAL REFERENCES ONLY: All references must be authentic, published works with correct details (authors, titles, journals, years, DOIs). NO FAKE or PLACEHOLDER references.\n"
        "5. SELF-CONTAINED CONTENT: ALL tables, figures, diagrams must be defined within the LaTeX file using TikZ, tabular, or other LaTeX constructs. NO external image files.\n"
        "6. DATA-DRIVEN RESULTS: All numerical values in tables/figures must come from actual simulation results, not made-up numbers.\n"
        "7. APPROPRIATE STRUCTURE: The paper structure and sections must align with the paper type and field conventions:\n"
        "   - Theoretical papers: Abstract, Introduction, Related Work, Theory/Methods, Analysis, Discussion, Conclusion\n"
        "   - Experimental papers: Abstract, Introduction, Related Work, Methodology, Experiments, Results, Discussion, Conclusion\n"
        "   - Survey papers: Abstract, Introduction, Background, Classification/Taxonomy, Comparative Analysis, Future Directions, Conclusion\n"
        "   - Systems papers: Abstract, Introduction, Related Work, System Design, Implementation, Evaluation, Discussion, Conclusion\n"
        "   - Algorithm papers: Abstract, Introduction, Related Work, Problem Definition, Algorithm Description, Analysis, Experiments, Conclusion\n\n"
        
        "STRUCTURE ALIGNMENT REQUIREMENTS:\n"
        "- Identify the paper type based on the research question and field\n"
        "- Use appropriate section names and organization for that paper type\n"
        "- Include field-specific sections (e.g., 'Threat Model' for security papers, 'Clinical Validation' for medical papers)\n"
        "- Follow established conventions for the target journal/conference\n"
        "- Ensure logical flow appropriate for the paper's contribution type\n"
        "- Include appropriate evaluation methodology for the paper type\n\n"
```

### 2. Review Prompt (Enhanced)

```python
def _review_prompt(paper_tex: str, sim_summary: str) -> List[Dict[str, str]]:
    sys_prompt = (
        "MANDATORY REQUIREMENTS - CHECK CAREFULLY:\n"
        "1. SINGLE FILE ONLY: Paper must be ONE LaTeX file with NO \\input{} or \\include{} commands\n"
        "2. EMBEDDED REFERENCES: References must be embedded using \\begin{filecontents*}{refs.bib}...\\end{filecontents*} or \\begin{thebibliography}...\\end{thebibliography}\n"
        "3. COMPILABLE: File must compile successfully with pdflatex (check for syntax errors, missing packages, etc.)\n"
        "4. AUTHENTIC REFERENCES ONLY: ALL references must be real, published works with correct bibliographic details\n"
        "5. SELF-CONTAINED VISUALS: ALL tables, figures, and diagrams must be defined within the LaTeX file using TikZ, tabular, PGFPlots, etc.\n"
        "6. APPROPRIATE STRUCTURE: The paper structure must align with the paper type and field conventions:\n"
        "   - Verify section organization matches the paper's contribution type\n"
        "   - Check for field-specific sections and evaluation methodologies\n"
        "   - Ensure logical flow appropriate for the research area\n"
        "   - Validate that section names and content align with journal standards\n\n"
        
        "STRUCTURE ALIGNMENT CRITERIA:\n"
        "- Theoretical papers should emphasize theory, proofs, and mathematical analysis\n"
        "- Experimental papers should have clear methodology, controlled experiments, and statistical analysis\n"
        "- Survey papers should provide comprehensive coverage, classification, and comparative analysis\n"
        "- Systems papers should detail architecture, implementation, and performance evaluation\n"
        "- Algorithm papers should include complexity analysis, correctness proofs, and empirical evaluation\n"
        "- Security papers should include threat models, security analysis, and attack/defense scenarios\n"
        "- Medical papers should follow clinical research standards with appropriate validation\n"
        "- Check that evaluation methodology matches the paper type and claims\n\n"
```

### 3. Revision Prompt (Enhanced)

```python
def _revise_prompt(paper_tex: str, sim_summary: str, review_text: str, latex_errors: str = "") -> List[Dict[str, str]]:
    sys_prompt = (
        "CRITICAL REQUIREMENTS - NO EXCEPTIONS:\n"
        "1. SINGLE FILE ONLY: The paper must be contained in ONE LaTeX file with NO separate bibliography files\n"
        "2. EMBEDDED REFERENCES: All references must be included in the paper using \\begin{filecontents*}{refs.bib}...\\end{filecontents*} or \\begin{thebibliography}...\\end{thebibliography}\n"
        "3. COMPILABLE: The file must compile successfully with pdflatex\n"
        "4. AUTHENTIC REFERENCES ONLY: ALL references must be real, published works\n"
        "5. SELF-CONTAINED VISUALS: ALL tables, figures, diagrams must be created using LaTeX code only\n"
        "6. APPROPRIATE STRUCTURE: Ensure the paper structure aligns with the paper type and field conventions:\n"
        "   - Organize sections according to the paper's contribution type\n"
        "   - Include field-specific sections and evaluation methodologies\n"
        "   - Follow established academic writing conventions for the research area\n"
        "   - Ensure logical flow and appropriate section transitions\n\n"
        
        "STRUCTURE ALIGNMENT REQUIREMENTS:\n"
        "- Theoretical papers: Focus on mathematical rigor, proofs, and theoretical analysis\n"
        "- Experimental papers: Include detailed methodology, controlled experiments, statistical validation\n"
        "- Survey papers: Provide comprehensive coverage, systematic classification, comparative analysis\n"
        "- Systems papers: Detail system architecture, implementation specifics, performance evaluation\n"
        "- Algorithm papers: Include complexity analysis, correctness proofs, empirical comparison\n"
        "- Security papers: Include threat models, security analysis, attack scenarios, defense mechanisms\n"
        "- Medical/Clinical papers: Follow clinical research standards with appropriate validation protocols\n"
        "- Use section names and organization appropriate for the paper type\n"
        "- Include evaluation methodology that matches the research contribution\n\n"
```

## Enhanced Validation System

### New Structure Validation Function

```python
def _check_paper_structure(paper_content: str) -> List[str]:
    """Check if paper structure aligns with paper type and field conventions."""
    issues = []
    
    # Extract all section titles
    sections = re.findall(r'\\section\*?\{([^}]+)\}', paper_content)
    section_titles = [s.lower() for s in sections]
    
    # Identify potential paper type based on content and sections
    content_lower = paper_content.lower()
    
    # Check for field-specific requirements
    if any(word in content_lower for word in ['security', 'attack', 'vulnerability', 'threat']):
        # Security paper - should have threat model
        if not any('threat' in title for title in section_titles):
            issues.append("Security paper missing 'Threat Model' or 'Security Model' section")
        if not any(word in ' '.join(section_titles) for word in ['security analysis', 'attack', 'defense']):
            issues.append("Security paper should include security analysis, attack scenarios, or defense mechanisms")
    
    if any(word in content_lower for word in ['clinical', 'medical', 'patient', 'diagnosis']):
        # Medical paper - should have validation protocols
        if not any(word in ' '.join(section_titles) for word in ['validation', 'clinical', 'evaluation']):
            issues.append("Medical paper should include clinical validation or evaluation section")
    
    if any(word in content_lower for word in ['algorithm', 'complexity', 'optimization']):
        # Algorithm paper - should have analysis
        if not any(word in ' '.join(section_titles) for word in ['analysis', 'complexity', 'performance']):
            issues.append("Algorithm paper should include complexity analysis or performance analysis section")
    
    if any(word in content_lower for word in ['survey', 'review', 'taxonomy', 'classification']):
        # Survey paper - should have comparative analysis
        if not any(word in ' '.join(section_titles) for word in ['comparison', 'comparative', 'analysis', 'classification']):
            issues.append("Survey paper should include comparative analysis or classification section")
        if not any(word in ' '.join(section_titles) for word in ['future', 'directions', 'challenges']):
            issues.append("Survey paper should include future work or research directions section")
    
    if any(word in content_lower for word in ['system', 'architecture', 'implementation', 'design']):
        # Systems paper - should have design and evaluation
        if not any(word in ' '.join(section_titles) for word in ['design', 'architecture', 'implementation']):
            issues.append("Systems paper should include system design or architecture section")
        if not any(word in ' '.join(section_titles) for word in ['evaluation', 'performance', 'experiments']):
            issues.append("Systems paper should include performance evaluation section")
    
    if any(word in content_lower for word in ['theory', 'theorem', 'proof', 'mathematical']):
        # Theoretical paper - should have theory and analysis
        if not any(word in ' '.join(section_titles) for word in ['theory', 'analysis', 'proof']):
            issues.append("Theoretical paper should include theory, analysis, or proof sections")
    
    # Check for logical section flow
    if sections:
        # Introduction should be early
        intro_pos = next((i for i, title in enumerate(section_titles) if 'introduction' in title), None)
        if intro_pos is not None and intro_pos > 2:
            issues.append("Introduction section should appear early in the paper")
        
        # Conclusion should be near the end
        conclusion_pos = next((i for i, title in enumerate(section_titles) if 'conclusion' in title), None)
        if conclusion_pos is not None and conclusion_pos < len(sections) - 3:
            issues.append("Conclusion section should appear near the end of the paper")
        
        # Related work should be early (typically sections 1-3)
        related_pos = next((i for i, title in enumerate(section_titles) if 'related' in title), None)
        if related_pos is not None and related_pos > 4:
            issues.append("Related Work section should appear early in the paper (after Introduction)")
    
    return issues
```

## Complete Requirements List (Now 7 Total)

1. **Single LaTeX File**: No external dependencies
2. **Embedded References**: All bibliography contained within the file  
3. **Compilable**: Must pass pdflatex compilation with error checking
4. **Authentic References**: Real, verified academic citations only
5. **Self-Contained Visuals**: All figures/tables created with LaTeX code
6. **Data-Driven Results**: Numerical values from actual simulation results
7. **🆕 Appropriate Structure**: Paper organization matches paper type and field conventions

## Paper Type Structure Templates

### Theoretical Papers
- Abstract → Introduction → Related Work → Theory/Methods → Analysis → Discussion → Conclusion

### Experimental Papers  
- Abstract → Introduction → Related Work → Methodology → Experiments → Results → Discussion → Conclusion

### Survey Papers
- Abstract → Introduction → Background → Classification/Taxonomy → Comparative Analysis → Future Directions → Conclusion

### Systems Papers
- Abstract → Introduction → Related Work → System Design → Implementation → Evaluation → Discussion → Conclusion

### Algorithm Papers
- Abstract → Introduction → Related Work → Problem Definition → Algorithm Description → Analysis → Experiments → Conclusion

### Security Papers
- Abstract → Introduction → Related Work → Threat Model → Security Analysis → Attack Scenarios → Defense Mechanisms → Evaluation → Conclusion

### Medical/Clinical Papers
- Abstract → Introduction → Related Work → Methodology → Clinical Validation → Results → Discussion → Limitations → Conclusion

## Field-Specific Section Requirements

The enhanced validation now automatically detects paper types and ensures:

- **Security papers**: Must include threat models, security analysis
- **Medical papers**: Must include clinical validation/evaluation
- **Algorithm papers**: Must include complexity/performance analysis  
- **Survey papers**: Must include comparative analysis and future directions
- **Systems papers**: Must include design/architecture and performance evaluation
- **Theoretical papers**: Must include theory/analysis/proof sections

## Benefits of Structure Alignment

1. **Academic Standards**: Papers follow established conventions for their field
2. **Review Quality**: Reviewers can better evaluate papers with proper structure
3. **Publication Readiness**: Papers meet journal/conference formatting expectations
4. **Reader Experience**: Logical flow appropriate for the paper's contribution type
5. **Field Compliance**: Includes necessary sections for specific research areas

This enhancement ensures that every generated paper not only meets technical requirements but also follows the academic writing conventions appropriate for its research type and field.
