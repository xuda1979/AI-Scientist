# Enhanced Workflow with Authentic References and Self-Contained Visuals

## Summary of New Requirements Added

The workflow has been enhanced with two critical new requirements:

### 1. Authentic References Only
- All references must be real, published works with correct bibliographic details
- NO fake, placeholder, or fictional citations allowed
- Verification of author names, journal names, publication years, and DOIs
- References must be directly relevant to the research topic

### 2. Self-Contained Visual Content
- ALL tables, figures, and diagrams must be created using LaTeX code only
- NO external image files or dependencies allowed
- Visual content must use TikZ, PGFPlots, tabular, or other LaTeX-native tools
- All numerical data must come from actual simulation results

## Updated Prompt Structure

### Initial Draft Prompt

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
        "6. DATA-DRIVEN RESULTS: All numerical values in tables/figures must come from actual simulation results, not made-up numbers.\n\n"
        
        "REFERENCE REQUIREMENTS:\n"
        "- Minimum 15-20 authentic, recently published references (prefer 2018-2025)\n"
        "- Include proper DOIs where available\n"
        "- Verify all author names, journal names, and publication details\n"
        "- References must be directly relevant to the research topic\n"
        "- Use proper citation style throughout the paper\n\n"
        
        "CONTENT SELF-CONTAINMENT:\n"
        "- Figures: Use TikZ, PGFPlots, or pure LaTeX constructs only\n"
        "- Tables: Create with tabular environment, populate with simulation data\n"
        "- Diagrams: Use TikZ or similar LaTeX-native tools\n"
        "- NO \\includegraphics commands for external files\n"
        "- ALL visual content must render from LaTeX code only\n\n"
```

### Review Prompt

```python
def _review_prompt(paper_tex: str, sim_summary: str) -> List[Dict[str, str]]:
    sys_prompt = (
        "Act as a top-tier journal reviewer (Nature, Science, Cell level) with expertise in LaTeX formatting and scientific programming. "
        
        "MANDATORY REQUIREMENTS - CHECK CAREFULLY:\n"
        "1. SINGLE FILE ONLY: Paper must be ONE LaTeX file with NO \\input{} or \\include{} commands\n"
        "2. EMBEDDED REFERENCES: References must be embedded using \\begin{filecontents*}{refs.bib}...\\end{filecontents*} or \\begin{thebibliography}...\\end{thebibliography}\n"
        "3. COMPILABLE: File must compile successfully with pdflatex (check for syntax errors, missing packages, etc.)\n"
        "4. AUTHENTIC REFERENCES ONLY: ALL references must be real, published works with correct bibliographic details. Verify:\n"
        "   - Author names are real and spelled correctly\n"
        "   - Journal/venue names are authentic\n"
        "   - Publication years are realistic\n"
        "   - DOIs are properly formatted (if provided)\n"
        "   - References are directly relevant to the topic\n"
        "   - NO placeholder, fake, or made-up citations\n"
        "5. SELF-CONTAINED VISUALS: ALL tables, figures, and diagrams must be:\n"
        "   - Defined within the LaTeX file using TikZ, tabular, PGFPlots, etc.\n"
        "   - NO external image files or \\includegraphics{external_file}\n"
        "   - Populated with REAL data from simulation results\n"
        "   - NO fake, placeholder, or estimated numbers\n\n"
        
        "REFERENCE QUALITY CRITERIA:\n"
        "- Minimum 15-20 authentic references from reputable sources\n"
        "- Recent publications (prefer 2018-2025) mixed with foundational works\n"
        "- Proper citation usage throughout the paper\n"
        "- References must support claims made in the text\n"
        "- Check for citation format consistency\n\n"
        
        "CONTENT SELF-CONTAINMENT CRITERIA:\n"
        "- All figures created with LaTeX code (TikZ, PGFPlots, etc.)\n"
        "- All tables use tabular environment with simulation data\n"
        "- All diagrams use LaTeX-native drawing tools\n"
        "- Numbers in tables/figures match simulation outputs exactly\n"
        "- NO references to external files or missing data\n\n"
```

### Revision Prompt

```python
def _revise_prompt(paper_tex: str, sim_summary: str, review_text: str, latex_errors: str = "") -> List[Dict[str, str]]:
    sys_prompt = (
        "You are the paper author making revisions based on peer review. Your goal is to address ALL reviewer concerns "
        "while maintaining scientific integrity and clarity. Produce a COMPLETE revised LaTeX file.\n\n"
        
        "CRITICAL REQUIREMENTS - NO EXCEPTIONS:\n"
        "1. SINGLE FILE ONLY: The paper must be contained in ONE LaTeX file with NO separate bibliography files\n"
        "2. EMBEDDED REFERENCES: All references must be included in the paper using \\begin{filecontents*}{refs.bib}...\\end{filecontents*} or \\begin{thebibliography}...\\end{thebibliography}\n"
        "3. COMPILABLE: The file must compile successfully with pdflatex\n"
        "4. AUTHENTIC REFERENCES ONLY: ALL references must be real, published works. Verify and correct:\n"
        "   - Author names are real researchers in the field\n"
        "   - Journal/venue names are authentic and properly formatted\n"
        "   - Publication years are realistic and consistent\n"
        "   - DOIs are properly formatted (when available)\n"
        "   - References directly support claims in the text\n"
        "   - NO placeholder, fake, or fictional citations\n"
        "5. SELF-CONTAINED VISUALS: ALL tables, figures, diagrams must be:\n"
        "   - Created using LaTeX code only (TikZ, tabular, PGFPlots, etc.)\n"
        "   - NO external files or \\includegraphics{external_file} commands\n"
        "   - Populated with actual simulation data, not fake numbers\n"
        "   - Self-rendering within the LaTeX document\n\n"
        
        "REFERENCE AUTHENTICITY REQUIREMENTS:\n"
        "- Replace any suspicious or placeholder references with real publications\n"
        "- Ensure minimum 15-20 authentic references from reputable sources\n"
        "- Use recent publications (2018-2025) mixed with foundational works\n"
        "- Verify all bibliographic details are correct\n"
        "- Ensure proper citation usage throughout the paper\n\n"
        
        "VISUAL CONTENT SELF-CONTAINMENT:\n"
        "- Convert any external figure references to TikZ/PGFPlots code\n"
        "- Ensure all tables use tabular environment with real simulation data\n"
        "- Create diagrams using TikZ or other LaTeX-native tools\n"
        "- Populate all numerical content from actual simulation results\n"
        "- Remove any dependencies on external image files\n\n"
```

## Enhanced Validation Functions

### Reference Authenticity Checking

```python
def _check_reference_authenticity(paper_content: str) -> List[str]:
    """Check for potentially fake or suspicious references."""
    issues = []
    
    # Look for common fake reference patterns
    fake_patterns = [
        r'@article\{[^}]*,\s*author\s*=\s*\{[^}]*Example[^}]*\}',  # "Example" in author names
        r'@article\{[^}]*,\s*author\s*=\s*\{[^}]*Test[^}]*\}',     # "Test" in author names
        r'@article\{[^}]*,\s*author\s*=\s*\{[^}]*Placeholder[^}]*\}',  # "Placeholder" in author names
        r'@article\{[^}]*,\s*title\s*=\s*\{[^}]*Example[^}]*\}',   # "Example" in titles
        r'@article\{[^}]*,\s*title\s*=\s*\{[^}]*Sample[^}]*\}',    # "Sample" in titles
        r'@article\{[^}]*,\s*journal\s*=\s*\{[^}]*Journal of Example[^}]*\}',  # Fake journal names
        r'@article\{[^}]*,\s*journal\s*=\s*\{[^}]*Example Journal[^}]*\}',
    ]
    
    for pattern in fake_patterns:
        if re.search(pattern, paper_content, re.IGNORECASE):
            issues.append("Detected potentially fake or placeholder references - replace with authentic citations")
            break
    
    # Check for suspicious author patterns (single letters, too generic)
    author_matches = re.findall(r'author\s*=\s*\{([^}]+)\}', paper_content)
    for author in author_matches:
        if re.match(r'^[A-Z]\. [A-Z]\.$', author.strip()):  # Pattern like "A. B."
            issues.append(f"Suspicious author name '{author}' - use real researcher names")
        if any(word in author.lower() for word in ['example', 'test', 'sample', 'placeholder']):
            issues.append(f"Fake author name detected: '{author}' - replace with real researchers")
    
    # Check for unrealistic publication years
    year_matches = re.findall(r'year\s*=\s*\{(\d{4})\}', paper_content)
    current_year = 2025
    for year in year_matches:
        year_int = int(year)
        if year_int > current_year:
            issues.append(f"Future publication year {year} detected - use realistic years")
        if year_int < 1900:
            issues.append(f"Unrealistic publication year {year} detected")
    
    return issues
```

### Visual Content Self-Containment Checking

```python
def _check_visual_self_containment(paper_content: str) -> List[str]:
    """Check if all visual content is self-contained within LaTeX."""
    issues = []
    
    # Check for external image includes
    external_includes = re.findall(r'\\includegraphics\[[^\]]*\]\{([^}]+)\}', paper_content)
    for include in external_includes:
        # Allow some common extensions but flag external files
        if not include.endswith(('.tex', '.tikz')) and '.' in include:
            issues.append(f"External image file reference detected: '{include}' - convert to TikZ or remove")
    
    # Check for missing TikZ usage when figures are present
    has_figures = bool(re.search(r'\\begin\{figure\}', paper_content))
    has_tikz = bool(re.search(r'\\begin\{tikzpicture\}', paper_content))
    has_pgfplots = bool(re.search(r'\\begin\{axis\}', paper_content))
    
    if has_figures and not (has_tikz or has_pgfplots):
        issues.append("Figures found but no TikZ/PGFPlots content - ensure all visuals are LaTeX-generated")
    
    # Check for tables with placeholder data
    table_content = re.findall(r'\\begin\{tabular\}.*?\\end\{tabular\}', paper_content, re.DOTALL)
    for table in table_content:
        if any(word in table.lower() for word in ['example', 'placeholder', 'xxx', 'tbd', 'todo']):
            issues.append("Table contains placeholder data - populate with real simulation results")
    
    return issues
```

## Complete Workflow Requirements

The enhanced workflow now enforces these requirements:

1. **Single LaTeX File**: No external dependencies
2. **Embedded References**: All bibliography contained within the file
3. **Compilable**: Must pass pdflatex compilation with error checking
4. **Authentic References**: Real, verified academic citations only
5. **Self-Contained Visuals**: All figures/tables created with LaTeX code
6. **Data-Driven Results**: Numerical values from actual simulation results
7. **LaTeX Error Handling**: Compilation errors sent to LLM for fixing

## Key Benefits

- **Academic Integrity**: No fake or placeholder references
- **Portability**: Single file with no external dependencies
- **Reproducibility**: All visual content and data traceable to simulation results
- **Quality Assurance**: Systematic validation and error checking
- **Self-Sufficiency**: Papers can be shared and compiled anywhere without missing files

This enhanced workflow ensures that every generated paper meets the highest standards for academic publishing while maintaining complete self-containment and authenticity.
