"""
Content Depth and Quality Validator

This module detects superficial content that makes papers look unprofessional:
1. Short, shallow sections lacking depth
2. Simple, basic graphs and tables
3. Simplistic or missing mathematical derivations
"""

import re
from typing import List, Tuple, Optional, Dict
from pathlib import Path


def detect_shallow_sections(paper_content: str) -> List[str]:
    """
    Detect sections that are too short or lack depth.
    
    Returns:
        List of issues found
    """
    issues = []
    
    # Extract all sections with their content
    section_pattern = r'\\section\*?\{([^}]+)\}(.*?)(?=\\section|\\bibliography|\\end\{document\}|$)'
    sections = re.findall(section_pattern, paper_content, re.DOTALL | re.IGNORECASE)
    
    shallow_sections = []
    for section_name, section_content in sections:
        # Skip abstract, acknowledgments, references
        skip_sections = ['abstract', 'acknowledgment', 'reference', 'bibliography']
        if any(skip.lower() in section_name.lower() for skip in skip_sections):
            continue
        
        # Count words (excluding LaTeX commands)
        text_only = re.sub(r'\\[a-zA-Z]+(\[[^\]]*\])?(\{[^}]*\})?', '', section_content)
        text_only = re.sub(r'[{}\\]', '', text_only)
        words = len(text_only.split())
        
        # Check for subsections
        subsection_count = len(re.findall(r'\\subsection', section_content))
        
        # Check paragraph count (separated by blank lines)
        paragraphs = [p.strip() for p in text_only.split('\n\n') if p.strip()]
        paragraph_count = len(paragraphs)
        
        # Shallow section criteria
        if words < 200:  # Very short
            shallow_sections.append((section_name, words, 'too_short'))
        elif words < 400 and subsection_count == 0:  # Short without structure
            shallow_sections.append((section_name, words, 'lacks_structure'))
        elif paragraph_count < 3:  # Too few paragraphs
            shallow_sections.append((section_name, paragraph_count, 'few_paragraphs'))
    
    if shallow_sections:
        short_sections = [s for s in shallow_sections if s[2] == 'too_short']
        if short_sections:
            section_names = ', '.join([f"'{s[0]}' ({s[1]} words)" for s in short_sections[:3]])
            issues.append(
                f"CRITICAL: Sections are too short and lack depth: {section_names}. "
                "Serious research papers require comprehensive sections (500-1500 words each) "
                "with detailed explanations, examples, and thorough analysis. "
                "ADD: More detailed explanations, multiple subsections, concrete examples, "
                "literature comparison, technical details, and comprehensive discussion."
            )
        
        unstructured = [s for s in shallow_sections if s[2] == 'lacks_structure']
        if unstructured:
            section_names = ', '.join([f"'{s[0]}'" for s in unstructured[:3]])
            issues.append(
                f"WEAKNESS: Sections lack structure: {section_names}. "
                "Add subsections to break down complex topics. Each major section should "
                "have 2-5 subsections organizing the content hierarchically."
            )
    
    # Check for minimum total paper length
    total_words = len(paper_content.split())
    if total_words < 3000:
        issues.append(
            f"CRITICAL: Paper is too short ({total_words} words). "
            "Serious research papers should be 5000-8000 words (excluding references). "
            "Expand all sections with: detailed methodology, comprehensive results analysis, "
            "thorough literature review, extensive discussion, and complete derivations."
        )
    elif total_words < 4500:
        issues.append(
            f"WEAKNESS: Paper is somewhat short ({total_words} words). "
            "Consider expanding to 5000-8000 words for comprehensive coverage. "
            "Add more detailed explanations, additional experiments, deeper analysis."
        )
    
    return issues


def detect_simple_visualizations(paper_content: str, sim_summary: str) -> List[str]:
    """
    Detect overly simple graphs and tables that lack professional quality.
    
    Returns:
        List of issues found
    """
    issues = []
    
    # Analyze tables
    table_pattern = r'\\begin\{table\}(.*?)\\end\{table\}'
    tables = re.findall(table_pattern, paper_content, re.DOTALL)
    
    simple_tables = []
    for i, table_content in enumerate(tables, 1):
        # Count rows in tabular environment
        tabular_match = re.search(r'\\begin\{tabular\}(.*?)\\end\{tabular\}', table_content, re.DOTALL)
        if tabular_match:
            tabular_text = tabular_match.group(1)
            # Count \\ (row separators)
            rows = len(re.findall(r'\\\\', tabular_text))
            
            # Count columns (from format specifier like {|c|c|c|})
            format_match = re.search(r'\{([|lcrp\d.]*)\}', tabular_text)
            if format_match:
                cols = len([c for c in format_match.group(1) if c in 'lcrp'])
            else:
                cols = 0
            
            # Simple table criteria
            if rows < 5 or cols < 3:
                simple_tables.append(i)
            
            # Check for statistical elements
            has_std = bool(re.search(r'±|\\pm|std', table_content, re.IGNORECASE))
            has_confidence = bool(re.search(r'confidence|CI|95%', table_content, re.IGNORECASE))
            has_significance = bool(re.search(r'\*\*?|\$p\$|significant', table_content, re.IGNORECASE))
            
            if not (has_std or has_confidence or has_significance):
                issues.append(
                    f"WEAKNESS: Table {i} lacks statistical rigor. "
                    "Add: standard deviations (±), confidence intervals, significance markers (*, **), "
                    "p-values. Professional tables show statistical measures, not just point estimates."
                )
    
    if simple_tables:
        issues.append(
            f"CRITICAL: {len(simple_tables)} table(s) are too simple (few rows/columns). "
            "Professional research tables should have: (1) 5+ rows showing comprehensive results, "
            "(2) 4+ columns comparing multiple methods/conditions, (3) clear hierarchical structure, "
            "(4) statistical measures (mean±std), (5) significance markers. "
            "Consider multi-part tables or consolidated results tables."
        )
    
    # Analyze figures/plots
    includegraphics_count = len(re.findall(r'\\includegraphics', paper_content))
    tikz_plot_count = len(re.findall(r'\\begin\{tikzpicture\}', paper_content))
    
    total_figures = includegraphics_count + tikz_plot_count
    
    if total_figures > 0:
        # Check for multi-panel figures (subfigure, subcaption)
        has_multipanel = bool(re.search(r'\\subfigure|\\subcaption|\\subfloat', paper_content))
        
        if not has_multipanel and total_figures < 5:
            issues.append(
                "WEAKNESS: No multi-panel figures detected. "
                "Professional papers use multi-panel figures (a), (b), (c) to show: "
                "multiple views, different conditions, comparative results, or sequential steps. "
                "Consider consolidating related plots into multi-panel figures."
            )
        
        # Check for error bars in plots
        has_errorbars = bool(re.search(r'error\s*bar|errorbar|error_y|yerr', sim_summary, re.IGNORECASE))
        
        if not has_errorbars:
            issues.append(
                "CRITICAL: Plots appear to lack error bars. "
                "Scientific plots MUST show uncertainty: error bars (std dev), "
                "confidence intervals (shaded regions), or statistical ribbons. "
                "Update simulation.py to: (1) run multiple trials, (2) calculate std dev, "
                "(3) plot with error bars using matplotlib errorbar() or fill_between()."
            )
        
        # Check for figure complexity in simulation code
        if 'matplotlib' in sim_summary or 'plt.' in sim_summary:
            has_grid = bool(re.search(r'\.grid\(|grid\s*=\s*True', sim_summary))
            has_legend = bool(re.search(r'\.legend\(|legend\s*=', sim_summary))
            has_subplots = bool(re.search(r'subplot|add_subplot', sim_summary))
            
            missing_elements = []
            if not has_grid:
                missing_elements.append("grid lines")
            if not has_legend:
                missing_elements.append("legend")
            if not has_subplots and total_figures > 1:
                missing_elements.append("subplots for multi-panel")
            
            if missing_elements:
                issues.append(
                    f"WEAKNESS: Plots missing professional elements: {', '.join(missing_elements)}. "
                    "High-quality figures include: grid lines (for readability), legends (for clarity), "
                    "proper axis labels, title if needed, and consistent styling."
                )
    
    return issues


def detect_weak_mathematics(paper_content: str, sim_summary: str) -> List[str]:
    """
    Detect simplistic or missing mathematical content.
    
    Returns:
        List of issues found
    """
    issues = []
    
    # Count equations
    equation_count = len(re.findall(r'\\begin\{equation\}', paper_content))
    equation_count += len(re.findall(r'\\begin\{align\}', paper_content))
    equation_count += len(re.findall(r'\\begin\{eqnarray\}', paper_content))
    
    # Count inline math (rough estimate)
    inline_math = len(re.findall(r'\$[^$]+\$', paper_content))
    
    # Check for theoretical content indicators
    has_theorem = bool(re.search(r'\\begin\{theorem\}|\\begin\{lemma\}|\\begin\{proposition\}', paper_content))
    has_proof = bool(re.search(r'\\begin\{proof\}|\\textbf\{Proof', paper_content))
    has_definition = bool(re.search(r'\\begin\{definition\}|\\textbf\{Definition', paper_content))
    
    # Check for derivation language
    has_derivation = bool(re.search(r'deriv|proof|show that|it follows|therefore|thus', paper_content, re.IGNORECASE))
    
    # Assess mathematical rigor
    if equation_count < 3 and not has_theorem:
        issues.append(
            "CRITICAL: Paper lacks mathematical rigor. Only {equation_count} equations found. "
            "Serious research papers need: (1) formal problem formulation with equations, "
            "(2) method description with mathematical notation, (3) theoretical analysis, "
            "(4) complexity bounds, (5) derivations showing 'why' not just 'what'. "
            "ADD: Detailed mathematical formulations, step-by-step derivations, formal definitions."
        )
    
    if equation_count > 2 and not has_derivation:
        issues.append(
            "WEAKNESS: Equations present but no derivations shown. "
            "Don't just state equations - show HOW you derive them. "
            "Add: step-by-step mathematical derivations connecting equations, "
            "explaining each transformation with 'therefore', 'substituting', 'it follows that'."
        )
    
    # Check for theoretical depth
    theoretical_indicators = [
        'theorem', 'lemma', 'proposition', 'corollary', 
        'proof', 'complexity', 'bound', 'convergence'
    ]
    
    theory_count = sum(1 for word in theoretical_indicators 
                      if word in paper_content.lower())
    
    if theory_count < 2 and 'theoretical' in paper_content.lower():
        issues.append(
            "WEAKNESS: Claims theoretical contribution but lacks formal theorems. "
            "Add: (1) Theorem statements with formal conditions, (2) Complete proofs, "
            "(3) Lemmas for intermediate results, (4) Complexity analysis with Big-O bounds. "
            "Use proper theorem environments: \\begin{theorem}...\\end{theorem}"
        )
    
    # Check algorithm complexity
    algorithm_count = len(re.findall(r'\\begin\{algorithm\}', paper_content))
    has_complexity_analysis = bool(re.search(r'O\([^)]+\)|complexity|runtime', paper_content))
    
    if algorithm_count > 0 and not has_complexity_analysis:
        issues.append(
            "CRITICAL: Algorithms presented without complexity analysis. "
            "EVERY algorithm must include: (1) Time complexity: O(...), "
            "(2) Space complexity: O(...), (3) Explanation of bottlenecks, "
            "(4) Comparison to baseline complexity. This is mandatory for algorithms."
        )
    
    # Check simulation code complexity
    if sim_summary and 'def ' in sim_summary:
        # Count function definitions
        function_count = len(re.findall(r'def\s+\w+\s*\(', sim_summary))
        
        # Check for sophisticated code elements
        has_classes = bool(re.search(r'class\s+\w+', sim_summary))
        has_numpy = bool(re.search(r'import\s+numpy|from\s+numpy', sim_summary))
        has_optimization = bool(re.search(r'optimize|minimize|argmax|argmin', sim_summary, re.IGNORECASE))
        
        if function_count < 3:
            issues.append(
                "WEAKNESS: Simulation code is too simple (only {function_count} functions). "
                "Serious research implementations should have: (1) modular function design (5+ functions), "
                "(2) class-based architecture for complex systems, (3) proper abstraction layers, "
                "(4) comprehensive helper functions. Simple scripts look amateurish."
            )
        
        if not has_numpy and 'experiment' in paper_content.lower():
            issues.append(
                "WEAKNESS: Simulation lacks numerical computing libraries. "
                "Use numpy for: efficient array operations, mathematical functions, "
                "random number generation, statistical computations. Professional code uses numpy/scipy."
            )
    
    return issues


def detect_missing_details(paper_content: str) -> List[str]:
    """
    Detect missing detailed explanations and examples.
    
    Returns:
        List of issues found
    """
    issues = []
    
    # Check for examples
    has_examples = bool(re.search(r'for example|e\.g\.|such as|instance|consider', paper_content, re.IGNORECASE))
    
    if not has_examples:
        issues.append(
            "WEAKNESS: No concrete examples provided. "
            "Add specific examples to illustrate: (1) problem instances, "
            "(2) method application, (3) result interpretation. "
            "Examples make abstract concepts concrete and accessible."
        )
    
    # Check for related work depth
    related_work_match = re.search(
        r'\\section\*?\{.*?(?:Related Work|Literature Review).*?\}(.*?)(?=\\section|$)',
        paper_content,
        re.DOTALL | re.IGNORECASE
    )
    
    if related_work_match:
        related_content = related_work_match.group(1)
        citation_count = len(re.findall(r'\\cite', related_content))
        
        if citation_count < 10:
            issues.append(
                f"WEAKNESS: Related Work section cites only {citation_count} works. "
                "Comprehensive literature review should cite: (1) 15-25 relevant papers, "
                "(2) seminal works in the field, (3) recent advances (last 3 years), "
                "(4) competing approaches, (5) complementary methods. "
                "Add more citations with detailed comparison."
            )
        
        # Check for comparison
        has_comparison = bool(re.search(r'compar|differ|contrast|unlike|whereas|however', related_content, re.IGNORECASE))
        
        if not has_comparison:
            issues.append(
                "WEAKNESS: Related Work lacks critical comparison. "
                "Don't just list prior work - COMPARE it to your approach. "
                "For each cited work, explain: (1) What they do, (2) How it differs from yours, "
                "(3) Advantages/disadvantages, (4) Why your approach is needed."
            )
    
    # Check for limitations discussion
    has_limitations = bool(re.search(r'limitation|drawback|weakness|future work', paper_content, re.IGNORECASE))
    
    if not has_limitations:
        issues.append(
            "CRITICAL: No limitations discussed. "
            "Every serious paper must acknowledge limitations: (1) Assumptions that may not hold, "
            "(2) Scenarios where method fails, (3) Computational constraints, "
            "(4) Generalization concerns. Add a Limitations section or Discussion subsection."
        )
    
    return issues


def validate_content_quality(
    paper_content: str,
    sim_summary: str,
    project_dir: Optional[Path] = None
) -> Tuple[List[str], List[str]]:
    """
    Comprehensive validation of content depth and quality.
    
    Returns:
        Tuple of (critical_issues, warnings)
    """
    critical_issues = []
    warnings = []
    
    # Run all validators
    section_issues = detect_shallow_sections(paper_content)
    viz_issues = detect_simple_visualizations(paper_content, sim_summary)
    math_issues = detect_weak_mathematics(paper_content, sim_summary)
    detail_issues = detect_missing_details(paper_content)
    
    # Categorize by severity
    all_issues = section_issues + viz_issues + math_issues + detail_issues
    
    for issue in all_issues:
        if issue.startswith("CRITICAL:"):
            critical_issues.append(issue)
        else:
            warnings.append(issue)
    
    return critical_issues, warnings


def generate_depth_improvement_prompt(critical_issues: List[str], warnings: List[str]) -> str:
    """
    Generate a prompt section for improving content depth.
    
    Returns:
        Formatted prompt text
    """
    if not critical_issues and not warnings:
        return ""
    
    prompt = "\n📚 CONTENT DEPTH & QUALITY REQUIREMENTS:\n"
    prompt += "Address the following to meet publication quality standards:\n\n"
    
    if critical_issues:
        prompt += "CRITICAL ISSUES (MUST FIX):\n"
        for i, issue in enumerate(critical_issues, 1):
            prompt += f"{i}. {issue}\n\n"
    
    if warnings:
        prompt += "WARNINGS (SHOULD ADDRESS):\n"
        for i, issue in enumerate(warnings, 1):
            prompt += f"{i}. {issue}\n\n"
    
    prompt += (
        "GENERAL DEPTH & QUALITY GUIDELINES:\n\n"
        
        "📝 SECTION DEPTH:\n"
        "- Each major section: 500-1500 words with 2-5 subsections\n"
        "- Multiple paragraphs per subsection (3-5 paragraphs)\n"
        "- Detailed explanations with concrete examples\n"
        "- Total paper length: 5000-8000 words (excluding references)\n"
        "- Comprehensive coverage with hierarchical organization\n\n"
        
        "📊 VISUALIZATION QUALITY:\n"
        "- Tables: 5+ rows, 4+ columns, show mean±std, significance markers\n"
        "- Figures: Multi-panel (a,b,c), error bars/confidence intervals\n"
        "- Plots: Grid lines, legends, proper labels, professional styling\n"
        "- Data density: Show comprehensive results, not cherry-picked examples\n"
        "- Statistical rigor: Always show uncertainty and variance\n\n"
        
        "🔢 MATHEMATICAL RIGOR:\n"
        "- Problem formulation: Formal definitions with mathematical notation\n"
        "- Derivations: Step-by-step with explanations ('therefore', 'it follows')\n"
        "- Theorems/Lemmas: Formal statements with complete proofs\n"
        "- Complexity: Big-O bounds for all algorithms with justification\n"
        "- Notation: Consistent, standard mathematical symbols throughout\n\n"
        
        "💻 CODE QUALITY:\n"
        "- Modular design: 5+ well-structured functions\n"
        "- Professional libraries: numpy, scipy for numerical work\n"
        "- Class-based: OOP for complex systems\n"
        "- Comprehensive: Full implementation, not toy examples\n\n"
        
        "📖 DETAILED EXPLANATIONS:\n"
        "- Concrete examples illustrating concepts\n"
        "- Related Work: 15-25 citations with critical comparison\n"
        "- Limitations: Honest discussion of when method fails\n"
        "- Future Work: Specific research directions\n\n"
    )
    
    return prompt
