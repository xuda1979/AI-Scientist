#!/usr/bin/env python3
"""
Quality assessment and evaluation utilities for the sciresearch workflow.
"""
from __future__ import annotations
import re
from typing import Dict, Any, List, Tuple, Optional


def _extract_quality_metrics(paper_content: str, sim_summary: str) -> Dict[str, Any]:
    """Extract quantitative quality metrics from paper content."""
    metrics = {}
    
    # Basic structural metrics
    metrics['word_count'] = len(paper_content.split())
    metrics['character_count'] = len(paper_content)
    
    # Section analysis
    sections = re.findall(r'\\section\{([^}]+)\}', paper_content, re.IGNORECASE)
    subsections = re.findall(r'\\subsection\{([^}]+)\}', paper_content, re.IGNORECASE)
    metrics['section_count'] = len(sections) + len(subsections)
    
    # Content structure checks
    content_lower = paper_content.lower()
    metrics['has_abstract'] = 'abstract' in content_lower
    metrics['has_introduction'] = any(word in content_lower for word in ['introduction', 'intro'])
    metrics['has_methodology'] = any(word in content_lower for word in ['methodology', 'methods', 'approach'])
    metrics['has_results'] = 'results' in content_lower
    metrics['has_discussion'] = 'discussion' in content_lower
    metrics['has_conclusion'] = 'conclusion' in content_lower
    metrics['has_related_work'] = any(word in content_lower for word in ['related work', 'literature review', 'background'])
    
    # Citation and reference analysis
    citations = re.findall(r'\\cite\{[^}]+\}', paper_content)
    bibliography_items = re.findall(r'\\bibitem\{[^}]+\}', paper_content)
    metrics['citation_count'] = len(citations)
    metrics['reference_count'] = len(bibliography_items)
    
    # Figure and table analysis
    figures = re.findall(r'\\begin\{figure\}', paper_content, re.IGNORECASE)
    tables = re.findall(r'\\begin\{table\}', paper_content, re.IGNORECASE)
    metrics['figure_count'] = len(figures)
    metrics['table_count'] = len(tables)
    
    # Equation analysis
    equations = re.findall(r'\\begin\{equation\}|\$\$', paper_content)
    metrics['equation_count'] = len(equations)
    
    # Simulation analysis
    metrics['has_simulation'] = 'simulation.py' in paper_content.lower() or bool(sim_summary.strip())
    metrics['simulation_success'] = 'error' not in sim_summary.lower() if sim_summary else False
    
    # LaTeX structure analysis
    metrics['has_document_class'] = '\\documentclass' in paper_content
    metrics['has_begin_document'] = '\\begin{document}' in paper_content
    metrics['has_end_document'] = '\\end{document}' in paper_content
    
    return metrics


def _calculate_quality_score(metrics: Dict[str, Any], issues: List[str]) -> float:
    """Calculate a quality score based on metrics and issues."""
    score = 0.0
    
    # Structural completeness (0-40 points)
    if metrics.get('has_abstract'): score += 5
    if metrics.get('has_related_work'): score += 5
    if metrics.get('has_methodology'): score += 10
    if metrics.get('has_results'): score += 10
    if metrics.get('has_discussion'): score += 5
    if metrics.get('has_conclusion'): score += 5
    
    # Content richness (0-30 points)
    section_score = min(metrics.get('section_count', 0) * 2, 10)
    citation_score = min(metrics.get('citation_count', 0), 10)
    figure_table_score = min((metrics.get('figure_count', 0) + metrics.get('table_count', 0)) * 2, 10)
    score += section_score + citation_score + figure_table_score
    
    # Technical quality (0-20 points)
    if metrics.get('has_simulation'): score += 10
    if metrics.get('simulation_success'): score += 10
    
    # LaTeX structure (0-10 points)
    if metrics.get('has_document_class'): score += 3
    if metrics.get('has_begin_document'): score += 3
    if metrics.get('has_end_document'): score += 4
    
    # Issue penalty (0-10 points deduction)
    issue_penalty = min(len(issues) * 2, 10)
    score -= issue_penalty
    
    # Normalize to 0-1 scale
    return max(0.0, min(1.0, score / 100.0))


def _validate_research_quality(paper_content: str, sim_summary: str) -> List[str]:
    """Validate research quality and return list of issues."""
    issues = []
    
    # Check for filename references (security issue)
    filename_patterns = [
        r'\w+\.csv', r'\w+\.txt', r'\w+\.dat', r'\w+\.json',
        r'\w+\.xlsx?', r'\w+\.png', r'\w+\.jpg', r'\w+\.pdf'
    ]
    
    for pattern in filename_patterns:
        matches = re.findall(pattern, paper_content, re.IGNORECASE)
        for match in matches:
            # Skip LaTeX-specific files and common academic terms
            if not any(skip in match.lower() for skip in ['paper.tex', 'main.tex', 'document', 'figure', 'table']):
                issues.append(f"CRITICAL: Filenames found in paper text: {match}")
    
    # Check figure and table quality
    figures = re.findall(r'\\includegraphics.*?\{([^}]+)\}', paper_content)
    for fig in figures:
        if 'width=' not in paper_content or 'height=' not in paper_content:
            issues.append("Figures should include size constraints (width/height)")
    
    # Check for data visualization issues
    if 'plot' in sim_summary.lower() or 'chart' in sim_summary.lower():
        # Look for insufficient data points
        data_matches = re.findall(r'(\d+)\s*data\s*points?', sim_summary, re.IGNORECASE)
        for match in data_matches:
            if int(match) < 5:
                issues.append(f"CRITICAL: Plot has insufficient data points ({match} found, minimum 5 required)")
    
    # Check paper type and required sections
    content_lower = paper_content.lower()
    
    # Determine paper type based on content
    paper_type = _classify_paper_type(content_lower, paper_content)
    
    if paper_type == "security":
        if not any(term in content_lower for term in ['security', 'attack', 'defense', 'vulnerability']):
            issues.append("Security paper should include security analysis, attack scenarios, or defense mechanisms")
    
    elif paper_type == "survey":
        if not any(term in content_lower for term in ['future work', 'future research', 'research directions']):
            issues.append("Survey paper should include future work or research directions section")
    
    elif paper_type == "experimental":
        if not any(term in content_lower for term in ['experiment', 'evaluation', 'results', 'performance']):
            issues.append("Experimental paper should include comprehensive experimental evaluation")
    
    # Check section ordering
    sections = re.findall(r'\\section\{([^}]+)\}', paper_content, re.IGNORECASE)
    if sections:
        section_titles = [s.lower() for s in sections]
        
        # Check if conclusion appears near the end
        conclusion_pos = next((i for i, title in enumerate(section_titles) if 'conclusion' in title), None)
        if conclusion_pos is not None and conclusion_pos < len(section_titles) - 2:
            issues.append("Conclusion section should appear near the end of the paper")
        
        # Check if introduction is early
        intro_pos = next((i for i, title in enumerate(section_titles) if 'introduction' in title), None)
        if intro_pos is not None and intro_pos > 2:
            issues.append("Introduction should appear early in the paper")
    
    # Check reference quality
    bibliography_pattern = r'\\bibitem\{[^}]+\}\s*([^\n]+)'
    references = re.findall(bibliography_pattern, paper_content)
    
    fake_indicators = ['example', 'sample', 'placeholder', 'fake', 'dummy', 'test']
    for ref in references:
        if any(indicator in ref.lower() for indicator in fake_indicators):
            issues.append(f"CRITICAL: Potentially fake reference detected: {ref[:50]}...")
    
    # Check for minimum reference count
    if len(references) < 10:
        issues.append(f"Insufficient references ({len(references)} found, minimum 10 recommended)")
    
    return issues


def _classify_paper_type(content_lower: str, paper_content: str) -> str:
    """Classify the type of research paper based on content."""
    # Security/cybersecurity papers
    security_keywords = ['security', 'attack', 'vulnerability', 'encryption', 'authentication', 
                        'malware', 'privacy', 'cryptography', 'firewall', 'intrusion']
    if sum(1 for keyword in security_keywords if keyword in content_lower) >= 3:
        return "security"
    
    # Survey/review papers
    survey_keywords = ['survey', 'review', 'overview', 'taxonomy', 'classification', 'comparison']
    if any(keyword in content_lower for keyword in survey_keywords):
        return "survey"
    
    # Machine learning papers
    ml_keywords = ['machine learning', 'deep learning', 'neural network', 'algorithm', 'model', 'training']
    if sum(1 for keyword in ml_keywords if keyword in content_lower) >= 2:
        return "machine_learning"
    
    # Experimental/empirical papers
    exp_keywords = ['experiment', 'evaluation', 'performance', 'benchmark', 'test', 'measurement']
    if sum(1 for keyword in exp_keywords if keyword in content_lower) >= 2:
        return "experimental"
    
    # Theoretical papers
    theory_keywords = ['theorem', 'proof', 'lemma', 'proposition', 'mathematical', 'formal']
    if sum(1 for keyword in theory_keywords if keyword in content_lower) >= 2:
        return "theoretical"
    
    # System papers
    system_keywords = ['system', 'architecture', 'design', 'implementation', 'framework']
    if sum(1 for keyword in system_keywords if keyword in content_lower) >= 2:
        return "systems"
    
    return "general"


def _evaluate_response_quality(response: str) -> Dict[str, float]:
    """Evaluate the quality of an AI-generated response."""
    metrics = {
        'length': 0.0,
        'structure': 0.0,
        'technical_content': 0.0,
        'references': 0.0,
        'latex_quality': 0.0
    }
    
    # Length assessment
    word_count = len(response.split())
    if word_count > 5000:
        metrics['length'] = 1.0
    elif word_count > 3000:
        metrics['length'] = 0.8
    elif word_count > 1000:
        metrics['length'] = 0.6
    else:
        metrics['length'] = 0.3
    
    # Structure assessment
    sections = len(re.findall(r'\\section\{', response, re.IGNORECASE))
    if sections >= 6:
        metrics['structure'] = 1.0
    elif sections >= 4:
        metrics['structure'] = 0.8
    elif sections >= 2:
        metrics['structure'] = 0.6
    else:
        metrics['structure'] = 0.3
    
    # Technical content assessment
    technical_indicators = ['algorithm', 'method', 'approach', 'analysis', 'evaluation', 
                          'experiment', 'result', 'performance', 'optimization']
    tech_count = sum(1 for indicator in technical_indicators if indicator in response.lower())
    metrics['technical_content'] = min(1.0, tech_count / 8.0)
    
    # Reference assessment
    bibitem_count = len(re.findall(r'\\bibitem', response))
    cite_count = len(re.findall(r'\\cite\{', response))
    
    if bibitem_count >= 15 and cite_count >= 10:
        metrics['references'] = 1.0
    elif bibitem_count >= 10 and cite_count >= 5:
        metrics['references'] = 0.8
    elif bibitem_count >= 5:
        metrics['references'] = 0.6
    else:
        metrics['references'] = 0.3
    
    # LaTeX quality assessment
    latex_elements = ['\\documentclass', '\\begin{document}', '\\end{document}', 
                     '\\begin{figure}', '\\begin{table}', '\\label{', '\\ref{']
    latex_score = sum(1 for element in latex_elements if element in response) / len(latex_elements)
    metrics['latex_quality'] = latex_score
    
    return metrics


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
    
    # Check for figure positioning issues
    bib_start_patterns = [
        r'\\begin\{thebibliography\}',
        r'\\bibliography\{',
        r'\\section\*?\{.*?[Rr]eferences.*?\}',
        r'\\section\*?\{.*?[Bb]ibliography.*?\}'
    ]
    
    bib_positions = []
    for pattern in bib_start_patterns:
        matches = list(re.finditer(pattern, paper_content, re.IGNORECASE))
        bib_positions.extend([match.start() for match in matches])
    
    if bib_positions:
        bib_start = min(bib_positions)
        figures_after_bib = re.finditer(r'\\begin\{figure\}', paper_content[bib_start:])
        if any(figures_after_bib):
            issues.append("CRITICAL: Figures found after bibliography/references section - figures must be positioned before references")
    
    # Check for adequate data samples in plots (minimum 5 samples)
    if has_pgfplots:
        plot_coords = re.findall(r'\\addplot.*?coordinates\s*\{([^}]+)\}', paper_content, re.DOTALL)
        for coords in plot_coords:
            coord_pairs = re.findall(r'\([^)]+\)', coords)
            if len(coord_pairs) < 5:
                issues.append(f"CRITICAL: Plot has insufficient data points ({len(coord_pairs)} found, minimum 5 required)")
        
        # Look for table-based plots and check CSV data
        table_plots = re.findall(r'\\addplot.*?table.*?\{([^}]+)\}', paper_content)
        for table_ref in table_plots:
            csv_pattern = rf'\\begin\{{filecontents\*?\}}\{{{re.escape(table_ref)}\}}(.*?)\\end\{{filecontents\*?\}}' 
            csv_matches = re.search(csv_pattern, paper_content, re.DOTALL)
            if csv_matches:
                csv_content = csv_matches.group(1)
                data_lines = [line.strip() for line in csv_content.strip().split('\n') if line.strip() and not line.startswith('%')]
                if len(data_lines) < 6:  # Header + 5 data rows minimum
                    issues.append(f"CRITICAL: CSV data in '{table_ref}' has insufficient samples ({len(data_lines)-1} data rows, minimum 5 required)")
    
    return issues

def _evaluate_revision_quality(revised_text: str, review_text: str, latex_errors: str) -> Dict[str, float]:
    """
    Evaluate the quality of a revision candidate.
    
    Args:
        revised_text: The revised paper content
        review_text: Original review feedback
        latex_errors: LaTeX compilation errors
    
    Returns:
        Dictionary of quality metrics
    """
    metrics = {}
    
    # Length and detail score (good revisions often add content)
    metrics['length_score'] = min(len(revised_text) / 15000, 1.0)  # Reasonable paper length
    
    # LaTeX structure and formatting
    latex_indicators = ['\\begin{', '\\end{', '\\section', '\\subsection', '\\cite{', 
                       '\\ref{', '\\label{', '\\begin{equation}', '\\begin{figure}', '\\begin{table}']
    latex_count = sum(1 for indicator in latex_indicators if indicator in revised_text)
    metrics['latex_structure'] = min(latex_count / 15, 1.0)
    
    # Academic content quality
    academic_terms = ['methodology', 'analysis', 'results', 'discussion', 'conclusion',
                     'evaluation', 'experiment', 'validation', 'algorithm', 'approach',
                     'performance', 'comparison', 'implementation', 'framework']
    academic_count = sum(1 for term in academic_terms if term.lower() in revised_text.lower())
    metrics['academic_depth'] = min(academic_count / 10, 1.0)
    
    # Reference and citation quality
    citation_count = revised_text.count('\\cite{')
    metrics['citation_quality'] = min(citation_count / 20, 1.0)
    
    # Mathematical rigor
    math_indicators = ['\\begin{equation}', '\\begin{align}', 'theorem', 'proof', 'lemma']
    math_count = sum(1 for indicator in math_indicators if indicator.lower() in revised_text.lower())
    metrics['mathematical_rigor'] = min(math_count / 8, 1.0)
    
    # Review feedback addressing (keyword matching)
    if review_text:
        # Extract key concerns from review
        review_keywords = re.findall(r'\b(?:improve|add|clarify|expand|strengthen|include|enhance|better|more|detail)\w*\b', review_text.lower())
        addressed_count = sum(1 for keyword in review_keywords if keyword in revised_text.lower())
        metrics['review_addressing'] = min(addressed_count / max(len(review_keywords), 1), 1.0)
    else:
        metrics['review_addressing'] = 0.5  # Neutral score if no review
    
    # LaTeX error penalty
    if latex_errors:
        error_penalty = min(len(latex_errors) / 1000, 0.3)  # Up to 30% penalty
        metrics['error_penalty'] = error_penalty
    else:
        metrics['error_penalty'] = 0.0
    
    # Overall quality score (weighted combination)
    metrics['overall_quality'] = (
        metrics['length_score'] * 0.15 +
        metrics['latex_structure'] * 0.20 +
        metrics['academic_depth'] * 0.20 +
        metrics['citation_quality'] * 0.10 +
        metrics['mathematical_rigor'] * 0.10 +
        metrics['review_addressing'] * 0.25 -
        metrics['error_penalty']
    )
    
    # Ensure overall quality is between 0 and 1
    metrics['overall_quality'] = max(0.0, min(1.0, metrics['overall_quality']))
    
    return metrics

def _validate_doi_with_crossref(doi: str) -> bool:
    """
    Validate a DOI using the Crossref API.
    
    Args:
        doi: DOI string to validate
    
    Returns:
        True if DOI is valid and exists, False otherwise
    """
    import urllib.request
    import json
    
    if not doi:
        return False
    
    # Clean DOI
    clean_doi = doi.strip().replace("doi:", "").replace("https://doi.org/", "")
    
    try:
        # Query Crossref API
        url = f"https://api.crossref.org/works/{clean_doi}"
        headers = {
            'User-Agent': 'SciResearchWorkflow/1.0 (mailto:research@example.com)'
        }
        
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, timeout=10) as response:
            if response.status == 200:
                data = json.loads(response.read().decode())
                return data.get('status') == 'ok' and 'message' in data
            return False
            
    except Exception as e:
        print(f"DOI validation error for {doi}: {e}")
        return False

def _extract_simulation_code_with_validation(paper_content: str) -> Tuple[Optional[str], List[str]]:
    """
    Extract simulation code from paper content and validate it.
    
    Args:
        paper_content: Full paper LaTeX content
    
    Returns:
        Tuple of (simulation_code, validation_issues)
    """
    issues = []
    
    # Extract Python code blocks
    code_blocks = re.findall(
        r'\\begin{lstlisting}(?:\[.*?\])?(.*?)\\end{lstlisting}', 
        paper_content, 
        re.DOTALL
    )
    
    if not code_blocks:
        # Try alternative code environments
        code_blocks = re.findall(
            r'\\begin{verbatim}(.*?)\\end{verbatim}', 
            paper_content, 
            re.DOTALL
        )
        
        if not code_blocks:
            issues.append("No simulation code blocks found in paper")
            return None, issues
    
    # Find the largest/most complete code block
    simulation_code = max(code_blocks, key=len) if code_blocks else None
    
    if not simulation_code:
        return None, issues
    
    simulation_code = simulation_code.strip()
    
    # Validate Python code
    try:
        import ast
        ast.parse(simulation_code)
    except SyntaxError as e:
        issues.append(f"Simulation code has syntax errors: {e}")
    
    # Check for essential imports
    required_imports = ['numpy', 'matplotlib']
    for imp in required_imports:
        if f'import {imp}' not in simulation_code and f'from {imp}' not in simulation_code:
            issues.append(f"Missing essential import: {imp}")
    
    # Check for main execution logic
    if 'if __name__' not in simulation_code and 'def main(' not in simulation_code:
        issues.append("Simulation code lacks main execution entry point")
    
    # Check for data generation/analysis
    analysis_keywords = ['plot', 'figure', 'results', 'experiment', 'simulation']
    if not any(keyword in simulation_code.lower() for keyword in analysis_keywords):
        issues.append("Simulation code lacks analysis or visualization components")
    
    return simulation_code, issues

def _validate_figures_tables(paper_content: str) -> List[str]:
    """
    Validate figures and tables in the paper content.
    
    Args:
        paper_content: Full paper LaTeX content
    
    Returns:
        List of validation issues
    """
    issues = []
    
    # Check figure references
    figure_refs = re.findall(r'\\ref{([^}]*fig[^}]*)}', paper_content, re.IGNORECASE)
    figure_labels = re.findall(r'\\label{([^}]*fig[^}]*)}', paper_content, re.IGNORECASE)
    
    for ref in figure_refs:
        if ref not in figure_labels:
            issues.append(f"Figure reference \\ref{{{ref}}} has no corresponding \\label")
    
    # Check table references
    table_refs = re.findall(r'\\ref{([^}]*tab[^}]*)}', paper_content, re.IGNORECASE)
    table_labels = re.findall(r'\\label{([^}]*tab[^}]*)}', paper_content, re.IGNORECASE)
    
    for ref in table_refs:
        if ref not in table_labels:
            issues.append(f"Table reference \\ref{{{ref}}} has no corresponding \\label")
    
    # Check for figures without captions
    figure_envs = re.findall(r'\\begin{figure}(.*?)\\end{figure}', paper_content, re.DOTALL)
    for i, fig_content in enumerate(figure_envs):
        if '\\caption{' not in fig_content:
            issues.append(f"Figure {i+1} is missing a caption")
    
    # Check for tables without captions
    table_envs = re.findall(r'\\begin{table}(.*?)\\end{table}', paper_content, re.DOTALL)
    for i, table_content in enumerate(table_envs):
        if '\\caption{' not in table_content:
            issues.append(f"Table {i+1} is missing a caption")
    
    # Check for orphaned figures/tables (no references in text)
    all_figure_labels = re.findall(r'\\begin{figure}.*?\\label{([^}]+)}', paper_content, re.DOTALL)
    for label in all_figure_labels:
        if f'\\ref{{{label}}}' not in paper_content:
            issues.append(f"Figure with label '{label}' is not referenced in the text")
    
    all_table_labels = re.findall(r'\\begin{table}.*?\\label{([^}]+)}', paper_content, re.DOTALL)
    for label in all_table_labels:
        if f'\\ref{{{label}}}' not in paper_content:
            issues.append(f"Table with label '{label}' is not referenced in the text")
    
    return issues

def _validate_bibliography(paper_content: str) -> List[str]:
    """
    Validate bibliography and references in the paper content.
    
    Args:
        paper_content: Full paper LaTeX content
    
    Returns:
        List of validation issues
    """
    issues = []
    
    # Check for bibliography section
    has_thebibliography = '\\begin{thebibliography}' in paper_content
    has_bibliography_cmd = '\\bibliography{' in paper_content
    has_filecontents_bib = '\\begin{filecontents}' in paper_content and '.bib}' in paper_content
    
    if not (has_thebibliography or has_bibliography_cmd or has_filecontents_bib):
        issues.append("No bibliography section found - paper must include references")
        return issues
    
    # Extract citations from text
    citations = re.findall(r'\\cite(?:p|t|author)?\{([^}]+)\}', paper_content)
    all_cite_keys = []
    for citation in citations:
        all_cite_keys.extend([key.strip() for key in citation.split(',')])
    
    if not all_cite_keys:
        issues.append("No citations found in paper text")
    
    # Extract bibliography entries
    bib_entries = []
    
    if has_thebibliography:
        # Parse \bibitem entries
        bibitem_pattern = r'\\bibitem\{([^}]+)\}'
        bib_entries = re.findall(bibitem_pattern, paper_content)
    elif has_filecontents_bib:
        # Parse .bib entries
        filecontents_pattern = r'\\begin{filecontents\*?}\{[^}]*\.bib\}(.*?)\\end{filecontents\*?}'
        bib_content_match = re.search(filecontents_pattern, paper_content, re.DOTALL)
        if bib_content_match:
            bib_content = bib_content_match.group(1)
            bib_entries = re.findall(r'@\w+\{([^,]+),', bib_content)
    
    # Check for missing bibliography entries
    for cite_key in set(all_cite_keys):
        if cite_key not in bib_entries:
            issues.append(f"Citation '{cite_key}' has no corresponding bibliography entry")
    
    # Check for unused bibliography entries
    for bib_key in bib_entries:
        if bib_key not in all_cite_keys:
            issues.append(f"Bibliography entry '{bib_key}' is not cited in the text")
    
    # Check minimum number of references
    if len(bib_entries) < 10:
        issues.append(f"Insufficient references ({len(bib_entries)} found, minimum 10 recommended)")
    
    return issues

def _evaluate_simulation_content(text: str) -> float:
    """
    Evaluate the quality of simulation content in the paper.
    
    Args:
        text: Paper content to evaluate
        
    Returns:
        Simulation quality score (0.0 to 1.0)
    """
    score = 0.0
    
    # Check for filecontents simulation blocks
    if '\\begin{filecontents' in text and 'simulation.py' in text:
        score += 0.3
        
        # Extract simulation content from filecontents
        sim_pattern = r'\\begin\{filecontents\*?\}\{simulation\.py\}(.*?)\\end\{filecontents\*?\}'
        matches = re.findall(sim_pattern, text, re.DOTALL | re.IGNORECASE)
        
        if matches:
            sim_content = matches[0]
            
            # Check for comprehensive simulation indicators
            quality_indicators = [
                ('class ', 0.15),  # Object-oriented structure
                ('def ', 0.1),     # Function definitions
                ('import ', 0.1),  # Module imports
                ('numpy', 0.05),   # Scientific computing
                ('matplotlib', 0.05),  # Visualization
                ('experiment', 0.1),   # Experimental design
                ('result', 0.05),  # Results generation
                ('main()', 0.1),   # Main execution
                ('if __name__', 0.1),  # Proper script structure
                ('test', 0.05),    # Testing/validation
                ('scaling', 0.1),  # Scaling analysis
                ('candidate', 0.1), # Candidate generation (test-time scaling)
                ('quality', 0.05), # Quality evaluation
                ('algorithm', 0.05), # Algorithmic content
                ('performance', 0.05)  # Performance analysis
            ]
            
            for indicator, weight in quality_indicators:
                if indicator.lower() in sim_content.lower():
                    score += weight
            
            # Length bonus for substantial simulations
            sim_lines = len([line for line in sim_content.split('\n') 
                           if line.strip() and not line.strip().startswith('#')])
            if sim_lines > 50:
                score += 0.1
            if sim_lines > 100:
                score += 0.1
            if sim_lines > 200:
                score += 0.1
    
    # Check for simulation discussion in paper text
    simulation_terms = ['simulation', 'experiment', 'implementation', 'algorithm',
                       'numerical', 'computational', 'benchmark', 'evaluation']
    term_count = sum(1 for term in simulation_terms if term.lower() in text.lower())
    score += min(term_count / 20, 0.2)  # Up to 0.2 bonus for simulation discussion
    
    return min(score, 1.0)  # Cap at 1.0
