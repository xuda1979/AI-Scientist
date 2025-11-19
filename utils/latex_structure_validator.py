"""
LaTeX Structure Validator

Detects common structural issues in LaTeX files that prevent proper compilation
and bibliography rendering. These issues are automatically added to quality_issues
so the AI knows to fix them.
"""

import re
from pathlib import Path
from typing import List, Tuple


def validate_latex_structure(tex_content: str, tex_path: Path = None) -> List[str]:
    """
    Validate LaTeX file structure and return list of critical issues.
    
    Args:
        tex_content: The LaTeX file content
        tex_path: Optional path to the .tex file for context
        
    Returns:
        List of issue descriptions that should be fixed
    """
    issues = []
    lines = tex_content.split('\n')
    
    # Check 1: Content before \begin{filecontents*}
    first_line = lines[0].strip() if lines else ""
    if first_line and not first_line.startswith('\\begin{filecontents*}'):
        issues.append(
            "CRITICAL LATEX STRUCTURE: File must start with \\begin{filecontents*}{refs.bib} on line 1. "
            f"Currently starts with: {first_line[:80]}"
        )
    
    # Check 2: Find filecontents block
    filecontents_start = None
    filecontents_end = None
    documentclass_line = None
    
    for i, line in enumerate(lines):
        if '\\begin{filecontents*}' in line:
            if filecontents_start is None:
                filecontents_start = i
        if '\\end{filecontents*}' in line:
            if filecontents_end is None:
                filecontents_end = i
        if '\\documentclass' in line:
            if documentclass_line is None:
                documentclass_line = i
    
    # Check 3: filecontents closure before documentclass
    if filecontents_start is not None and documentclass_line is not None:
        if filecontents_end is None:
            issues.append(
                "CRITICAL LATEX STRUCTURE: Missing \\end{filecontents*} before \\documentclass. "
                "The filecontents block must be properly closed."
            )
        elif filecontents_end > documentclass_line:
            issues.append(
                f"CRITICAL LATEX STRUCTURE: \\end{{filecontents*}} appears AFTER \\documentclass "
                f"(line {filecontents_end} vs {documentclass_line}). Must close filecontents before documentclass."
            )
    
    # Check 4: Bibliography entries outside filecontents block
    if filecontents_start is not None and filecontents_end is not None:
        # Check content outside filecontents block for bib entries
        before_content = '\n'.join(lines[:filecontents_start])
        after_content = '\n'.join(lines[filecontents_end + 1:])
        combined_outside = before_content + '\n' + after_content
        
        bib_pattern = r'@(?:article|inproceedings|book|techreport|misc)\{'
        outside_bibs = re.findall(bib_pattern, combined_outside)
        
        if outside_bibs:
            issues.append(
                f"CRITICAL LATEX STRUCTURE: Found {len(outside_bibs)} bibliography entries OUTSIDE the filecontents block. "
                "ALL @article/@inproceedings entries must be inside \\begin{filecontents*}...\\end{filecontents*}"
            )
    
    # Check 5: Content between \end{filecontents*} and \documentclass
    if filecontents_end is not None and documentclass_line is not None:
        between_content = '\n'.join(lines[filecontents_end + 1:documentclass_line]).strip()
        # Remove comments and empty lines
        between_content_clean = '\n'.join([l for l in between_content.split('\n') 
                                           if l.strip() and not l.strip().startswith('%')])
        if between_content_clean:
            issues.append(
                f"CRITICAL LATEX STRUCTURE: Found content between \\end{{filecontents*}} and \\documentclass. "
                f"This area should be empty. Found: {between_content_clean[:100]}"
            )
    
    # Check 6: Duplicate environment end tags
    duplicate_patterns = [
        (r'\\end\{abstract\}', 'abstract'),
        (r'\\end\{figure\}', 'figure'),
        (r'\\end\{table\}', 'table'),
        (r'\\end\{document\}', 'document'),
    ]
    
    for pattern, env_name in duplicate_patterns:
        matches = list(re.finditer(pattern, tex_content))
        if len(matches) > 50:  # More than reasonable for a paper
            issues.append(
                f"CRITICAL LATEX STRUCTURE: Found {len(matches)} instances of \\end{{{env_name}}}. "
                f"This likely indicates duplicate/malformed end tags that must be removed."
            )
    
    # Check 7: Content before \begin{document}
    if documentclass_line is not None:
        begin_document_line = None
        for i, line in enumerate(lines):
            if '\\begin{document}' in line:
                begin_document_line = i
                break
        
        if begin_document_line:
            # Check for text content (not commands) between documentclass and begin{document}
            preamble = '\n'.join(lines[documentclass_line + 1:begin_document_line])
            # Look for non-command text (paragraphs, citations, etc.)
            suspicious_content = []
            for line in preamble.split('\n'):
                stripped = line.strip()
                # Skip empty, comments, and LaTeX commands
                if (stripped and 
                    not stripped.startswith('%') and 
                    not stripped.startswith('\\') and
                    len(stripped) > 20):  # Substantial text
                    suspicious_content.append(stripped[:80])
            
            if suspicious_content:
                issues.append(
                    f"CRITICAL LATEX STRUCTURE: Found paragraph text in preamble (between \\documentclass and \\begin{{document}}). "
                    f"Content should start AFTER \\begin{{document}}. Found: {suspicious_content[0]}"
                )
    
    # Check 8: Citation coverage
    if filecontents_start is not None and filecontents_end is not None:
        # Count bib entries in filecontents
        filecontents_content = '\n'.join(lines[filecontents_start:filecontents_end + 1])
        bib_entries = re.findall(r'@(?:article|inproceedings|book|techreport|misc)\{([a-zA-Z0-9_]+),', 
                                  filecontents_content)
        
        # Count citations in document
        cite_pattern = r'\\cite\{([^}]+)\}'
        citations = re.findall(cite_pattern, tex_content)
        cited_keys = set()
        for cite in citations:
            cited_keys.update([k.strip() for k in cite.split(',')])
        
        uncited_count = len(set(bib_entries)) - len(cited_keys)
        if uncited_count > 3:  # Allow a few uncited references
            issues.append(
                f"CRITICAL BIBLIOGRAPHY: Bibliography has {len(bib_entries)} entries but only "
                f"{len(cited_keys)} unique keys are cited in text. {uncited_count} references are never cited. "
                "Add \\cite{} commands throughout the paper to cite all relevant references."
            )
    
    return issues


def validate_pdf_visual_issues(tex_path: Path, pdf_path: Path = None) -> List[str]:
    """
    Attempt to detect visual issues that would appear in the PDF.
    This is a heuristic check based on LaTeX patterns.
    
    Args:
        tex_path: Path to .tex file
        pdf_path: Optional path to compiled PDF
        
    Returns:
        List of potential visual issues
    """
    issues = []
    
    if not tex_path.exists():
        return issues
    
    tex_content = tex_path.read_text(encoding='utf-8')
    
    # Check for empty/broken figure environments
    # Look for figures without \includegraphics
    figure_pattern = r'\\begin\{figure\}.*?\\end\{figure\}'
    figures = re.findall(figure_pattern, tex_content, re.DOTALL)
    
    for i, fig in enumerate(figures, 1):
        if '\\includegraphics' not in fig and 'tikzpicture' not in fig and 'plot' not in fig.lower():
            issues.append(
                f"PDF VISUAL ISSUE: Figure {i} has no \\includegraphics or plot content. "
                "This will appear as an empty/black box in the PDF. "
                "Ensure all figures have actual content to display."
            )
    
    # Check for missing required packages for plots
    has_plots = 'tikzpicture' in tex_content or '\\addplot' in tex_content
    has_pgfplots = '\\usepackage{pgfplots}' in tex_content or '\\usepackage[' in tex_content and 'pgfplots' in tex_content
    
    if has_plots and not has_pgfplots:
        issues.append(
            "PDF VISUAL ISSUE: Document uses TikZ plots but doesn't load pgfplots package. "
            "Add \\usepackage{pgfplots} to preamble to render plots correctly."
        )
    
    return issues


if __name__ == "__main__":
    # Test with a malformed file
    test_content = """  author={Someone},
  title={Test}
@article{test2023,
  title={Another},
  author={Person}
}
\\begin{filecontents*}{refs.bib}
@article{real2023,
  title={Real Entry},
  author={Author}
}
\\end{filecontents*}
\\documentclass{article}
Some text here before begin document
\\begin{document}
Hello \\cite{real2023}
\\end{document}
"""
    
    issues = validate_latex_structure(test_content)
    print("Detected Issues:")
    for i, issue in enumerate(issues, 1):
        print(f"{i}. {issue}")
