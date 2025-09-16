#!/usr/bin/env python3
"""
LaTeX compilation and PDF generation utilities.
"""
from __future__ import annotations
import re
import subprocess
from pathlib import Path
from typing import Tuple, Optional, List


def _calculate_dynamic_timeout(tex_content: str, config) -> int:
    """Calculate dynamic timeout based on LaTeX content complexity."""
    base_timeout = 120
    
    # Count complexity indicators
    tikz_count = len(re.findall(r'\\begin\{tikzpicture\}', tex_content, re.IGNORECASE))
    table_count = len(re.findall(r'\\begin\{table\}', tex_content, re.IGNORECASE))
    figure_count = len(re.findall(r'\\begin\{figure\}', tex_content, re.IGNORECASE))
    
    # Estimate page count (rough approximation)
    estimated_pages = len(tex_content) // 3000  # ~3000 chars per page
    estimated_pages = max(1, min(estimated_pages, 50))  # Cap at reasonable range
    
    # Calculate dynamic timeout
    complexity_timeout = (tikz_count * 30) + (table_count * 10) + (figure_count * 5)
    page_timeout = estimated_pages * 8
    
    total_timeout = base_timeout + complexity_timeout + page_timeout
    
    # Cap at reasonable maximum
    total_timeout = min(total_timeout, 600)  # 10 minutes max
    
    print(f"Dynamic timeout calculated: {total_timeout}s (TikZ:{tikz_count}, Tables:{table_count}, Figures:{figure_count}, Pages:{estimated_pages})")
    return total_timeout


def _compile_latex_and_get_errors(paper_path: Path, timeout: int = 120) -> Tuple[bool, str]:
    """Compile LaTeX and return success status and error messages."""
    working_dir = paper_path.parent
    
    try:
        # Run pdflatex with timeout
        result = subprocess.run(
            ["pdflatex", "-interaction=nonstopmode", paper_path.name],
            cwd=working_dir,
            capture_output=True,
            text=True,
            timeout=timeout,
            encoding='utf-8',
            errors='ignore'
        )
        
        # Check if PDF was generated successfully
        pdf_path = paper_path.with_suffix('.pdf')
        latex_success = pdf_path.exists() and pdf_path.stat().st_size > 0
        
        if latex_success:
            return True, ""
        else:
            # Extract error information from log file and .compile.log (terminal output)
            log_path = paper_path.with_suffix('.log')
            compile_log_path = paper_path.with_suffix('.compile.log')
            error_log = ""
            # Read .log file (standard)
            if log_path.exists():
                try:
                    log_content = log_path.read_text(encoding='utf-8', errors='ignore')
                    log_lines = log_content.split('\n')
                    error_log = '\n'.join(log_lines[-20:])
                except Exception as e:
                    error_log = f"Could not read log file: {e}"
            else:
                error_log = "No log file generated"
            # Read .compile.log file (terminal output)
            compile_log_tail = ""
            if compile_log_path.exists():
                try:
                    compile_log_content = compile_log_path.read_text(encoding='utf-8', errors='ignore')
                    compile_log_lines = compile_log_content.split('\n')
                    compile_log_tail = '\n'.join(compile_log_lines[-20:])
                except Exception as e:
                    compile_log_tail = f"Could not read .compile.log file: {e}"
            if compile_log_tail:
                error_log += f"\n\n[Terminal output: .compile.log tail]\n{compile_log_tail}"
            # Also include stderr if available
            if result.stderr:
                error_log += f"\n\nStderr:\n{result.stderr}"
            return False, error_log
            
    except subprocess.TimeoutExpired:
        return False, f"LaTeX compilation timed out after {timeout} seconds"
    except FileNotFoundError:
        return False, "pdflatex not found. Please install LaTeX (e.g., TeX Live, MiKTeX)"
    except Exception as e:
        return False, f"LaTeX compilation error: {str(e)}"


def _generate_pdf_for_review(paper_path: Path, timeout: int = 120) -> Tuple[bool, Optional[Path], str]:
    """Generate PDF specifically for AI review purposes."""
    working_dir = paper_path.parent
    
    try:
        # Run pdflatex multiple times for proper cross-references
        for run_num in range(2):  # Usually 2 runs are enough
            result = subprocess.run(
                ["pdflatex", "-interaction=nonstopmode", paper_path.name],
                cwd=working_dir,
                capture_output=True,
                text=True,
                timeout=timeout,
                encoding='utf-8',
                errors='ignore'
            )
        
        # Check if PDF was generated successfully
        pdf_path = paper_path.with_suffix('.pdf')
        if pdf_path.exists() and pdf_path.stat().st_size > 0:
            return True, pdf_path, ""
        else:
            # Get error information
            log_path = paper_path.with_suffix('.log')
            error_msg = ""
            
            if log_path.exists():
                try:
                    log_content = log_path.read_text(encoding='utf-8', errors='ignore')
                    log_lines = log_content.split('\n')
                    error_msg = '\n'.join(log_lines[-10:])  # Last 10 lines for review PDF
                except Exception as e:
                    error_msg = f"Could not read log file: {e}"
            
            if result.stderr:
                error_msg += f"\n{result.stderr}"
            
            return False, None, error_msg or "Unknown PDF generation error"
            
    except subprocess.TimeoutExpired:
        return False, None, f"PDF generation timed out after {timeout} seconds"
    except FileNotFoundError:
        return False, None, "pdflatex not found. Please install LaTeX"
    except Exception as e:
        return False, None, f"PDF generation error: {str(e)}"


def _validate_latex_structure(tex_content: str) -> Tuple[bool, List[str]]:
    """Validate basic LaTeX structure and return issues found."""
    issues = []
    
    # Check for document class
    if not re.search(r'\\documentclass', tex_content):
        issues.append("Missing \\documentclass declaration")
    
    # Check for begin/end document
    if not re.search(r'\\begin\{document\}', tex_content):
        issues.append("Missing \\begin{document}")
    
    if not re.search(r'\\end\{document\}', tex_content):
        issues.append("Missing \\end{document}")
    
    # Check for balanced braces (basic check)
    open_braces = tex_content.count('{')
    close_braces = tex_content.count('}')
    if open_braces != close_braces:
        issues.append(f"Unbalanced braces: {open_braces} open, {close_braces} close")
    
    # Check for common problematic patterns
    if re.search(r'\\begin\{[^}]+\}.*?\\end\{[^}]+\}', tex_content, re.DOTALL):
        # Check for unmatched begin/end pairs
        begin_matches = re.findall(r'\\begin\{([^}]+)\}', tex_content)
        end_matches = re.findall(r'\\end\{([^}]+)\}', tex_content)
        
        for env in begin_matches:
            if begin_matches.count(env) != end_matches.count(env):
                issues.append(f"Unmatched \\begin{{}} / \\end{{}} for environment: {env}")
    
    # Check for references without labels
    refs = re.findall(r'\\ref\{([^}]+)\}', tex_content)
    labels = re.findall(r'\\label\{([^}]+)\}', tex_content)
    
    for ref in refs:
        if ref not in labels:
            issues.append(f"Reference to undefined label: {ref}")
    
    return len(issues) == 0, issues


def _sanitize_latex_content(tex_content: str) -> str:
    """Sanitize LaTeX content to prevent common compilation issues."""
    # Remove or fix common problematic patterns
    
    # Fix multiple consecutive empty lines
    tex_content = re.sub(r'\n\s*\n\s*\n+', '\n\n', tex_content)
    
    # Fix spacing around equations
    tex_content = re.sub(r'([^\n])\n\s*\$\$', r'\1\n\n$$', tex_content)
    tex_content = re.sub(r'\$\$\s*\n([^\n])', r'$$\n\n\1', tex_content)
    
    # Ensure proper spacing around sections
    tex_content = re.sub(r'([^\n])\n\s*(\\(?:sub)*section)', r'\1\n\n\2', tex_content)
    
    # Fix common character issues
    tex_content = tex_content.replace('"', '``')  # Replace straight quotes
    tex_content = tex_content.replace('"', "''")  # Replace straight quotes
    tex_content = tex_content.replace('…', '\\ldots')  # Replace ellipsis
    
    return tex_content
