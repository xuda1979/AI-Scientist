"""
LaTeX Document Validation Module
Validates LaTeX documents for completeness and common issues before compilation.
"""
import re
from pathlib import Path
from typing import List, Tuple, Dict, Optional
import logging

logger = logging.getLogger(__name__)


class LaTeXValidationError(Exception):
    """Raised when LaTeX document has critical validation errors."""
    pass


def validate_latex_document(tex_content: str, filepath: Optional[Path] = None) -> Tuple[bool, List[str]]:
    """
    Validate LaTeX document for common issues that prevent PDF generation.
    
    Args:
        tex_content: The LaTeX document content as string
        filepath: Optional path to the .tex file for better error messages
        
    Returns:
        Tuple of (is_valid, list_of_issues)
        is_valid is True only if no critical issues found
    """
    issues = []
    
    # 1. Check for document structure completeness
    if not tex_content.strip():
        issues.append("CRITICAL: Empty document")
        return False, issues
    
    # 2. Check for \documentclass
    if not re.search(r'\\documentclass', tex_content):
        issues.append("CRITICAL: Missing \\documentclass command")
    
    # 3. Check for \begin{document}
    if not re.search(r'\\begin\{document\}', tex_content):
        issues.append("CRITICAL: Missing \\begin{document}")
    
    # 4. Check for \end{document} - MOST CRITICAL
    if not re.search(r'\\end\{document\}', tex_content):
        issues.append("CRITICAL: Missing \\end{document} - Document appears truncated!")
        # This is a fatal error
        return False, issues
    
    # 5. Check that \end{document} is near the end (not in middle of file)
    end_doc_match = re.search(r'\\end\{document\}', tex_content)
    if end_doc_match:
        pos = end_doc_match.end()
        remaining = tex_content[pos:].strip()
        # Allow up to 200 chars after \end{document} (for comments)
        if len(remaining) > 200:
            issues.append(f"WARNING: {len(remaining)} characters after \\end{{document}} - possible truncation")
    
    # 6. Check for balanced environments
    env_issues = check_balanced_environments(tex_content)
    if env_issues:
        issues.extend(env_issues)
    
    # 7. Check for balanced braces
    brace_issues = check_balanced_braces(tex_content)
    if brace_issues:
        issues.extend(brace_issues)
    
    # 8. Check for problematic characters in labels/refs
    label_issues = check_problematic_labels(tex_content)
    if label_issues:
        issues.extend(label_issues)
    
    # 9. Check for incomplete/broken tables
    table_issues = check_table_syntax(tex_content)
    if table_issues:
        issues.extend(table_issues)
    
    # 10. Check for broken plot syntax
    plot_issues = check_plot_syntax(tex_content)
    if plot_issues:
        issues.extend(plot_issues)
    
    # 11. Check for missing sections (CRITICAL for detecting truncation)
    section_issues = check_required_sections(tex_content)
    if section_issues:
        issues.extend(section_issues)
    
    # 12. Check document length (suspiciously short documents are likely truncated)
    length_issues = check_document_length(tex_content)
    if length_issues:
        issues.extend(length_issues)
    
    # 13. Check for bibliography/references (CRITICAL - papers MUST have references)
    bib_issues = check_bibliography(tex_content)
    if bib_issues:
        issues.extend(bib_issues)
    
    # Determine if document is valid
    critical_count = sum(1 for issue in issues if issue.startswith("CRITICAL"))
    is_valid = critical_count == 0
    
    if not is_valid:
        logger.error(f"LaTeX validation failed with {critical_count} critical issues")
        for issue in issues:
            if issue.startswith("CRITICAL"):
                logger.error(f"  - {issue}")
    
    return is_valid, issues


def check_balanced_environments(tex_content: str) -> List[str]:
    """Check for balanced \\begin{...} and \\end{...} pairs."""
    issues = []
    
    # Find all \begin{env} and \end{env} commands
    begins = re.findall(r'\\begin\{(\w+)\}', tex_content)
    ends = re.findall(r'\\end\{(\w+)\}', tex_content)
    
    # Count occurrences
    from collections import Counter
    begin_counts = Counter(begins)
    end_counts = Counter(ends)
    
    # Check for mismatches
    all_envs = set(begins) | set(ends)
    for env in all_envs:
        begin_c = begin_counts.get(env, 0)
        end_c = end_counts.get(env, 0)
        if begin_c != end_c:
            issues.append(f"CRITICAL: Unbalanced environment '{env}': {begin_c} begin(s), {end_c} end(s)")
    
    return issues


def check_balanced_braces(tex_content: str) -> List[str]:
    """Check for balanced curly braces (basic check, may have false positives)."""
    issues = []
    
    # Remove comments first
    lines = tex_content.split('\n')
    no_comments = '\n'.join(
        line.split('%')[0] if '%' in line else line
        for line in lines
    )
    
    # Count braces
    open_braces = no_comments.count('{')
    close_braces = no_comments.count('}')
    
    if open_braces != close_braces:
        diff = abs(open_braces - close_braces)
        issues.append(f"WARNING: Unbalanced braces: {open_braces} opening, {close_braces} closing (diff: {diff})")
    
    return issues


def check_problematic_labels(tex_content: str) -> List[str]:
    """Check for labels with problematic characters."""
    issues = []
    
    # Find all labels and refs
    labels = re.findall(r'\\label\{([^}]+)\}', tex_content)
    refs = re.findall(r'\\ref\{([^}]+)\}', tex_content)
    cites = re.findall(r'\\cite\{([^}]+)\}', tex_content)
    
    # Check for problematic characters (spaces, special chars)
    problematic = re.compile(r'[^\w\-:.]')
    
    for label in labels:
        if problematic.search(label):
            issues.append(f"WARNING: Label contains problematic characters: '{label}'")
    
    # Check for undefined refs (refs that don't have matching labels)
    label_set = set(labels)
    for ref in refs:
        if ref not in label_set and not ref.startswith('eq:') and not ref.startswith('fig:'):
            # Skip equation/figure refs as they might be defined differently
            pass  # Soft warning, not critical
    
    return issues


def check_table_syntax(tex_content: str) -> List[str]:
    """Check for problematic table syntax."""
    issues = []
    
    # Check for pgfplotstable inside tabular environment (common error)
    tabular_sections = re.finditer(
        r'\\begin\{tabular\}.*?\\end\{tabular\}',
        tex_content,
        re.DOTALL
    )
    
    for match in tabular_sections:
        table_content = match.group(0)
        if '\\pgfplotstable' in table_content:
            issues.append("CRITICAL: pgfplotstable command found inside tabular environment - this will fail!")
        if '\\pgfplotstableread' in table_content:
            issues.append("CRITICAL: pgfplotstableread found inside tabular - move outside environment")
    
    return issues


def check_plot_syntax(tex_content: str) -> List[str]:
    """Check for problematic plot syntax."""
    issues = []
    
    # Check for coordinates with parentheses in symbolic coords (common pgfplots error)
    axis_blocks = re.finditer(
        r'\\begin\{axis\}.*?\\end\{axis\}',
        tex_content,
        re.DOTALL
    )
    
    for match in axis_blocks:
        axis_content = match.group(0)
        
        # Check for symbolic coords with special characters
        symbolic_match = re.search(
            r'symbolic\s+[xy]\s+coords\s*=\s*\{([^}]+)\}',
            axis_content
        )
        if symbolic_match:
            coords = symbolic_match.group(1)
            # Check each coordinate
            coord_list = [c.strip() for c in coords.split(',')]
            for coord in coord_list:
                if '(' in coord or ')' in coord:
                    issues.append(f"WARNING: Symbolic coordinate contains parentheses: '{coord}' - may cause parsing errors")
                if '/' in coord and ' ' in coord:
                    issues.append(f"WARNING: Symbolic coordinate with space and slash: '{coord}' - may cause issues")
    
    return issues


def check_required_sections(tex_content: str) -> List[str]:
    """
    Check for presence of required academic paper sections.
    Missing sections often indicate severe truncation.
    """
    issues = []
    
    # Extract content between \begin{document} and \end{document}
    doc_match = re.search(
        r'\\begin\{document\}(.*?)\\end\{document\}',
        tex_content,
        re.DOTALL
    )
    
    if not doc_match:
        return []  # Already caught by other checks
    
    document_body = doc_match.group(1)
    
    # Define required sections for a research paper
    required_sections = {
        'introduction': r'\\section\*?\{.*?[Ii]ntroduction.*?\}',
        'methods': r'\\section\*?\{.*?([Mm]ethod|[Mm]ethodology|[Aa]pproach|[Dd]esign).*?\}',
        'experiments': r'\\section\*?\{.*?([Ee]xperiment|[Rr]esult|[Ee]valuation).*?\}',
        'conclusion': r'\\section\*?\{.*?([Cc]onclusion|[Dd]iscussion|[Ff]uture [Ww]ork).*?\}',
    }
    
    missing_sections = []
    for section_name, pattern in required_sections.items():
        if not re.search(pattern, document_body):
            missing_sections.append(section_name)
    
    # If more than 2 major sections are missing, this is CRITICAL (likely truncation)
    if len(missing_sections) >= 2:
        issues.append(
            f"CRITICAL: Missing {len(missing_sections)} major sections: {', '.join(missing_sections)}. "
            f"Document appears severely truncated!"
        )
    elif missing_sections:
        issues.append(
            f"WARNING: Missing sections: {', '.join(missing_sections)}. "
            f"Document may be incomplete."
        )
    
    return issues


def check_document_length(tex_content: str) -> List[str]:
    """
    Check if document length is suspiciously short.
    Very short documents often indicate truncation.
    """
    issues = []
    
    # Extract content between \begin{document} and \end{document}
    doc_match = re.search(
        r'\\begin\{document\}(.*?)\\end\{document\}',
        tex_content,
        re.DOTALL
    )
    
    if not doc_match:
        return []
    
    document_body = doc_match.group(1)
    
    # Count actual content (excluding commands, whitespace, comments)
    # Remove LaTeX commands
    content_only = re.sub(r'\\[a-zA-Z]+(\{[^}]*\}|\[[^\]]*\])?', '', document_body)
    # Remove comments
    content_only = re.sub(r'%.*$', '', content_only, flags=re.MULTILINE)
    # Remove excessive whitespace
    content_only = re.sub(r'\s+', ' ', content_only).strip()
    
    content_length = len(content_only)
    
    # Check for suspiciously short content
    # A typical academic paper should have at least 10,000 characters of content
    if content_length < 2000:
        issues.append(
            f"CRITICAL: Document body only has {content_length} characters. "
            f"This is extremely short - document is likely severely truncated!"
        )
    elif content_length < 5000:
        issues.append(
            f"WARNING: Document body only has {content_length} characters. "
            f"This seems short for an academic paper - possible truncation."
        )
    elif content_length < 10000:
        issues.append(
            f"INFO: Document body has {content_length} characters. "
            f"This is relatively short for a full paper."
        )
    
    # Count sections
    sections = re.findall(r'\\section\*?\{', document_body)
    if len(sections) < 3:
        issues.append(
            f"WARNING: Only {len(sections)} sections found. "
            f"A complete paper should have at least 4-5 major sections."
        )
    
    return issues


def check_bibliography(tex_content: str) -> List[str]:
    """
    Check if paper has proper bibliography/references section.
    Academic papers MUST have references - this is CRITICAL.
    """
    issues = []
    
    # Check if document has citations
    citations = re.findall(r'\\cite\{[^}]+\}', tex_content)
    has_citations = len(citations) > 0
    
    # Check for bibliography commands
    has_bibliography_cmd = bool(re.search(r'\\bibliography\{[^}]+\}', tex_content))
    has_bibliographystyle = bool(re.search(r'\\bibliographystyle\{[^}]+\}', tex_content))
    has_thebibliography = bool(re.search(r'\\begin\{thebibliography\}', tex_content))
    
    # A paper should have EITHER \bibliography{} OR \begin{thebibliography}
    has_any_bib = has_bibliography_cmd or has_thebibliography
    
    if has_citations and not has_any_bib:
        issues.append(
            f"CRITICAL: Paper has {len(citations)} citations but NO bibliography section! "
            f"Missing \\bibliography{{refs}} or \\begin{{thebibliography}}. "
            f"This is a SERIOUS validation failure - references are REQUIRED!"
        )
    elif has_citations and not has_bibliographystyle and has_bibliography_cmd:
        issues.append(
            f"WARNING: Paper has \\bibliography{{}} but missing \\bibliographystyle{{}}. "
            f"References may not format correctly."
        )
    elif not has_citations:
        # No citations is also suspicious for an academic paper
        issues.append(
            f"WARNING: Paper has NO citations (\\cite commands). "
            f"Academic papers should cite prior work."
        )
    
    # Check if bibliography commands are BEFORE \end{document}
    if has_bibliography_cmd:
        bib_match = re.search(r'\\bibliography\{[^}]+\}', tex_content)
        end_match = re.search(r'\\end\{document\}', tex_content)
        
        if bib_match and end_match:
            if bib_match.start() > end_match.start():
                issues.append(
                    f"CRITICAL: \\bibliography{{}} command appears AFTER \\end{{document}}! "
                    f"Bibliography will not be included in the paper."
                )
    
    # Count references in refs.bib if bibliography command exists
    if has_bibliography_cmd:
        # Extract bibliography filename
        bib_match = re.search(r'\\bibliography\{([^}]+)\}', tex_content)
        if bib_match:
            bib_filename = bib_match.group(1)
            if not bib_filename.endswith('.bib'):
                bib_filename += '.bib'
            
            # Try to check if references are sufficient
            if has_citations and len(citations) > 0:
                # Just a sanity check - can't validate without file access
                pass
    
    return issues


def auto_fix_common_issues(tex_content: str) -> Tuple[str, List[str]]:
    """
    Automatically fix common LaTeX issues that prevent PDF generation.
    
    Returns:
        Tuple of (fixed_content, list_of_fixes_applied)
    """
    fixes_applied = []
    fixed = tex_content
    
    # 1. Add \end{document} if missing (CRITICAL FIX)
    if not re.search(r'\\end\{document\}', fixed):
        fixed += '\n\n\\end{document}\n'
        fixes_applied.append("Added missing \\end{document}")
        logger.warning("Auto-fix: Added missing \\end{document} tag")
    
    # 1.5. Add missing bibliography commands if paper has citations (CRITICAL FIX)
    citations = re.findall(r'\\cite\{[^}]+\}', fixed)
    has_bibliography_cmd = bool(re.search(r'\\bibliography\{[^}]+\}', fixed))
    has_bibliographystyle = bool(re.search(r'\\bibliographystyle\{[^}]+\}', fixed))
    has_thebibliography = bool(re.search(r'\\begin\{thebibliography\}', fixed))
    
    if citations and not (has_bibliography_cmd or has_thebibliography):
        # Paper has citations but no bibliography - ADD IT!
        end_doc_pos = fixed.rfind('\\end{document}')
        if end_doc_pos > 0:
            bibliography_block = '\n\n% Bibliography\n'
            if not has_bibliographystyle:
                bibliography_block += '\\bibliographystyle{plain}\n'
                fixes_applied.append("Added missing \\bibliographystyle{plain}")
            bibliography_block += '\\bibliography{refs}\n'
            fixes_applied.append("Added missing \\bibliography{refs}")
            
            # Insert before \end{document}
            fixed = fixed[:end_doc_pos] + bibliography_block + fixed[end_doc_pos:]
            logger.warning(f"Auto-fix: Added missing bibliography commands ({len(citations)} citations found)")
    elif citations and has_bibliography_cmd and not has_bibliographystyle:
        # Has bibliography but missing style
        bib_match = re.search(r'\\bibliography\{[^}]+\}', fixed)
        if bib_match:
            insert_pos = bib_match.start()
            fixed = fixed[:insert_pos] + '\\bibliographystyle{plain}\n' + fixed[insert_pos:]
            fixes_applied.append("Added missing \\bibliographystyle{plain}")
            logger.warning("Auto-fix: Added missing \\bibliographystyle")
    
    # 2. Fix truncated sentences before \end{document}
    # Look for incomplete sentences (no period before \end{document})
    end_doc_pos = fixed.rfind('\\end{document}')
    if end_doc_pos > 0:
        before_end = fixed[:end_doc_pos].rstrip()
        if before_end and not before_end.endswith(('.', '!', '?', '}', '\n')):
            # Sentence appears truncated, try to complete it
            # Look backwards for last complete sentence
            last_period = before_end.rfind('.')
            last_brace = before_end.rfind('}')
            last_newline = before_end.rfind('\n')
            
            cutoff = max(last_period, last_brace, last_newline)
            if cutoff > 0 and cutoff > len(before_end) - 200:
                # Cut at last complete structure
                fixed = before_end[:cutoff+1] + '\n\n\\end{document}\n'
                fixes_applied.append(f"Removed {len(before_end) - cutoff - 1} chars of truncated text before \\end{{document}}")
                logger.warning(f"Auto-fix: Removed truncated text before \\end{{document}}")
    
    # 3. Simplify problematic plot coordinates
    # Replace symbolic coords with parentheses to numeric
    def simplify_coords(match):
        axis_block = match.group(0)
        
        # Check for symbolic coords with problematic characters
        if 'symbolic y coords' in axis_block and '(' in axis_block:
            # Replace with numeric ytick/yticklabels
            symbolic_match = re.search(
                r'symbolic\s+y\s+coords\s*=\s*\{([^}]+)\}',
                axis_block
            )
            if symbolic_match:
                coords = symbolic_match.group(1)
                coord_list = [c.strip() for c in coords.split(',')]
                
                # Create numeric labels
                numeric_labels = ','.join(str(i) for i in range(len(coord_list)))
                labels_str = ','.join(coord_list)
                
                # Replace symbolic coords with ytick and yticklabels
                axis_block = re.sub(
                    r'symbolic\s+y\s+coords\s*=\s*\{[^}]+\}',
                    '',
                    axis_block
                )
                axis_block = re.sub(
                    r'ytick\s*=\s*data',
                    f'ytick={{{numeric_labels}}},\n    yticklabels={{{labels_str}}}',
                    axis_block
                )
                
                fixes_applied.append("Simplified problematic plot coordinates")
        
        return axis_block
    
    fixed = re.sub(
        r'\\begin\{axis\}.*?\\end\{axis\}',
        simplify_coords,
        fixed,
        flags=re.DOTALL
    )
    
    return fixed, fixes_applied


def validate_and_fix(tex_file: Path, auto_fix: bool = True) -> Tuple[bool, List[str], Optional[str]]:
    """
    Validate a LaTeX file and optionally auto-fix common issues.
    
    Args:
        tex_file: Path to the .tex file
        auto_fix: Whether to automatically fix common issues
        
    Returns:
        Tuple of (is_valid, issues_found, fixed_content_if_autofix)
    """
    if not tex_file.exists():
        return False, [f"File not found: {tex_file}"], None
    
    content = tex_file.read_text(encoding='utf-8', errors='ignore')
    
    # Validate first
    is_valid, issues = validate_latex_document(content, tex_file)
    
    fixed_content = None
    if auto_fix and not is_valid:
        logger.info(f"Attempting to auto-fix LaTeX issues in {tex_file.name}")
        fixed_content, fixes = auto_fix_common_issues(content)
        
        # Re-validate after fixes
        is_valid_after, issues_after = validate_latex_document(fixed_content, tex_file)
        
        if is_valid_after:
            logger.info(f"Auto-fix successful! Applied: {', '.join(fixes)}")
            issues = [f"AUTO-FIXED: {fix}" for fix in fixes] + issues_after
            return True, issues, fixed_content
        else:
            logger.warning(f"Auto-fix applied but issues remain: {issues_after}")
            issues = [f"AUTO-FIX ATTEMPTED: {fix}" for fix in fixes] + issues_after
            return False, issues, fixed_content
    
    return is_valid, issues, None


if __name__ == "__main__":
    # Test the validator
    import sys
    
    if len(sys.argv) > 1:
        tex_path = Path(sys.argv[1])
        is_valid, issues, fixed = validate_and_fix(tex_path, auto_fix=True)
        
        print(f"\nValidation Results for {tex_path.name}:")
        print(f"Valid: {is_valid}")
        print(f"\nIssues found ({len(issues)}):")
        for issue in issues:
            print(f"  - {issue}")
        
        if fixed:
            print(f"\nAuto-fix was applied. Review the fixes above.")
            save = input("Save fixed version? (y/n): ")
            if save.lower() == 'y':
                backup = tex_path.with_suffix('.tex.bak')
                tex_path.rename(backup)
                tex_path.write_text(fixed, encoding='utf-8')
                print(f"Saved! Backup: {backup}")
    else:
        print("Usage: python latex_validator.py <path_to_tex_file>")
