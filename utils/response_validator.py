"""
AI Response Validation Module
Detects truncated or incomplete responses from AI models.
"""
import re
import logging
from typing import Tuple, List, Optional

logger = logging.getLogger(__name__)


class ResponseTruncationError(Exception):
    """Raised when AI response appears to be truncated."""
    pass


def detect_response_truncation(response_text: str, expected_type: str = "latex") -> Tuple[bool, List[str]]:
    """
    Detect if an AI response appears to be truncated.
    
    Args:
        response_text: The response text from the AI
        expected_type: Type of expected response ("latex", "review", "general")
        
    Returns:
        Tuple of (is_truncated, list_of_issues)
    """
    issues = []
    is_truncated = False
    
    if not response_text or not response_text.strip():
        issues.append("CRITICAL: Empty response received")
        return True, issues
    
    # Check for common truncation indicators
    truncation_patterns = [
        (r'\.\.\.+\s*$', "Response ends with ellipsis (...)"),
        (r'\w+\s*$(?<![.!?])', "Response ends mid-sentence without punctuation"),
        (r'\\emph\{[^}]*$', "Unclosed \\emph{ command at end"),
        (r'\\textbf\{[^}]*$', "Unclosed \\textbf{ command at end"),
        (r'\\section\{[^}]*$', "Unclosed \\section{ command at end"),
        (r'\{[^}]{100,}$', "Long unclosed brace at end (>100 chars)"),
        (r'\\begin\{\w+\}\s*$', "\\begin{environment} at end without \\end"),
    ]
    
    for pattern, description in truncation_patterns:
        if re.search(pattern, response_text.strip(), re.MULTILINE):
            issues.append(f"WARNING: {description} - possible truncation")
            is_truncated = True
    
    # LaTeX-specific checks
    if expected_type == "latex":
        latex_issues, latex_truncated = _check_latex_truncation(response_text)
        issues.extend(latex_issues)
        is_truncated = is_truncated or latex_truncated
    
    # Check for incomplete sentences
    last_100_chars = response_text[-100:].strip()
    if last_100_chars and not re.search(r'[.!?}]\s*$', last_100_chars):
        issues.append("WARNING: Response ends without proper sentence terminator")
        is_truncated = True
    
    # Check response length (suspiciously short responses)
    if len(response_text) < 200:
        issues.append(f"WARNING: Response is very short ({len(response_text)} chars)")
    
    return is_truncated, issues


def _check_latex_truncation(tex_content: str) -> Tuple[List[str], bool]:
    """Check for LaTeX-specific truncation indicators."""
    issues = []
    is_truncated = False
    
    # Check for missing \end{document}
    if '\\begin{document}' in tex_content and '\\end{document}' not in tex_content:
        issues.append("CRITICAL: LaTeX has \\begin{document} but missing \\end{document}")
        is_truncated = True
    
    # Check for incomplete sections (section header with no content)
    incomplete_sections = re.findall(
        r'\\section\*?\{([^}]+)\}\s*(?=\\section|\\end\{document\}|$)',
        tex_content
    )
    if incomplete_sections:
        issues.append(
            f"WARNING: Found {len(incomplete_sections)} sections with no/minimal content: "
            f"{', '.join(incomplete_sections[:3])}"
        )
    
    # Check for unbalanced environments
    begins = re.findall(r'\\begin\{(\w+)\}', tex_content)
    ends = re.findall(r'\\end\{(\w+)\}', tex_content)
    
    from collections import Counter
    begin_counts = Counter(begins)
    end_counts = Counter(ends)
    
    unbalanced = []
    for env in set(begins):
        if begin_counts[env] != end_counts.get(env, 0):
            unbalanced.append(env)
            is_truncated = True
    
    if unbalanced:
        issues.append(
            f"CRITICAL: Unbalanced LaTeX environments: {', '.join(unbalanced)} - "
            f"indicates truncation"
        )
    
    return issues, is_truncated


def check_finish_reason(api_response: any) -> Tuple[bool, Optional[str]]:
    """
    Check the finish_reason from API response to detect truncation.
    
    Args:
        api_response: The raw API response object
        
    Returns:
        Tuple of (was_truncated, reason_description)
    """
    try:
        # OpenAI API structure
        if hasattr(api_response, 'choices') and len(api_response.choices) > 0:
            finish_reason = api_response.choices[0].finish_reason
            
            if finish_reason == 'length':
                return True, "Response hit maximum token limit - content truncated"
            elif finish_reason == 'stop':
                return False, "Response completed normally"
            elif finish_reason == 'content_filter':
                return True, "Response stopped by content filter"
            elif finish_reason:
                return False, f"Response finished: {finish_reason}"
        
        # Google/Gemini API structure
        if hasattr(api_response, 'candidates') and len(api_response.candidates) > 0:
            finish_reason = api_response.candidates[0].finish_reason
            # Map Gemini finish reasons
            reason_map = {
                1: (False, "Completed normally (STOP)"),
                2: (True, "Hit maximum token limit (MAX_TOKENS)"),
                3: (True, "Safety filter triggered (SAFETY)"),
                4: (True, "Recitation detected (RECITATION)"),
                5: (True, "Other reason (OTHER)"),
            }
            if finish_reason in reason_map:
                return reason_map[finish_reason]
        
    except Exception as e:
        logger.warning(f"Could not extract finish_reason: {e}")
    
    return False, None


def validate_paper_structure(tex_content: str) -> Tuple[bool, List[str]]:
    """
    Validate that a paper has all required sections.
    
    Returns:
        Tuple of (is_complete, list_of_missing_sections)
    """
    missing = []
    
    # Extract document body
    doc_match = re.search(
        r'\\begin\{document\}(.*?)\\end\{document\}',
        tex_content,
        re.DOTALL
    )
    
    if not doc_match:
        return False, ["No document body found"]
    
    body = doc_match.group(1)
    
    # Required sections
    required = {
        'Abstract': r'\\begin\{abstract\}',
        'Introduction': r'\\section\*?\{[^}]*[Ii]ntroduction[^}]*\}',
        'Methods/Approach': r'\\section\*?\{[^}]*([Mm]ethod|[Aa]pproach|[Dd]esign)[^}]*\}',
        'Experiments/Results': r'\\section\*?\{[^}]*([Ee]xperiment|[Rr]esult|[Ee]valuation)[^}]*\}',
        'Conclusion': r'\\section\*?\{[^}]*([Cc]onclusion|[Dd]iscussion)[^}]*\}',
    }
    
    for section_name, pattern in required.items():
        if not re.search(pattern, body):
            missing.append(section_name)
    
    is_complete = len(missing) == 0
    
    if missing:
        logger.warning(f"Paper structure incomplete. Missing: {', '.join(missing)}")
    
    return is_complete, missing


def estimate_paper_completeness(tex_content: str) -> float:
    """
    Estimate what percentage of the paper appears to be complete.
    
    Returns:
        Float between 0.0 and 1.0 indicating completeness
    """
    score = 0.0
    
    # Check for \end{document} (20%)
    if re.search(r'\\end\{document\}', tex_content):
        score += 0.2
    
    # Check for required sections (60%)
    _, missing = validate_paper_structure(tex_content)
    sections_present = 5 - len(missing)
    score += (sections_present / 5) * 0.6
    
    # Check content length (20%)
    doc_match = re.search(
        r'\\begin\{document\}(.*?)\\end\{document\}',
        tex_content,
        re.DOTALL
    )
    if doc_match:
        body = doc_match.group(1)
        # Remove LaTeX commands and count actual content
        content = re.sub(r'\\[a-zA-Z]+(\{[^}]*\}|\[[^\]]*\])?', '', body)
        content = re.sub(r'\s+', ' ', content).strip()
        
        # A complete paper should have ~10,000+ chars of content
        # Give partial credit for shorter papers
        content_score = min(len(content) / 10000, 1.0)
        score += content_score * 0.2
    
    return min(score, 1.0)
