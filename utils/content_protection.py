"""
Content Protection Module
Provides multiple layers of protection against accidental content loss during paper revisions.
"""

import re
import difflib
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime
import shutil
import json

def timeout_input(prompt: str, timeout: int = 30, default: str = "") -> str:
    """
    Get user input with a timeout. If no input is received within the timeout,
    return the default value.
    
    Args:
        prompt: The prompt to display to the user
        timeout: Timeout in seconds (default: 30)
        default: Default value to return if timeout occurs
        
    Returns:
        User input or default value if timeout
    """
    # On Windows, we need a different approach
    if os.name == 'nt':  # Windows
        import msvcrt
        
        print(f"{prompt}", end="", flush=True)
        if default:
            print(f" [default: {default}]", end="", flush=True)
        print(" ", end="", flush=True)
        
        start_time = time.time()
        input_chars = []
        
        while time.time() - start_time < timeout:
            if msvcrt.kbhit():
                char = msvcrt.getch()
                if char == b'\r':  # Enter key
                    print()  # New line
                    return ''.join(input_chars)
                elif char == b'\x08':  # Backspace
                    if input_chars:
                        input_chars.pop()
                        print('\b \b', end='', flush=True)
                else:
                    char_str = char.decode('utf-8', errors='ignore')
                    if char_str.isprintable():
                        input_chars.append(char_str)
                        print(char_str, end='', flush=True)
            time.sleep(0.1)
        
        print(f"\nTimeout reached. Using default: {default}")
        return default
    
    else:  # Unix/Linux/Mac
        import select
        print(f"{prompt}", end="", flush=True)
        if default:
            print(f" [default: {default}]", end="", flush=True)
        print(" ", end="", flush=True)
        
        ready, _, _ = select.select([sys.stdin], [], [], timeout)
        if ready:
            return sys.stdin.readline().strip()
        else:
            print(f"\nTimeout reached. Using default: {default}")
            return default


@dataclass
class ContentMetrics:
    """Metrics for tracking paper content."""
    word_count: int
    char_count: int
    section_count: int
    subsection_count: int
    figure_count: int
    table_count: int
    equation_count: int
    citation_count: int
    paragraph_count: int
    page_estimate: float
    
    def similarity_score(self, other: 'ContentMetrics') -> float:
        """Calculate similarity between two sets of metrics."""
        # Weight different metrics by importance
        weights = {
            'word_count': 0.3,
            'section_count': 0.2,
            'figure_count': 0.15,
            'table_count': 0.15,
            'equation_count': 0.1,
            'citation_count': 0.1
        }
        
        total_similarity = 0.0
        for metric, weight in weights.items():
            old_val = getattr(self, metric)
            new_val = getattr(other, metric)
            
            if old_val == 0 and new_val == 0:
                similarity = 1.0
            elif old_val == 0 or new_val == 0:
                similarity = 0.0
            else:
                similarity = min(old_val, new_val) / max(old_val, new_val)
            
            total_similarity += similarity * weight
        
        return total_similarity


@dataclass
class ContentChangeAnalysis:
    """Analysis of changes between two versions."""
    old_metrics: ContentMetrics
    new_metrics: ContentMetrics
    similarity_score: float
    word_count_change: int
    word_count_change_percent: float
    sections_removed: List[str]
    sections_added: List[str]
    major_deletions: List[str]
    is_safe: bool
    warnings: List[str]
    requires_approval: bool


class ContentProtector:
    """Main content protection system."""
    
    def __init__(self, project_dir: Path):
        self.project_dir = project_dir
        self.backup_dir = project_dir / "backups"
        self.backup_dir.mkdir(exist_ok=True)
        
    def extract_metrics(self, latex_content: str) -> ContentMetrics:
        """Extract comprehensive metrics from LaTeX content."""
        # Basic counts
        word_count = len(latex_content.split())
        char_count = len(latex_content)
        
        # Structural elements
        section_count = len(re.findall(r'\\section\{[^}]*\}', latex_content))
        subsection_count = len(re.findall(r'\\subsection\{[^}]*\}', latex_content))
        figure_count = len(re.findall(r'\\begin\{figure\}', latex_content))
        table_count = len(re.findall(r'\\begin\{table\}', latex_content))
        equation_count = len(re.findall(r'\\begin\{equation\}', latex_content)) + len(re.findall(r'\$\$', latex_content)) // 2
        citation_count = len(re.findall(r'\\cite\{[^}]*\}', latex_content))
        
        # Paragraph count (approximate)
        paragraph_count = len([p for p in latex_content.split('\n\n') if p.strip() and not p.strip().startswith('%')])
        
        # Page estimate (rough: 250 words per page)
        page_estimate = word_count / 250.0
        
        return ContentMetrics(
            word_count=word_count,
            char_count=char_count,
            section_count=section_count,
            subsection_count=subsection_count,
            figure_count=figure_count,
            table_count=table_count,
            equation_count=equation_count,
            citation_count=citation_count,
            paragraph_count=paragraph_count,
            page_estimate=page_estimate
        )
    
    def extract_sections(self, latex_content: str) -> List[str]:
        """Extract section titles from LaTeX content."""
        sections = []
        for match in re.finditer(r'\\(?:sub)?section\{([^}]*)\}', latex_content):
            sections.append(match.group(1))
        return sections
    
    def create_backup(self, paper_path: Path, backup_name: Optional[str] = None) -> Path:
        """Create a timestamped backup of the paper."""
        if backup_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_name = f"paper_backup_{timestamp}.tex"
        
        backup_path = self.backup_dir / backup_name
        shutil.copy2(paper_path, backup_path)
        
        # Also save metrics
        content = paper_path.read_text(encoding='utf-8', errors='ignore')
        metrics = self.extract_metrics(content)
        metrics_path = self.backup_dir / f"{backup_name}.metrics.json"
        
        with open(metrics_path, 'w') as f:
            json.dump({
                'timestamp': datetime.now().isoformat(),
                'file': str(paper_path),
                'backup': str(backup_path),
                'metrics': {
                    'word_count': metrics.word_count,
                    'char_count': metrics.char_count,
                    'section_count': metrics.section_count,
                    'subsection_count': metrics.subsection_count,
                    'figure_count': metrics.figure_count,
                    'table_count': metrics.table_count,
                    'equation_count': metrics.equation_count,
                    'citation_count': metrics.citation_count,
                    'paragraph_count': metrics.paragraph_count,
                    'page_estimate': metrics.page_estimate
                }
            }, f, indent=2)
        
        print(f"✓ Backup created: {backup_path.name}")
        return backup_path
    
    def analyze_changes(self, old_content: str, new_content: str) -> ContentChangeAnalysis:
        """Analyze changes between old and new content."""
        old_metrics = self.extract_metrics(old_content)
        new_metrics = self.extract_metrics(new_content)
        
        # Calculate changes
        word_change = new_metrics.word_count - old_metrics.word_count
        word_change_percent = (word_change / old_metrics.word_count * 100) if old_metrics.word_count > 0 else 0
        
        # Section analysis
        old_sections = set(self.extract_sections(old_content))
        new_sections = set(self.extract_sections(new_content))
        sections_removed = list(old_sections - new_sections)
        sections_added = list(new_sections - old_sections)
        
        # Content similarity
        similarity = old_metrics.similarity_score(new_metrics)
        
        # Find major deletions (large blocks of text removed)
        major_deletions = []
        diff = difflib.unified_diff(
            old_content.splitlines(keepends=True),
            new_content.splitlines(keepends=True),
            n=0
        )
        
        current_deletion = []
        for line in diff:
            if line.startswith('-') and not line.startswith('---'):
                current_deletion.append(line[1:])
            else:
                if current_deletion and len(' '.join(current_deletion).split()) > 50:
                    major_deletions.append(' '.join(current_deletion)[:200] + "...")
                current_deletion = []
        
        # Safety analysis
        warnings = []
        requires_approval = False
        
        # Check for dangerous changes
        if word_change_percent < -20:
            warnings.append(f"Large content reduction: {word_change_percent:.1f}% ({abs(word_change)} words)")
            requires_approval = True
        
        if sections_removed:
            warnings.append(f"Sections removed: {', '.join(sections_removed)}")
            requires_approval = True
        
        if similarity < 0.7:
            warnings.append(f"Low content similarity: {similarity:.2f}")
            requires_approval = True
        
        if new_metrics.page_estimate < old_metrics.page_estimate * 0.8:
            warnings.append(f"Significant page reduction: {old_metrics.page_estimate:.1f} → {new_metrics.page_estimate:.1f} pages")
            requires_approval = True
        
        if major_deletions:
            warnings.append(f"Major text blocks deleted: {len(major_deletions)} blocks")
            requires_approval = True
        
        is_safe = not requires_approval
        
        return ContentChangeAnalysis(
            old_metrics=old_metrics,
            new_metrics=new_metrics,
            similarity_score=similarity,
            word_count_change=word_change,
            word_count_change_percent=word_change_percent,
            sections_removed=sections_removed,
            sections_added=sections_added,
            major_deletions=major_deletions,
            is_safe=is_safe,
            warnings=warnings,
            requires_approval=requires_approval
        )
    
    def validate_revision(self, old_content: str, new_content: str, auto_approve: bool = False) -> Tuple[bool, ContentChangeAnalysis]:
        """
        Validate a revision and determine if it should be applied.
        
        Returns:
            (approved, analysis)
        """
        analysis = self.analyze_changes(old_content, new_content)
        
        if analysis.is_safe:
            print("✓ Revision passed safety checks")
            return True, analysis
        
        # Print warnings
        print("⚠ Content protection warnings detected:")
        for i, warning in enumerate(analysis.warnings, 1):
            print(f"   {i}. {warning}")
        
        if analysis.major_deletions:
            print("\n📋 Major deletions detected:")
            for i, deletion in enumerate(analysis.major_deletions[:3], 1):
                print(f"   {i}. {deletion}")
        
        print(f"\n📊 Content metrics comparison:")
        print(f"   Words: {analysis.old_metrics.word_count:,} → {analysis.new_metrics.word_count:,} ({analysis.word_count_change_percent:+.1f}%)")
        print(f"   Pages: {analysis.old_metrics.page_estimate:.1f} → {analysis.new_metrics.page_estimate:.1f}")
        print(f"   Sections: {analysis.old_metrics.section_count} → {analysis.new_metrics.section_count}")
        print(f"   Similarity: {analysis.similarity_score:.2f}")
        
        if auto_approve:
            print("⚠ Auto-approval mode - applying changes despite warnings")
            return True, analysis
        
        # Require user approval
        while True:
            choice = timeout_input("Approve this revision? [y]es/[n]o/[s]how diff:", timeout=30, default="y").lower().strip()
            if choice in ['y', 'yes']:
                return True, analysis
            elif choice in ['n', 'no']:
                return False, analysis
            elif choice in ['s', 'show']:
                self._show_content_diff(old_content, new_content)
            else:
                print("Please enter 'y', 'n', or 's'")
    
    def _show_content_diff(self, old_content: str, new_content: str, max_lines: int = 50):
        """Show a condensed diff of the changes."""
        diff = list(difflib.unified_diff(
            old_content.splitlines(keepends=True),
            new_content.splitlines(keepends=True),
            fromfile='original',
            tofile='revised',
            n=3
        ))
        
        print(f"\n📝 Content diff (showing first {max_lines} lines):")
        for i, line in enumerate(diff[:max_lines]):
            print(line.rstrip())
        
        if len(diff) > max_lines:
            print(f"... ({len(diff) - max_lines} more lines)")
        print()
