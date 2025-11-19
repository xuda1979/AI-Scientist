"""
Paper Protection System - Prevents Catastrophic Content Loss
=============================================================
Multi-layer validation to catch truncation before it destroys papers.
"""

import sys
from pathlib import Path
from typing import Tuple, Optional, List, Dict
import re
import json
from datetime import datetime

class PaperProtectionSystem:
    """Comprehensive protection against paper content loss."""
    
    # Critical thresholds
    MIN_LINES_THRESHOLD = 1000  # Paper should have at least 1000 lines
    MIN_SIZE_RATIO = 0.85  # New version must be >= 85% of original
    MAX_SIZE_RATIO = 1.50  # New version must be <= 150% of original
    
    # Danger patterns that indicate truncation
    TRUNCATION_INDICATORS = [
        r"% \(retained; unchanged\)",
        r"% \(content continues\)",
        r"\.\.\.",
        r"\[truncated\]",
        r"\[content omitted\]",
        r"# Output truncated",
        r"% Content cut",
        r"% Abbreviated",
        r"% Omitted for brevity",
    ]
    
    # Critical sections that must exist
    REQUIRED_SECTIONS = [
        r"\\begin{document}",
        r"\\end{document}",
        r"\\begin{abstract}",
        r"\\end{abstract}",
        r"\\section{Introduction",
        r"\\section\*{Acknowledgements}",
    ]
    
    @staticmethod
    def validate_paper(
        paper_path: Path,
        original_path: Optional[Path] = None
    ) -> Tuple[bool, List[str]]:
        """
        Comprehensive validation of paper content.
        
        Args:
            paper_path: Path to paper to validate
            original_path: Path to original for comparison (optional)
            
        Returns:
            (is_valid, list_of_issues)
        """
        issues = []
        
        if not paper_path.exists():
            return False, [f"Paper file does not exist: {paper_path}"]
        
        # Read content
        try:
            content = paper_path.read_text(encoding='utf-8')
        except Exception as e:
            return False, [f"Failed to read paper: {e}"]
        
        lines = content.split('\n')
        line_count = len(lines)
        size_bytes = len(content.encode('utf-8'))
        
        # Check 1: Minimum line count
        if line_count < PaperProtectionSystem.MIN_LINES_THRESHOLD:
            issues.append(
                f"❌ CRITICAL: Paper too short! {line_count} lines < "
                f"{PaperProtectionSystem.MIN_LINES_THRESHOLD} minimum"
            )
        
        # Check 2: Required sections
        for required_section in PaperProtectionSystem.REQUIRED_SECTIONS:
            if not re.search(required_section, content):
                issues.append(
                    f"❌ CRITICAL: Missing required section: {required_section}"
                )
        
        # Check 3: Truncation indicators
        for pattern in PaperProtectionSystem.TRUNCATION_INDICATORS:
            matches = re.findall(pattern, content, re.IGNORECASE)
            if matches:
                issues.append(
                    f"⚠️  WARNING: Truncation indicator found: '{pattern}' "
                    f"({len(matches)} occurrences)"
                )
        
        # Check 4: Balanced environments
        env_patterns = [
            (r"\\begin\{abstract\}", r"\\end\{abstract\}"),
            (r"\\begin\{document\}", r"\\end\{document\}"),
            (r"\\begin\{equation\}", r"\\end\{equation\}"),
            (r"\\begin\{theorem\}", r"\\end\{theorem\}"),
            (r"\\begin\{algorithm\}", r"\\end\{algorithm\}"),
        ]
        
        for begin_pat, end_pat in env_patterns:
            begin_count = len(re.findall(begin_pat, content))
            end_count = len(re.findall(end_pat, content))
            if begin_count != end_count:
                env_name = begin_pat.replace(r"\\begin\{", "").replace(r"\}", "")
                issues.append(
                    f"❌ CRITICAL: Unbalanced {env_name}: "
                    f"{begin_count} begin, {end_count} end"
                )
        
        # Check 5: Size comparison with original
        if original_path and original_path.exists():
            original_content = original_path.read_text(encoding='utf-8')
            original_size = len(original_content.encode('utf-8'))
            original_lines = len(original_content.split('\n'))
            
            size_ratio = size_bytes / original_size
            line_ratio = line_count / original_lines
            
            if size_ratio < PaperProtectionSystem.MIN_SIZE_RATIO:
                issues.append(
                    f"❌ CRITICAL: Paper size drastically reduced! "
                    f"{size_bytes:,} bytes ({size_ratio:.1%}) vs original "
                    f"{original_size:,} bytes"
                )
            
            if line_ratio < PaperProtectionSystem.MIN_SIZE_RATIO:
                issues.append(
                    f"❌ CRITICAL: Line count drastically reduced! "
                    f"{line_count:,} lines ({line_ratio:.1%}) vs original "
                    f"{original_lines:,} lines"
                )
            
            if size_ratio > PaperProtectionSystem.MAX_SIZE_RATIO:
                issues.append(
                    f"⚠️  WARNING: Paper size unexpectedly increased! "
                    f"{size_bytes:,} bytes ({size_ratio:.1%}) vs original "
                    f"{original_size:,} bytes"
                )
        
        # Check 6: Bibliography present
        has_bib = (
            "\\bibliography{" in content or 
            "\\begin{thebibliography}" in content
        )
        has_citations = bool(re.search(r"\\cite[pt]?\{", content))
        
        if has_citations and not has_bib:
            issues.append(
                "⚠️  WARNING: Citations present but no bibliography"
            )
        
        is_valid = all("❌" not in issue for issue in issues)
        return is_valid, issues
    
    @staticmethod
    def create_emergency_backup(paper_path: Path) -> Path:
        """Create timestamped emergency backup."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_dir = paper_path.parent / "backups"
        backup_dir.mkdir(exist_ok=True)
        
        backup_path = backup_dir / f"{paper_path.stem}_emergency_{timestamp}{paper_path.suffix}"
        
        import shutil
        shutil.copy2(paper_path, backup_path)
        
        # Save metrics
        content = paper_path.read_text(encoding='utf-8')
        metrics = {
            "timestamp": timestamp,
            "lines": len(content.split('\n')),
            "size_bytes": len(content.encode('utf-8')),
            "backup_path": str(backup_path),
        }
        
        metrics_path = backup_path.with_suffix(backup_path.suffix + ".metrics.json")
        metrics_path.write_text(json.dumps(metrics, indent=2))
        
        return backup_path
    
    @staticmethod
    def safe_paper_update(
        paper_path: Path,
        new_content: str,
        force: bool = False
    ) -> Tuple[bool, Optional[str], Optional[Path]]:
        """
        Safely update paper with validation.
        
        Args:
            paper_path: Path to paper
            new_content: New content to write
            force: If True, bypass validation (DANGEROUS!)
            
        Returns:
            (success, error_message, backup_path)
        """
        # Create emergency backup first
        backup_path = None
        if paper_path.exists():
            backup_path = PaperProtectionSystem.create_emergency_backup(paper_path)
            print(f"✅ Emergency backup created: {backup_path}")
        
        # Write to temporary file
        temp_path = paper_path.with_suffix('.temp.tex')
        temp_path.write_text(new_content, encoding='utf-8')
        
        # Validate temporary file
        is_valid, issues = PaperProtectionSystem.validate_paper(
            temp_path,
            original_path=paper_path if paper_path.exists() else None
        )
        
        if not is_valid and not force:
            # Validation failed!
            print("\n" + "="*70)
            print("❌ VALIDATION FAILED - UPDATE BLOCKED!")
            print("="*70)
            for issue in issues:
                print(f"  {issue}")
            print("="*70)
            print(f"\n⚠️  Paper was NOT updated to prevent data loss!")
            print(f"✅ Original safe in: {backup_path}")
            print(f"🔍 Failed content saved in: {temp_path}")
            print("\nTo force update (DANGEROUS!), use force=True")
            print("="*70 + "\n")
            
            temp_path.unlink()  # Clean up temp file
            return False, "\n".join(issues), backup_path
        
        # Validation passed or forced - proceed with update
        if force and not is_valid:
            print("\n⚠️  WARNING: Validation failed but proceeding due to force=True")
            for issue in issues:
                print(f"  {issue}")
        
        # Move temp to actual
        import shutil
        shutil.move(str(temp_path), str(paper_path))
        
        print(f"\n✅ Paper updated successfully")
        if backup_path:
            print(f"✅ Backup available: {backup_path}")
        
        return True, None, backup_path


def test_protection_system():
    """Test the protection system."""
    print("\n" + "="*70)
    print("🧪 Testing Paper Protection System")
    print("="*70 + "\n")
    
    # Test 1: Valid paper
    print("Test 1: Valid paper")
    valid_paper = r"""\documentclass{article}
\begin{document}
\begin{abstract}
This is valid content with over 1000 lines when we add enough content.
""" + "\n".join([f"Line {i}" for i in range(1000)]) + r"""
\end{abstract}
\section{Introduction}
Content here.
\section*{Acknowledgements}
Thanks.
\bibliography{refs}
\end{document}
"""
    
    test_path = Path("test_valid.tex")
    test_path.write_text(valid_paper, encoding='utf-8')
    
    is_valid, issues = PaperProtectionSystem.validate_paper(test_path)
    print(f"  Valid paper: {is_valid}")
    if issues:
        for issue in issues:
            print(f"    {issue}")
    
    # Test 2: Truncated paper
    print("\nTest 2: Truncated paper with danger patterns")
    truncated_paper = r"""\documentclass{article}
\begin{document}
\begin{abstract}
Content here.
\end{abstract}
\section{Introduction}
Content.
% (retained; unchanged)
\end{document}
"""
    
    test_path2 = Path("test_truncated.tex")
    test_path2.write_text(truncated_paper, encoding='utf-8')
    
    is_valid2, issues2 = PaperProtectionSystem.validate_paper(test_path2)
    print(f"  Truncated paper valid: {is_valid2} (expected False)")
    for issue in issues2:
        print(f"    {issue}")
    
    # Cleanup
    test_path.unlink()
    test_path2.unlink()
    
    print("\n" + "="*70)
    print("✅ Protection System Tests Complete")
    print("="*70 + "\n")


# Export
__all__ = [
    'PaperProtectionSystem',
    'test_protection_system',
]


if __name__ == "__main__":
    test_protection_system()
