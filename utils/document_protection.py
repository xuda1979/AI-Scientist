"""
Document Protection with Validation and Safe Writing
====================================================
Prevents content loss through validation, chunking, and atomic writes.
"""

import os
import re
import tempfile
import shutil
from pathlib import Path
from typing import Optional, Tuple, List, Dict
from datetime import datetime

class DocumentValidator:
    """Validates document completeness and correctness."""
    
    @staticmethod
    def validate_latex(content: str, original_content: Optional[str] = None) -> Tuple[bool, List[str]]:
        """
        Validate LaTeX document structure and completeness.
        
        Args:
            content: Document content to validate
            original_content: Original content for comparison (optional)
            
        Returns:
            (is_valid, list_of_issues)
        """
        issues = []
        
        # 1. Check basic structure
        if not content or len(content.strip()) == 0:
            issues.append("Document is empty")
            return False, issues
        
        # 2. Check for document environment
        if "\\begin{document}" not in content:
            issues.append("Missing \\begin{document}")
        if "\\end{document}" not in content:
            issues.append("Missing \\end{document}")
        
        # 3. Check for paired environments
        env_patterns = [
            (r"\\begin{abstract}", r"\\end{abstract}"),
            (r"\\begin{equation}", r"\\end{equation}"),
            (r"\\begin{align}", r"\\end{align}"),
            (r"\\begin{theorem}", r"\\end{theorem}"),
            (r"\\begin{lemma}", r"\\end{lemma}"),
            (r"\\begin{proposition}", r"\\end{proposition}"),
            (r"\\begin{proof}", r"\\end{proof}"),
            (r"\\begin{figure}", r"\\end{figure}"),
            (r"\\begin{table}", r"\\end{table}"),
        ]
        
        for begin_pat, end_pat in env_patterns:
            begin_count = len(re.findall(begin_pat, content))
            end_count = len(re.findall(end_pat, content))
            if begin_count != end_count:
                env_name = begin_pat.replace(r"\\begin{", "").replace("}", "")
                issues.append(f"Mismatched {env_name} environment: {begin_count} begin, {end_count} end")
        
        # 4. Check for balanced braces (simple check)
        open_braces = content.count("{")
        close_braces = content.count("}")
        if open_braces != close_braces:
            issues.append(f"Mismatched braces: {open_braces} open, {close_braces} close")
        
        # 5. Check for truncation indicators
        truncation_indicators = [
            "...",
            "[truncated]",
            "[content continues]",
            "# Output truncated",
        ]
        for indicator in truncation_indicators:
            if indicator in content[-500:]:  # Check last 500 chars
                issues.append(f"Possible truncation detected: '{indicator}' near end")
        
        # 6. Compare with original if provided
        if original_content:
            orig_lines = len(original_content.split('\n'))
            new_lines = len(content.split('\n'))
            
            # Check if new content is significantly shorter
            if new_lines < orig_lines * 0.8:  # More than 20% reduction
                ratio = (new_lines / orig_lines) * 100
                issues.append(f"Content significantly reduced: {orig_lines} → {new_lines} lines ({ratio:.1f}%)")
            
            # Check size ratio
            orig_size = len(original_content)
            new_size = len(content)
            if new_size < orig_size * 0.8:  # More than 20% reduction
                ratio = (new_size / orig_size) * 100
                issues.append(f"Size significantly reduced: {orig_size} → {new_size} bytes ({ratio:.1f}%)")
        
        # 7. Check for bibliography
        has_bib = "\\bibliography" in content or "\\begin{thebibliography}" in content
        has_citations = "\\cite" in content or "\\citep" in content or "\\citet" in content
        if has_citations and not has_bib:
            issues.append("Document has citations but no bibliography")
        
        is_valid = len(issues) == 0
        return is_valid, issues
    
    @staticmethod
    def validate_size_ratio(
        original_path: Path,
        new_content: str,
        min_ratio: float = 0.8
    ) -> Tuple[bool, Optional[str]]:
        """
        Validate that new content isn't suspiciously smaller than original.
        
        Args:
            original_path: Path to original file
            new_content: New content to check
            min_ratio: Minimum acceptable size ratio (default 0.8 = 80%)
            
        Returns:
            (is_valid, error_message)
        """
        if not original_path.exists():
            return True, None  # Can't compare if original doesn't exist
        
        original_size = original_path.stat().st_size
        new_size = len(new_content.encode('utf-8'))
        
        if new_size < original_size * min_ratio:
            ratio = (new_size / original_size) * 100
            error = f"New content too small: {new_size:,} bytes vs original {original_size:,} bytes ({ratio:.1f}%)"
            return False, error
        
        return True, None


class DocumentChunker:
    """Chunks large documents for processing."""
    
    @staticmethod
    def split_latex_document(
        content: str,
        max_lines_per_chunk: int = 300,
        overlap_lines: int = 50
    ) -> List[Dict[str, any]]:
        """
        Split LaTeX document into chunks at logical boundaries.
        
        Args:
            content: Document content
            max_lines_per_chunk: Maximum lines per chunk
            overlap_lines: Number of overlapping lines between chunks
            
        Returns:
            List of chunk dictionaries with 'content', 'start_line', 'end_line'
        """
        lines = content.split('\n')
        total_lines = len(lines)
        
        if total_lines <= max_lines_per_chunk:
            return [{
                'content': content,
                'start_line': 0,
                'end_line': total_lines,
                'chunk_index': 0,
                'total_chunks': 1,
            }]
        
        # Find section boundaries for smart splitting
        section_lines = []
        for i, line in enumerate(lines):
            if re.match(r'\\section|\\subsection|\\subsubsection', line):
                section_lines.append(i)
        
        chunks = []
        chunk_index = 0
        current_line = 0
        
        while current_line < total_lines:
            # Find next chunk boundary
            target_line = min(current_line + max_lines_per_chunk, total_lines)
            
            # Try to split at section boundary
            split_line = target_line
            for sec_line in section_lines:
                if current_line < sec_line <= target_line:
                    split_line = sec_line
                    break
            
            # Extract chunk
            chunk_end = min(split_line, total_lines)
            chunk_content = '\n'.join(lines[current_line:chunk_end])
            
            chunks.append({
                'content': chunk_content,
                'start_line': current_line,
                'end_line': chunk_end,
                'chunk_index': chunk_index,
                'total_chunks': -1,  # Will update after loop
            })
            
            # Move to next chunk with overlap
            # Ensure we always advance forward to prevent infinite loops
            next_line = chunk_end - overlap_lines
            if next_line <= current_line:
                next_line = chunk_end  # No overlap if it would prevent progress
            
            current_line = next_line
            
            # Safety check: if we're at the end, break
            if current_line >= total_lines:
                break
            
            chunk_index += 1
        
        # Update total chunks
        for chunk in chunks:
            chunk['total_chunks'] = len(chunks)
        
        return chunks
    
    @staticmethod
    def merge_chunks(chunks: List[str], overlap_lines: int = 50) -> str:
        """
        Merge overlapping chunks back together.
        
        Args:
            chunks: List of chunk strings
            overlap_lines: Number of overlapping lines
            
        Returns:
            Merged content
        """
        if len(chunks) == 1:
            return chunks[0]
        
        merged = chunks[0]
        
        for i in range(1, len(chunks)):
            chunk_lines = chunks[i].split('\n')
            # Skip overlap lines
            merged += '\n' + '\n'.join(chunk_lines[overlap_lines:])
        
        return merged


class SafeFileWriter:
    """Atomic file writing with validation and backup."""
    
    @staticmethod
    def write_atomic(
        file_path: Path,
        content: str,
        create_backup: bool = True,
        validate_before_save: bool = True
    ) -> Tuple[bool, Optional[str]]:
        """
        Write file atomically with validation and backup.
        
        Args:
            file_path: Path to file
            content: Content to write
            create_backup: Whether to create backup before writing
            validate_before_save: Whether to validate content
            
        Returns:
            (success, error_message)
        """
        file_path = Path(file_path)
        
        # 1. Validate content if requested
        if validate_before_save and file_path.suffix == '.tex':
            original_content = None
            if file_path.exists():
                original_content = file_path.read_text(encoding='utf-8')
            
            is_valid, issues = DocumentValidator.validate_latex(content, original_content)
            if not is_valid:
                error_msg = "Validation failed:\n" + "\n".join(f"  - {issue}" for issue in issues)
                return False, error_msg
        
        # 2. Create backup if file exists
        backup_path = None
        if create_backup and file_path.exists():
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_dir = file_path.parent / "backups"
            backup_dir.mkdir(exist_ok=True)
            backup_path = backup_dir / f"{file_path.stem}_backup_{timestamp}{file_path.suffix}"
            
            try:
                shutil.copy2(file_path, backup_path)
            except Exception as e:
                return False, f"Backup creation failed: {e}"
        
        # 3. Write to temporary file first
        temp_fd, temp_path = tempfile.mkstemp(
            suffix=file_path.suffix,
            prefix=f"{file_path.stem}_temp_",
            dir=file_path.parent,
            text=True
        )
        
        try:
            # Write content
            with os.fdopen(temp_fd, 'w', encoding='utf-8') as f:
                f.write(content)
            
            # Verify temp file
            temp_size = Path(temp_path).stat().st_size
            if temp_size == 0:
                os.unlink(temp_path)
                return False, "Temporary file is empty after write"
            
            # Atomic move (replace original)
            shutil.move(temp_path, file_path)
            
            # Verify final file
            if not file_path.exists():
                return False, "File does not exist after atomic move"
            
            final_size = file_path.stat().st_size
            if final_size == 0:
                # Restore from backup if available
                if backup_path:
                    shutil.copy2(backup_path, file_path)
                return False, "Final file is empty, restored from backup"
            
            return True, None
            
        except Exception as e:
            # Clean up temp file
            if Path(temp_path).exists():
                os.unlink(temp_path)
            
            # Restore from backup if available
            if backup_path and backup_path.exists():
                shutil.copy2(backup_path, file_path)
                return False, f"Write failed, restored from backup: {e}"
            
            return False, f"Write failed: {e}"


def test_document_protection():
    """Test document protection features."""
    print("\n" + "="*70)
    print("🧪 Testing Document Protection")
    print("="*70 + "\n")
    
    # Test 1: LaTeX validation
    print("Test 1: LaTeX Validation")
    valid_latex = r"""
\documentclass{article}
\begin{document}
\begin{abstract}
This is valid LaTeX.
\end{abstract}
\section{Introduction}
Content here.
\end{document}
"""
    
    invalid_latex = r"""
\documentclass{article}
\begin{document}
\begin{abstract}
Missing end abstract!
\section{Introduction}
Content here.
"""
    
    is_valid, issues = DocumentValidator.validate_latex(valid_latex)
    print(f"  Valid LaTeX: {is_valid} (expected True)")
    
    is_valid, issues = DocumentValidator.validate_latex(invalid_latex)
    print(f"  Invalid LaTeX: {is_valid} (expected False)")
    print(f"  Issues: {issues}")
    
    # Test 2: Document chunking
    print("\nTest 2: Document Chunking")
    long_doc = "\n".join([f"Line {i}" for i in range(500)])
    chunks = DocumentChunker.split_latex_document(long_doc, max_lines_per_chunk=100)
    print(f"  Split 500 lines into {len(chunks)} chunks")
    print(f"  Chunk sizes: {[c['end_line'] - c['start_line'] for c in chunks]}")
    
    # Test 3: Safe file writing
    print("\nTest 3: Safe File Writing")
    test_file = Path("test_safe_write.txt")
    success, error = SafeFileWriter.write_atomic(
        test_file,
        "Test content",
        create_backup=False,
        validate_before_save=False
    )
    print(f"  Write success: {success}")
    if test_file.exists():
        test_file.unlink()
        print(f"  Cleanup: OK")
    
    print("\n" + "="*70)
    print("✅ Document Protection Tests Complete")
    print("="*70 + "\n")


# Export main classes
__all__ = [
    'DocumentValidator',
    'DocumentChunker',
    'SafeFileWriter',
    'test_document_protection',
]
