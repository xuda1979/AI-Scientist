"""
LaTeX processing and compilation utilities.
"""
from __future__ import annotations
import subprocess
import hashlib
from typing import Tuple, Optional
from pathlib import Path


class LaTeXProcessor:
    """Handle LaTeX compilation and validation with caching."""
    
    def __init__(self):
        self.last_tex_hash = None
        self.last_success = False
        self.last_errors = ""
    
    def compile_and_validate(
        self, 
        paper_path: Path, 
        timeout: int = 30,
        force_recompile: bool = False
    ) -> Tuple[bool, str]:
        """Compile LaTeX with intelligent caching."""
        tex_content = paper_path.read_text(encoding="utf-8", errors="ignore")
        current_hash = hashlib.md5(tex_content.encode()).hexdigest()
        
        # Use cache if content hasn't changed and previous compilation succeeded
        if (not force_recompile and 
            current_hash == self.last_tex_hash and 
            self.last_success):
            return True, ""
        
        # Compile LaTeX
        success, errors = self._compile_latex(paper_path, timeout)
        
        # Update cache
        self.last_tex_hash = current_hash
        self.last_success = success
        self.last_errors = errors if not success else ""
        
        return success, errors
    
    def _compile_latex(self, paper_path: Path, timeout: int) -> Tuple[bool, str]:
        """Perform actual LaTeX compilation."""

        def run_pdflatex() -> subprocess.CompletedProcess:
            return subprocess.run(
                ["pdflatex", "-interaction=nonstopmode", paper_path.name],
                cwd=paper_path.parent,
                capture_output=True,
                text=True,
                timeout=timeout
            )

        output_chunks = []

        try:
            # First LaTeX run to generate auxiliary data
            result = run_pdflatex()
            output_chunks.append(result.stdout + result.stderr)

            if result.returncode != 0:
                return False, "".join(output_chunks)

            needs_bibliography = False
            aux_path = paper_path.with_suffix('.aux')
            if aux_path.exists():
                try:
                    aux_content = aux_path.read_text(encoding="utf-8", errors="ignore")
                    bibliography_indicators = ("\\bibdata", "\\bibstyle", "\\citation{")
                    needs_bibliography = any(indicator in aux_content for indicator in bibliography_indicators)
                except Exception:
                    needs_bibliography = False

            if needs_bibliography:
                try:
                    bibtex_result = subprocess.run(
                        ["bibtex", paper_path.stem],
                        cwd=paper_path.parent,
                        capture_output=True,
                        text=True,
                        timeout=timeout
                    )
                except FileNotFoundError:
                    output_chunks.append("BibTeX not found. Install bibtex to build references. ")
                    return False, "".join(output_chunks)

                output_chunks.append(bibtex_result.stdout + bibtex_result.stderr)

                if bibtex_result.returncode != 0:
                    return False, "".join(output_chunks)

                # Run pdflatex twice more to incorporate bibliography
                for _ in range(2):
                    result = run_pdflatex()
                    output_chunks.append(result.stdout + result.stderr)
                    if result.returncode != 0:
                        return False, "".join(output_chunks)
            else:
                # Second pass for cross-references when bibliography is not needed
                result = run_pdflatex()
                output_chunks.append(result.stdout + result.stderr)
                if result.returncode != 0:
                    return False, "".join(output_chunks)

            return True, "".join(output_chunks)

        except subprocess.TimeoutExpired:
            return False, f"LaTeX compilation timed out after {timeout}s"
        except Exception as e:
            return False, f"LaTeX compilation error: {str(e)}"
        finally:
            # Clean up auxiliary files
            self._cleanup_aux_files(paper_path)
    
    def _cleanup_aux_files(self, paper_path: Path) -> None:
        """Clean up LaTeX auxiliary files."""
        aux_extensions = ['.aux', '.log', '.out', '.toc', '.bbl', '.blg']
        base_name = paper_path.stem
        
        for ext in aux_extensions:
            aux_file = paper_path.parent / f"{base_name}{ext}"
            if aux_file.exists():
                try:
                    aux_file.unlink()
                except Exception:
                    pass  # Ignore cleanup errors
    
    def calculate_dynamic_timeout(self, tex_content: str) -> int:
        """Calculate compilation timeout based on document complexity."""
        base_timeout = 30
        
        # Add time based on content length
        length_factor = len(tex_content) // 10000
        
        # Add time based on complexity indicators
        complexity_indicators = [
            r"\\includegraphics",
            r"\\begin\{figure\}",
            r"\\begin\{table\}",
            r"\\begin\{equation\}",
            r"\\begin\{align\}",
            r"\\cite\{",
            r"\\bibliography"
        ]
        
        import re
        complexity_score = sum(
            len(re.findall(pattern, tex_content)) 
            for pattern in complexity_indicators
        )
        
        timeout = base_timeout + length_factor + (complexity_score * 2)
        return min(timeout, 300)  # Cap at 5 minutes
    
    def validate_latex_structure(self, tex_content: str) -> Tuple[bool, list]:
        """Validate LaTeX document structure."""
        issues = []
        
        # Check for required document structure
        if "\\documentclass" not in tex_content:
            issues.append("Missing \\documentclass")
        
        if "\\begin{document}" not in tex_content:
            issues.append("Missing \\begin{document}")
        
        if "\\end{document}" not in tex_content:
            issues.append("Missing \\end{document}")
        
        # Check for balanced environments
        import re
        
        # Find all \begin{env} and \end{env} pairs
        begin_envs = re.findall(r"\\begin\{([^}]+)\}", tex_content)
        end_envs = re.findall(r"\\end\{([^}]+)\}", tex_content)
        
        # Check if environments are balanced
        for env in set(begin_envs + end_envs):
            begin_count = begin_envs.count(env)
            end_count = end_envs.count(env)
            if begin_count != end_count:
                issues.append(f"Unbalanced {env} environment ({begin_count} begin, {end_count} end)")
        
        # Check for common LaTeX errors
        error_patterns = [
            (r"\\cite\{[^}]*,\s*\}", "Trailing comma in citation"),
            (r"\\ref\{\s*\}", "Empty reference"),
            (r"\\label\{\s*\}", "Empty label"),
            (r"\$\$[^$]*\$\$", "Use of $$ instead of equation environment"),
        ]
        
        for pattern, message in error_patterns:
            if re.search(pattern, tex_content):
                issues.append(message)
        
        return len(issues) == 0, issues
    
    def generate_pdf_for_review(self, paper_path: Path, timeout: int) -> Tuple[bool, Optional[Path], str]:
        """Generate PDF specifically for AI review."""
        success, errors = self.compile_and_validate(paper_path, timeout, force_recompile=True)
        
        if success:
            pdf_path = paper_path.with_suffix('.pdf')
            if pdf_path.exists():
                return True, pdf_path, ""
            else:
                return False, None, "PDF not generated despite successful compilation"
        else:
            return False, None, errors
    
    def get_compilation_stats(self) -> dict:
        """Get compilation statistics."""
        return {
            "last_success": self.last_success,
            "has_cached_result": self.last_tex_hash is not None,
            "last_errors": self.last_errors
        }
