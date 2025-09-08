"""
File management and project setup utilities.
"""
from __future__ import annotations
import logging
import shutil
import hashlib
from typing import Tuple, Optional
from pathlib import Path


# Compatibility function to replace utils.sim_runner.ensure_single_tex_py
def ensure_single_tex_py(project_dir: Path, strict: bool = True, preserve_original_filename: bool = False) -> Tuple[Path, Path]:
    """Compatibility wrapper for ensure_single_tex_file."""
    file_manager = FileManager()
    return file_manager.ensure_single_tex_file(project_dir, strict, preserve_original_filename)


class FileManager:
    """Handle file operations and project setup."""
    
    def __init__(self):
        self.file_hashes = {}
    
    def prepare_project_directory(self, output_dir: Path, modify_existing: bool = False) -> Path:
        """Prepare project directory for workflow."""
        if modify_existing and output_dir.exists():
            print(f"Using existing project directory: {output_dir}")
            return output_dir
        
        # Create new project directory
        if output_dir.exists() and not modify_existing:
            # Create unique directory name
            counter = 1
            base_name = output_dir.name
            while output_dir.exists():
                output_dir = output_dir.parent / f"{base_name}_{counter}"
                counter += 1
        
        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"Created project directory: {output_dir}")
        
        return output_dir
    
    def ensure_single_tex_file(
        self, 
        project_dir: Path, 
        strict: bool = True,
        preserve_original_filename: bool = False
    ) -> Tuple[Path, Path]:
        """Ensure exactly one .tex file and one .py file exist."""
        tex_files = list(project_dir.glob("*.tex"))
        py_files = list(project_dir.glob("*.py"))
        
        # Handle .tex files
        if len(tex_files) == 0:
            # Create minimal template
            paper_path = project_dir / "paper.tex"
            paper_path.write_text(
                "\\documentclass{article}\\begin{document}\\end{document}",
                encoding="utf-8"
            )
        elif len(tex_files) == 1:
            paper_path = tex_files[0]
            # Rename to paper.tex if needed (unless preserving original filename)
            if not preserve_original_filename and paper_path.name != "paper.tex":
                new_path = project_dir / "paper.tex"
                paper_path.rename(new_path)
                paper_path = new_path
        else:
            if strict:
                raise ValueError(f"Multiple .tex files found: {[f.name for f in tex_files]}")
            # Use the largest file
            paper_path = max(tex_files, key=lambda f: f.stat().st_size)
            # Remove others
            for f in tex_files:
                if f != paper_path:
                    f.unlink()
        
        # Handle .py files  
        if len(py_files) == 0:
            # Create minimal simulation template
            sim_path = project_dir / "simulation.py"
            sim_path.write_text(
                "# Simulation code will be extracted from LaTeX\\nprint('No simulation code found')",
                encoding="utf-8"
            )
        elif len(py_files) == 1:
            sim_path = py_files[0]
            # Rename to simulation.py if needed
            if sim_path.name != "simulation.py":
                new_path = project_dir / "simulation.py"
                sim_path.rename(new_path)
                sim_path = new_path
        else:
            if strict:
                raise ValueError(f"Multiple .py files found: {[f.name for f in py_files]}")
            # Use the largest file
            sim_path = max(py_files, key=lambda f: f.stat().st_size)
            # Remove others
            for f in py_files:
                if f != sim_path:
                    f.unlink()
        
        return paper_path, sim_path
    
    def check_file_changed(self, file_path: Path) -> bool:
        """Check if file has changed since last check."""
        if not file_path.exists():
            return False
        
        content = file_path.read_text(encoding="utf-8", errors="ignore")
        current_hash = hashlib.md5(content.encode()).hexdigest()
        
        file_key = str(file_path)
        if file_key not in self.file_hashes:
            self.file_hashes[file_key] = current_hash
            return True
        
        if self.file_hashes[file_key] != current_hash:
            self.file_hashes[file_key] = current_hash
            return True
        
        return False
    
    def backup_file(self, file_path: Path, backup_suffix: str = ".bak") -> Path:
        """Create backup of file."""
        backup_path = file_path.with_suffix(file_path.suffix + backup_suffix)
        shutil.copy2(file_path, backup_path)
        return backup_path
    
    def save_iteration_diff(
        self, 
        original_content: str, 
        new_content: str, 
        project_dir: Path, 
        iteration: int,
        filename: str = "paper.tex"
    ) -> None:
        """Save diff between iterations."""
        import difflib
        
        diff_dir = project_dir / "diffs"
        diff_dir.mkdir(exist_ok=True)
        
        # Generate unified diff
        diff = difflib.unified_diff(
            original_content.splitlines(keepends=True),
            new_content.splitlines(keepends=True),
            fromfile=f"{filename} (iteration {iteration-1})",
            tofile=f"{filename} (iteration {iteration})",
            lineterm=""
        )
        
        diff_path = diff_dir / f"iteration_{iteration}_{filename}.diff"
        with open(diff_path, 'w', encoding='utf-8') as f:
            f.writelines(diff)
        
        logging.info(f"Diff saved: {diff_path.name}")
    
    def cleanup_temp_files(self, project_dir: Path) -> None:
        """Clean up temporary files."""
        temp_patterns = ["*.aux", "*.log", "*.out", "*.toc", "*.bbl", "*.blg", "*.fls", "*.fdb_latexmk"]
        
        for pattern in temp_patterns:
            for temp_file in project_dir.glob(pattern):
                try:
                    temp_file.unlink()
                except Exception:
                    pass  # Ignore cleanup errors
    
    def get_file_stats(self, project_dir: Path) -> dict:
        """Get statistics about project files."""
        stats = {
            "tex_files": len(list(project_dir.glob("*.tex"))),
            "py_files": len(list(project_dir.glob("*.py"))),
            "pdf_files": len(list(project_dir.glob("*.pdf"))),
            "total_files": len(list(project_dir.glob("*"))),
            "project_size_mb": sum(f.stat().st_size for f in project_dir.rglob("*") if f.is_file()) / 1024 / 1024
        }
        return stats
