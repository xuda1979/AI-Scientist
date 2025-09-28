"""Tests for file management utilities."""
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest

from processing.files import FileManager


def test_ensure_single_tex_file_creates_defaults():
    """Ensure default files are created when none exist."""
    manager = FileManager()
    with TemporaryDirectory() as tmpdir:
        project_dir = Path(tmpdir)

        paper_path, sim_path = manager.ensure_single_tex_file(project_dir)

        assert paper_path.name == "paper.tex"
        assert sim_path.name == "simulation.py"
        assert paper_path.exists()
        assert sim_path.exists()

        tex_contents = paper_path.read_text(encoding="utf-8")
        py_contents = sim_path.read_text(encoding="utf-8")
        assert "\\documentclass" in tex_contents
        assert "\\section{Introduction}" in tex_contents
        assert "run_experiments" in py_contents
        assert "Experiment scaffold generated" in py_contents


def test_ensure_single_tex_file_renames_existing_files():
    """Existing .tex and .py files should be renamed to canonical names."""
    manager = FileManager()
    with TemporaryDirectory() as tmpdir:
        project_dir = Path(tmpdir)
        custom_tex = project_dir / "custom.tex"
        custom_py = project_dir / "model.py"
        custom_tex.write_text("\\documentclass{article}", encoding="utf-8")
        custom_py.write_text("print('hi')", encoding="utf-8")

        paper_path, sim_path = manager.ensure_single_tex_file(project_dir)

        assert paper_path == project_dir / "paper.tex"
        assert sim_path == project_dir / "simulation.py"
        assert not custom_tex.exists()
        assert not custom_py.exists()


@pytest.mark.parametrize("ext", [".aux", ".log", ".out", ".toc", ".bbl", ".blg", ".fls", ".fdb_latexmk"])
def test_cleanup_temp_files_removes_auxiliary_files(ext):
    """Temporary LaTeX artifacts should be cleaned up."""
    manager = FileManager()
    with TemporaryDirectory() as tmpdir:
        project_dir = Path(tmpdir)
        temp_file = project_dir / f"paper{ext}"
        temp_file.write_text("temp", encoding="utf-8")

        manager.cleanup_temp_files(project_dir)

        assert not temp_file.exists()
