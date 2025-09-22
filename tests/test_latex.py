"""Tests for LaTeX processing utilities."""
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import MagicMock, patch

from processing.latex import LaTeXProcessor


MINIMAL_TEX = "\\documentclass{article}\\begin{document}Hello\\end{document}"


def test_compile_and_validate_uses_cache_and_cleans_aux_files():
    """Successful compilation should clean aux files and reuse cached result."""
    processor = LaTeXProcessor()
    with TemporaryDirectory() as tmpdir:
        project_dir = Path(tmpdir)
        paper_path = project_dir / "paper.tex"
        paper_path.write_text(MINIMAL_TEX, encoding="utf-8")
        aux_file = project_dir / "paper.aux"
        aux_file.write_text("aux", encoding="utf-8")

        mock_result = MagicMock(returncode=0, stdout="compiled", stderr="")

        with patch("processing.latex.subprocess.run", return_value=mock_result) as mock_run:
            success, output = processor.compile_and_validate(paper_path)
            assert success is True
            assert "compiled" in output
            assert not aux_file.exists()
            mock_run.assert_called_once()

            cached_success, cached_output = processor.compile_and_validate(paper_path)
            assert cached_success is True
            assert cached_output == ""
            mock_run.assert_called_once()


def test_compile_and_validate_failure_reports_errors():
    """Failed compilation should surface combined stdout/stderr."""
    processor = LaTeXProcessor()
    with TemporaryDirectory() as tmpdir:
        project_dir = Path(tmpdir)
        paper_path = project_dir / "paper.tex"
        paper_path.write_text(MINIMAL_TEX, encoding="utf-8")

        mock_result = MagicMock(returncode=1, stdout="warning", stderr="error")

        with patch("processing.latex.subprocess.run", return_value=mock_result):
            success, errors = processor.compile_and_validate(paper_path)
            assert success is False
            assert "warning" in errors
            assert "error" in errors


def test_compile_and_validate_handles_exceptions():
    """Unexpected subprocess errors should be reported gracefully."""
    processor = LaTeXProcessor()
    with TemporaryDirectory() as tmpdir:
        project_dir = Path(tmpdir)
        paper_path = project_dir / "paper.tex"
        paper_path.write_text(MINIMAL_TEX, encoding="utf-8")

        with patch("processing.latex.subprocess.run", side_effect=RuntimeError("boom")):
            success, message = processor.compile_and_validate(paper_path)
            assert success is False
            assert "boom" in message


def test_generate_pdf_for_review_success():
    """PDF should be returned when compilation succeeds and file exists."""
    processor = LaTeXProcessor()
    with TemporaryDirectory() as tmpdir:
        project_dir = Path(tmpdir)
        paper_path = project_dir / "paper.tex"
        paper_path.write_text(MINIMAL_TEX, encoding="utf-8")
        pdf_path = paper_path.with_suffix(".pdf")
        pdf_path.write_text("pdf", encoding="utf-8")

        with patch.object(processor, "compile_and_validate", return_value=(True, "")) as mock_compile:
            success, resolved_pdf, errors = processor.generate_pdf_for_review(paper_path, timeout=42)

        assert success is True
        assert resolved_pdf == pdf_path
        assert errors == ""
        mock_compile.assert_called_once_with(paper_path, 42, force_recompile=True)


def test_generate_pdf_for_review_handles_missing_pdf():
    """A successful compile without a PDF should report the issue."""
    processor = LaTeXProcessor()
    with TemporaryDirectory() as tmpdir:
        project_dir = Path(tmpdir)
        paper_path = project_dir / "paper.tex"
        paper_path.write_text(MINIMAL_TEX, encoding="utf-8")

        with patch.object(processor, "compile_and_validate", return_value=(True, "")):
            success, resolved_pdf, errors = processor.generate_pdf_for_review(paper_path, timeout=10)

        assert success is False
        assert resolved_pdf is None
        assert "PDF not generated" in errors


def test_generate_pdf_for_review_failure():
    """Compilation failures should propagate errors."""
    processor = LaTeXProcessor()
    with TemporaryDirectory() as tmpdir:
        project_dir = Path(tmpdir)
        paper_path = project_dir / "paper.tex"
        paper_path.write_text(MINIMAL_TEX, encoding="utf-8")

        with patch.object(processor, "compile_and_validate", return_value=(False, "failure")):
            success, resolved_pdf, errors = processor.generate_pdf_for_review(paper_path, timeout=10)

        assert success is False
        assert resolved_pdf is None
        assert errors == "failure"
