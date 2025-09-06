from __future__ import annotations
from pathlib import Path
from typing import Optional, Tuple


def run_review_revision_step(
    current_tex: str,
    sim_summary: str,
    latex_errors: str,
    project_dir: Path,
    user_prompt: Optional[str],
    iteration: int,
    model: str,
    request_timeout: int,
    config,
    pdf_path: Optional[Path],
    output_diffs: bool,
    paper_path: Path,
) -> Tuple[str, str]:
    """Run combined review and revision step."""
    from sciresearch_workflow import (
        _combined_review_edit_revise_prompt,
        _universal_chat,
        _parse_combined_response,
        _apply_file_changes,
        _revise_prompt,
        _save_iteration_diff,
    )

    combined_response = _universal_chat(
        _combined_review_edit_revise_prompt(current_tex, sim_summary, latex_errors, project_dir, user_prompt, iteration),
        model=model,
        request_timeout=request_timeout,
        prompt_type="combined_review_edit_revise",
        fallback_models=config.fallback_models,
        pdf_path=pdf_path,
    )
    review, decision, file_changes = _parse_combined_response(combined_response, project_dir)

    original_content = paper_path.read_text(encoding="utf-8", errors="ignore") if output_diffs else None

    if file_changes:
        _apply_file_changes(file_changes, project_dir)
        if output_diffs and original_content is not None:
            new_content = paper_path.read_text(encoding="utf-8", errors="ignore")
            _save_iteration_diff(original_content, new_content, project_dir, iteration, "paper.tex")
    else:
        revised = _universal_chat(
            _revise_prompt(current_tex, sim_summary, review, latex_errors, project_dir, user_prompt),
            model=model,
            request_timeout=request_timeout,
            prompt_type="revise",
            fallback_models=config.fallback_models,
            pdf_path=pdf_path,
        )
        if revised.strip():
            paper_path.write_text(revised, encoding="utf-8")
            if output_diffs and original_content is not None:
                _save_iteration_diff(original_content, revised, project_dir, iteration, "paper.tex")

    return review, decision
