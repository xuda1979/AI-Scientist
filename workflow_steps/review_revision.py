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
    quality_issues: Optional[list] = None,
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
        _combined_review_edit_revise_prompt(current_tex, sim_summary, latex_errors, project_dir, user_prompt, iteration, quality_issues),
        model=model,
        request_timeout=request_timeout,
        prompt_type="combined_review_edit_revise",
        fallback_models=config.fallback_models,
        pdf_path=pdf_path,
    )
    review, decision, file_changes = _parse_combined_response(combined_response, project_dir)

    original_content = paper_path.read_text(encoding="utf-8", errors="ignore") if output_diffs else None

    if file_changes:
        changes_applied = _apply_file_changes(file_changes, project_dir, config)
        if not changes_applied:
            print("⚠ Content protection prevented revision - using fallback revision method")
            # Fall back to the simple revision method if changes were rejected
            file_changes = None
        else:
            if output_diffs and original_content is not None:
                new_content = paper_path.read_text(encoding="utf-8", errors="ignore")
                _save_iteration_diff(original_content, new_content, project_dir, iteration, "paper.tex")
    else:
        revised = _universal_chat(
            _revise_prompt(current_tex, sim_summary, review, latex_errors, project_dir, user_prompt, quality_issues),
            model=model,
            request_timeout=request_timeout,
            prompt_type="revise",
            fallback_models=config.fallback_models,
            pdf_path=pdf_path,
        )
        if revised.strip():
            # Apply content protection to fallback revision
            from utils.content_protection import ContentProtector
            
            enable_protection = getattr(config, 'enable_content_protection', True)
            auto_approve = getattr(config, 'auto_approve_safe_changes', False)
            
            if enable_protection:
                protector = ContentProtector(project_dir)
                
                # Create backup
                from datetime import datetime
                backup_path = protector.create_backup(paper_path, f"paper_pre_fallback_revision_{datetime.now().strftime('%H%M%S')}")
                
                # Validate revision
                approved, analysis = protector.validate_revision(current_tex, revised, auto_approve)
                
                if approved:
                    paper_path.write_text(revised, encoding="utf-8")
                    change_percent = analysis.word_count_change_percent
                    print(f"✓ Applied fallback revision: {analysis.old_metrics.word_count:,} → {analysis.new_metrics.word_count:,} words ({change_percent:+.1f}%)")
                    
                    if output_diffs and original_content is not None:
                        _save_iteration_diff(original_content, revised, project_dir, iteration, "paper.tex")
                else:
                    print("❌ Fallback revision also rejected by content protection - keeping original content")
                    print("   This indicates the AI model is making unsafe changes. Manual review recommended.")
            else:
                # Content protection disabled - apply changes directly
                print("⚠ Applying fallback revision WITHOUT content protection (DANGEROUS)")
                paper_path.write_text(revised, encoding="utf-8")
                print(f"✓ Applied fallback revision (content protection disabled)")
                
                if output_diffs and original_content is not None:
                    _save_iteration_diff(original_content, revised, project_dir, iteration, "paper.tex")

    return review, decision
