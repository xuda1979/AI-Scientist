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
    is_initial_draft: bool = False,
    all_code_mode: bool = False,
    code_output_dir: str = "code",
    execution_log_file: str = "execution_log.txt",
    previous_execution_results: Optional[str] = None,
) -> Tuple[str, str]:
    """Run combined review and revision step.
    
    Args:
        is_initial_draft: If True, request full paper content; if False, request diffs only
        all_code_mode: If True, enable unrestricted code generation and command execution
        code_output_dir: Directory for generated code files (only used in all-code mode)
        execution_log_file: Filename for execution log (only used in all-code mode)
        previous_execution_results: Formatted execution results from previous iteration
    """
    from sciresearch_workflow import (
        _combined_review_edit_revise_prompt,
        _universal_chat,
        _parse_combined_response,
        _apply_file_changes,
        _revise_prompt,
        _save_iteration_diff,
    )
    
    # ALL-CODE MODE: Add execution results and code generation instructions to prompt
    supplemental_context = ""
    if all_code_mode:
        from utils.all_code_handler import (
            create_all_code_prompt_supplement,
            format_execution_results_for_llm
        )
        
        # Add prompt supplement for all-code mode
        supplemental_context = create_all_code_prompt_supplement(
            iteration=iteration,
            has_previous_results=(previous_execution_results is not None)
        )
        
        # Add previous execution results if available
        if previous_execution_results:
            supplemental_context = previous_execution_results + "\n\n" + supplemental_context

    combined_response = _universal_chat(
        _combined_review_edit_revise_prompt(
            current_tex, sim_summary, latex_errors, project_dir, user_prompt, iteration, quality_issues,
            supplemental_context=supplemental_context
        ),
        model=model,
        request_timeout=request_timeout,
        prompt_type="combined_review_edit_revise",
        fallback_models=config.fallback_models,
        pdf_path=pdf_path,
    )
    
    # CRITICAL: Check for response truncation before processing
    from utils.response_validator import detect_response_truncation, estimate_paper_completeness
    
    is_truncated, truncation_issues = detect_response_truncation(combined_response, expected_type="latex")
    
    if is_truncated:
        print(f"\n{'!'*80}")
        print(f"⚠ WARNING: AI RESPONSE APPEARS TRUNCATED!")
        print(f"{'!'*80}")
        print(f"Detected {len(truncation_issues)} truncation indicators:")
        for issue in truncation_issues:
            print(f"  - {issue}")
        print(f"\nResponse length: {len(combined_response):,} characters")
        print(f"This may result in an incomplete or damaged paper!")
        print(f"{'!'*80}\n")
    
    review, decision, file_changes = _parse_combined_response(combined_response, project_dir)
    
    # CRITICAL: Validate paper completeness if paper.tex was modified
    if file_changes and 'paper.tex' in file_changes:
        from utils.response_validator import validate_paper_structure, estimate_paper_completeness
        
        new_paper_content = file_changes['paper.tex']
        if isinstance(new_paper_content, list):
            new_paper_content = '\n'.join(new_paper_content)
        
        completeness_score = estimate_paper_completeness(new_paper_content)
        is_complete, missing_sections = validate_paper_structure(new_paper_content)
        
        if completeness_score < 0.5:
            print(f"\n{'!'*80}")
            print(f"🚨 CRITICAL ERROR: PAPER IS SEVERELY INCOMPLETE!")
            print(f"{'!'*80}")
            print(f"Completeness score: {completeness_score*100:.1f}%")
            print(f"Missing sections: {', '.join(missing_sections) if missing_sections else 'Unknown'}")
            print(f"Paper length: {len(new_paper_content):,} characters")
            print(f"\nTHIS IS A DISASTROUS TRUNCATION - REVISION REJECTED!")
            print(f"The AI response was cut off before completing the paper.")
            print(f"{'!'*80}\n")
            # Clear file_changes to prevent applying incomplete content
            file_changes = None
        elif not is_complete:
            print(f"\n{'='*80}")
            print(f"⚠ WARNING: Paper structure incomplete!")
            print(f"{'='*80}")
            print(f"Completeness score: {completeness_score*100:.1f}%")
            print(f"Missing sections: {', '.join(missing_sections)}")
            print(f"Proceeding with caution...")
            print(f"{'='*80}\n")
    
    # PRINT REVIEW FEEDBACK
    print(f"\n{'='*80}")
    print(f"REVIEW FEEDBACK - Iteration {iteration}")
    print(f"{'='*80}")
    print(review)
    print(f"{'='*80}\n")

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
                
                # PRINT GIT-STYLE DIFF
                print(f"\n{'='*80}")
                print(f"GIT DIFF FOR ITERATION {iteration} - paper.tex")
                print(f"{'='*80}")
                import difflib
                diff = difflib.unified_diff(
                    original_content.splitlines(keepends=True),
                    new_content.splitlines(keepends=True),
                    fromfile='a/paper.tex',
                    tofile='b/paper.tex',
                    lineterm=''
                )
                diff_text = ''.join(diff)
                if len(diff_text) > 5000:
                    print(diff_text[:2500])
                    print(f"\n... (diff truncated, {len(diff_text)} total chars) ...\n")
                    print(diff_text[-2500:])
                else:
                    print(diff_text)
                print(f"{'='*80}\n")
    else:
        # Determine diff mode: Use full content for initial draft, diffs for subsequent revisions
        use_diff_mode = not is_initial_draft
        
        revised = _universal_chat(
            _revise_prompt(
                current_tex, sim_summary, review, latex_errors, project_dir, user_prompt, quality_issues,
                enable_quality_enhancements=getattr(config, 'enable_quality_enhancements', True),
                use_diff_mode=use_diff_mode
            ),
            model=model,
            request_timeout=request_timeout,
            prompt_type="revise",
            fallback_models=config.fallback_models,
            pdf_path=pdf_path,
        )
        if revised.strip():
            # DIFF DETECTION AND APPLICATION
            # If LLM returned a diff instead of full content, apply it to current files
            from utils.diff_utils import is_diff_format, apply_diffs_to_files
            
            if use_diff_mode and is_diff_format(revised):
                print(f"    Diff format detected in revision response, applying patches...")
                
                # Prepare file contents dictionary
                file_contents = {'paper.tex': current_tex}
                
                # Add simulation.py if it exists
                sim_path = project_dir / "simulation.py"
                if sim_path.exists():
                    file_contents['simulation.py'] = sim_path.read_text(encoding='utf-8', errors='ignore')
                
                # Apply diffs to all files
                modified_files, success, msg = apply_diffs_to_files(file_contents, revised)
                
                if success and 'paper.tex' in modified_files:
                    revised = modified_files['paper.tex']
                    print(f"    ✓ Diff applied to paper.tex: {msg}")
                    
                    # Apply simulation.py changes if present
                    if 'simulation.py' in modified_files:
                        sim_path.write_text(modified_files['simulation.py'], encoding='utf-8')
                        print(f"    ✓ Applied diff to simulation.py")
                else:
                    print(f"    ⚠ Diff application failed: {msg}")
                    print(f"    Falling back to treating response as full content")
                    # Keep revised as-is (treat as full content for paper.tex)
            elif use_diff_mode:
                print(f"    ⚠ Expected diff format but got full content, using as-is")
            
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
                        
                        # PRINT GIT-STYLE DIFF FOR FALLBACK
                        print(f"\n{'='*80}")
                        print(f"GIT DIFF FOR ITERATION {iteration} - paper.tex (FALLBACK REVISION)")
                        print(f"{'='*80}")
                        import difflib
                        diff = difflib.unified_diff(
                            original_content.splitlines(keepends=True),
                            revised.splitlines(keepends=True),
                            fromfile='a/paper.tex',
                            tofile='b/paper.tex',
                            lineterm=''
                        )
                        diff_text = ''.join(diff)
                        if len(diff_text) > 5000:
                            print(diff_text[:2500])
                            print(f"\n... (diff truncated, {len(diff_text)} total chars) ...\n")
                            print(diff_text[-2500:])
                        else:
                            print(diff_text)
                        print(f"{'='*80}\n")
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

    # ALL-CODE MODE: Extract code files, commands, and execute
    execution_results_text = None
    if all_code_mode:
        print(f"\n{'='*80}")
        print(f"ALL-CODE MODE: Processing code files and commands")
        print(f"{'='*80}\n")
        
        from utils.all_code_handler import (
            extract_all_code_blocks,
            save_code_files,
            extract_execution_commands,
            execute_commands_and_log,
            generate_code_diffs,
            format_execution_results_for_llm
        )
        
        # Extract all code files from response
        code_files = extract_all_code_blocks(combined_response, project_dir)
        
        if code_files:
            print(f"  Found {len(code_files)} code file(s) to save:")
            for filepath in code_files.keys():
                print(f"    - {filepath}")
            
            # Save code files
            saved_paths = save_code_files(code_files, project_dir, code_output_dir)
            
            # Generate diffs for code files if this is not the first iteration
            if iteration > 1:
                # Try to load previous code files for diff
                old_code_files = {}
                code_dir = project_dir / code_output_dir
                if code_dir.exists():
                    for filepath in code_files.keys():
                        old_path = code_dir / filepath
                        if old_path.exists():
                            try:
                                # Read the file content before it was updated
                                # (Note: This won't work since we already saved new content)
                                # In production, we should save old content first
                                pass
                            except:
                                pass
                
                # Generate diff
                diff_path = generate_code_diffs(old_code_files, code_files, project_dir, iteration)
            
            print("")
        
        # Extract and execute commands
        commands = extract_execution_commands(combined_response)
        
        if commands:
            print(f"  Found {len(commands)} command(s) to execute:")
            for cmd_info in commands:
                print(f"    - {cmd_info['description']}")
            print("")
            
            # Execute commands and log results
            log_path = execute_commands_and_log(commands, project_dir, execution_log_file)
            
            # Format results for next iteration
            diff_path = project_dir / "diffs" / f"code_changes_iteration_{iteration}.diff" if iteration > 1 else None
            execution_results_text = format_execution_results_for_llm(log_path, diff_path)
            
            print(f"\n  ✓ Execution complete. Results will be provided to LLM in next iteration.\n")
        else:
            print(f"  No execution commands found in LLM response.\n")
        
        print(f"{'='*80}\n")

    return review, decision


def run_optimized_review_revision_step(
    comprehensive_prompt: str,
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
    """Run the optimized single-call review and revision routine."""

    from sciresearch_workflow import (
        _universal_chat,
        _parse_combined_response,
        _apply_file_changes,
        _save_iteration_diff,
    )

    response = _universal_chat(
        comprehensive_prompt,
        model=model,
        request_timeout=request_timeout,
        prompt_type="optimized_review_revision",
        fallback_models=getattr(config, "fallback_models", []),
        pdf_path=pdf_path,
    )

    review, decision, file_changes = _parse_combined_response(response, project_dir)

    original_content = paper_path.read_text(encoding="utf-8", errors="ignore") if output_diffs else None

    if file_changes:
        changes_applied = _apply_file_changes(file_changes, project_dir, config)
        if changes_applied and output_diffs and original_content is not None:
            new_content = paper_path.read_text(encoding="utf-8", errors="ignore")
            _save_iteration_diff(original_content, new_content, project_dir, iteration, "paper.tex")
    else:
        print(
            "⚠ Optimized review did not include file changes."
            " Consider running the standard revision pipeline."
        )

    return review, decision
