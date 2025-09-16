#!/usr/bin/env python3
"""
Refactored SciResearch Workflow - Main orchestration module.
"""
from __future__ import annotations
import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Optional, List

# Core modules
from core.config import WorkflowConfig, setup_workflow_logging, timeout_input, _prepare_project_dir
from ai.chat import _universal_chat
from latex.compiler import _compile_latex_and_get_errors, _generate_pdf_for_review, _calculate_dynamic_timeout
from evaluation.quality import _extract_quality_metrics, _calculate_quality_score, _validate_research_quality
from generation.content import _save_iteration_diff, _extract_paper_metadata

# Workflow steps
from workflow_steps.initial_draft import generate_initial_draft
from workflow_steps.simulation import run_simulation_step
from workflow_steps.review_revision import run_review_revision_step

# Utils
from utils.sim_runner import ensure_single_tex_py, extract_simulation_from_tex

# Document types
from document_types import infer_document_type


def _check_existing_paper(output_dir: Path) -> Optional[tuple[str, str, str]]:
    """Check if paper already exists and extract metadata."""
    paper_path = output_dir / "paper.tex"
    
    if not paper_path.exists():
        return None
    
    try:
        paper_content = paper_path.read_text(encoding="utf-8", errors="ignore")
        if len(paper_content.strip()) < 100:  # Too short to be a real paper
            return None
        
        topic, field, question = _extract_paper_metadata(paper_content)
        return topic, field, question
    
    except Exception as e:
        print(f"Warning: Could not read existing paper: {e}")
        return None


def _validate_references_with_external_apis(paper_content: str, config: WorkflowConfig) -> List[str]:
    """Validate references using external APIs (placeholder for now)."""
    issues = []
    
    # This would integrate with real APIs in production
    # For now, just basic validation
    import re
    
    bibliography_items = re.findall(r'\\bibitem\{[^}]+\}\s*([^\n]+)', paper_content)
    
    if len(bibliography_items) < 10:
        issues.append(f"Insufficient references: {len(bibliography_items)} found, minimum 10 recommended")
    
    # Check for potentially fake references
    fake_indicators = ['example.com', 'sample', 'placeholder', 'test.pdf']
    for ref in bibliography_items:
        if any(indicator in ref.lower() for indicator in fake_indicators):
            issues.append(f"Potentially fake reference: {ref[:50]}...")
    
    return issues


def _validate_figure_generation(paper_content: str, sim_path: Path, project_dir: Path) -> List[str]:
    """Validate that figures can be properly generated."""
    issues = []
    
    # Check for figure references
    import re
    figure_refs = re.findall(r'\\ref\{fig:([^}]+)\}', paper_content)
    figure_labels = re.findall(r'\\label\{fig:([^}]+)\}', paper_content)
    
    for ref in figure_refs:
        if ref not in figure_labels:
            issues.append(f"Figure reference without corresponding label: fig:{ref}")
    
    # Check if simulation generates expected output files
    if sim_path.exists():
        sim_content = sim_path.read_text(encoding='utf-8', errors='ignore')
        
        # Look for common plotting functions
        plot_functions = ['plt.savefig', 'fig.savefig', 'matplotlib', 'seaborn']
        has_plotting = any(func in sim_content for func in plot_functions)
        
        if not has_plotting and len(figure_refs) > 0:
            issues.append("Paper references figures but simulation doesn't appear to generate plots")
    
    return issues


def run_workflow(output_dir: Path, topic: str = "", field: str = "", question: str = "",
                document_type: str = "auto", model: str = "gpt-4o", 
                modify_existing: bool = False, user_prompt: Optional[str] = None,
                config: Optional[WorkflowConfig] = None, python_exec: Optional[str] = None,
                enable_ideation: bool = True, specify_idea: Optional[str] = None, 
                num_ideas: int = 15, use_test_time_scaling: bool = False,
                revision_candidates: int = 3, draft_candidates: int = 1,
                strict_singletons: bool = True) -> Path:
    """Main workflow orchestration function."""
    
    # Initialize configuration
    if config is None:
        config = WorkflowConfig(
            quality_threshold=1.0,
            max_iterations=4
        )
    
    # Set up logging
    logger = setup_workflow_logging()
    logger.info(f"Starting workflow with config: quality_threshold={config.quality_threshold}, max_iterations={config.max_iterations}")
    
    # Prepare project directory
    project_dir = _prepare_project_dir(output_dir, modify_existing)
    paper_path = project_dir / "paper.tex"
    sim_path = project_dir / "simulation.py"
    
    # Quality tracking
    quality_history = []
    best_quality_score = 0.0
    stagnation_count = 0
    
    print(f"Working with: {project_dir}")
    print(f"Using model: {model}")
    print(f"Max iterations: {config.max_iterations}")
    print(f"Quality threshold: {config.quality_threshold}")
    
    # Check for existing paper
    existing_paper_info = _check_existing_paper(project_dir)
    
    if existing_paper_info:
        topic, field, question = existing_paper_info
        print(f"Detected existing paper - Topic: {topic}, Field: {field}")
        
        # Auto-detect document type
        paper_content = paper_path.read_text(encoding="utf-8", errors="ignore")
        doc_type = infer_document_type(topic, field, question)
        logger.info(f"Auto-detected document type: {doc_type}")
        print(f"Auto-detected document type: {doc_type}")
        
        print(f"Using existing paper content ({len(paper_content)} characters)")
        print("  Skipping ideation phase for existing paper")
    else:
        # Handle new paper creation with optional ideation
        if specify_idea:
            # Use specified idea directly
            print(f"Using specified research idea: {specify_idea}")
            topic = specify_idea
            if not field:
                field = input("Enter field (e.g., 'Computer Science'): ").strip()
            if not question:
                question = f"How to investigate: {specify_idea}"
        elif enable_ideation and not all([topic, field, question]):
            # Run ideation phase
            from generation.content import run_ideation_phase
            print("🧠 Starting research ideation phase...")
            
            # Get basic input for ideation
            if not topic:
                topic = input("Enter research topic or area of interest: ").strip()
            if not field:
                field = input("Enter field (e.g., 'Computer Science'): ").strip()
            
            # Generate and select research ideas
            selected_idea = run_ideation_phase(topic, field, num_ideas, model, config)
            topic = selected_idea["title"]
            question = selected_idea["question"]
            
            print(f"✅ Selected research idea: {topic}")
            print(f"📝 Research question: {question}")
        else:
            # Traditional manual input
            if not topic:
                topic = input("Enter research topic: ").strip()
            if not field:
                field = input("Enter field (e.g., 'Computer Science'): ").strip()
            if not question:
                question = input("Enter research question: ").strip()
        
        # Generate initial draft (with test-time scaling if enabled)
        print(f"Creating paper draft for: {topic}")
        
        if use_test_time_scaling and draft_candidates > 1:
            from performance.scaling import _generate_best_initial_draft_candidate
            print(f"🧮 Using test-time compute scaling with {draft_candidates} candidates")
            draft = _generate_best_initial_draft_candidate(
                topic, field, question, user_prompt, model, config, draft_candidates
            )
        else:
            draft = generate_initial_draft(topic, field, question, user_prompt, model, 3600, config)
        
        paper_path.write_text(draft, encoding="utf-8")
        print("✅ Initial draft created")
    
    # Ensure single tex/py structure
    ensure_single_tex_py(project_dir)
    
    # Extract/refresh simulation from LaTeX
    extract_simulation_from_tex(paper_path, sim_path)
    
    # Main iteration loop
    for i in range(1, config.max_iterations + 1):
        print(f"Starting iteration {i} of {config.max_iterations}")
        
        # Run simulation
        print(f"Running simulation before review {i}...")
        python_interpreter = python_exec or "python"
        sim_summary, _ = run_simulation_step(paper_path, sim_path, project_dir, model, 3600, python_interpreter)
        
        # Compile LaTeX
        print(f"Compiling LaTeX file with pdflatex...")
        current_tex = paper_path.read_text(encoding="utf-8", errors="ignore")
        dynamic_timeout = _calculate_dynamic_timeout(current_tex, config)
        latex_success, latex_errors = _compile_latex_and_get_errors(paper_path, timeout=dynamic_timeout)
        
        if not latex_success:
            print(f" LaTeX compilation failed. Errors will be sent to LLM for fixing.")
            print(f"Error log (last 20 lines):\n{latex_errors}")
        else:
            print(f" LaTeX compilation successful!")
        
        # Quality validation
        quality_issues = _validate_research_quality(current_tex, sim_summary)
        
        # Additional validations
        if config.reference_validation:
            ref_issues = _validate_references_with_external_apis(current_tex, config)
            quality_issues.extend(ref_issues)
        
        if config.figure_validation:
            fig_issues = _validate_figure_generation(current_tex, sim_path, project_dir)
            quality_issues.extend(fig_issues)
        
        if quality_issues:
            print(f"⚠ Quality issues detected ({len(quality_issues)} total):")
            for idx, issue in enumerate(quality_issues[:10], 1):  # Fixed variable name conflict
                print(f"   {idx}. {issue}")
            if len(quality_issues) > 10:
                print(f"   ... and {len(quality_issues) - 10} more issues")
            logger.info(f"Quality issues detected: {len(quality_issues)} issues found")
        else:
            print("✓ No quality issues detected")
            logger.info("No quality issues detected")
        
        # Calculate quality score
        current_metrics = _extract_quality_metrics(current_tex, sim_summary)
        quality_score = _calculate_quality_score(current_metrics, quality_issues)
        quality_history.append(quality_score)
        
        logger.info(f"Iteration {i} quality score: {quality_score:.2f}")
        print(f"Iteration {i} quality score: {quality_score:.2f}")
        
        # Track improvement
        if quality_score > best_quality_score:
            best_quality_score = quality_score
            stagnation_count = 0
        else:
            stagnation_count += 1
        
        # Generate PDF for review if enabled
        pdf_path = None
        if latex_success and config.enable_pdf_review:
            pdf_success, generated_pdf_path, pdf_error = _generate_pdf_for_review(paper_path, dynamic_timeout)
            if pdf_success and generated_pdf_path:
                pdf_path = generated_pdf_path
                file_size = pdf_path.stat().st_size
                print(f"✓ PDF generated for AI review: {pdf_path.name} ({file_size:,} bytes)")
            else:
                print(f"✗ PDF generation failed: {pdf_error}")
        
        # Combined review and revision (with test-time scaling if enabled)
        print(f"Running combined review/editorial/revision process...")
        
        old_content = current_tex
        
        if use_test_time_scaling and revision_candidates > 1:
            from performance.scaling import _generate_best_revision_candidate
            print(f"🧮 Using test-time compute scaling with {revision_candidates} candidates")
            
            # Generate multiple revision candidates and select the best
            new_content = _generate_best_revision_candidate(
                current_tex, sim_summary, latex_errors, project_dir, user_prompt,
                i, model, config, pdf_path, quality_issues, revision_candidates
            )
            
            # Write the best revision
            paper_path.write_text(new_content, encoding="utf-8")
            print(f"✅ Selected best revision from {revision_candidates} candidates")
        else:
            # Standard single revision
            review, decision = run_review_revision_step(
                current_tex, sim_summary, latex_errors, project_dir, user_prompt,
                i, model, 3600, config, pdf_path, config.diff_output_tracking,
                paper_path, quality_issues
            )
        
        print(f"Review completed")
        
        # Check stopping conditions
        meets_quality_threshold = quality_score >= config.quality_threshold
        
        if latex_success and meets_quality_threshold:
            print(f"[OK] Quality threshold met at iteration {i} (score: {quality_score:.2f}) and LaTeX compiles successfully")
            final_metrics = _extract_quality_metrics(current_tex, sim_summary)
            print(f"Final paper metrics: {final_metrics}")
            break
        elif stagnation_count >= 2 and not config.no_early_stopping:
            print(f"[STOP] Quality stagnating for {stagnation_count} iterations. Ending revisions.")
            break
        
        # Save iteration diff if enabled
        if config.diff_output_tracking:
            new_content = paper_path.read_text(encoding="utf-8", errors="ignore")
            _save_iteration_diff(old_content, new_content, project_dir, i)
        
        print(f"Iteration {i}: Combined review and revision completed")
    
    # Final quality report
    print(f"\nQuality progression: {[f'{q:.2f}' for q in quality_history]}")
    print(f"Best quality score achieved: {best_quality_score:.2f}")
    print(f" Workflow completed! Results in: {project_dir}")
    
    return project_dir


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    """Parse command line arguments with complete feature set."""
    parser = argparse.ArgumentParser(
        description="AI-powered research paper generation and improvement system",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
EXAMPLES:
  # Generate new research paper
  python sciresearch_workflow_refactored.py --topic "machine learning" --field "computer science" --question "How to improve neural network efficiency?"
  
  # Modify existing paper
  python sciresearch_workflow_refactored.py --modify-existing --output-dir my_paper
  
  # Use test-time compute scaling
  python sciresearch_workflow_refactored.py --use-test-time-scaling --revision-candidates 5 --topic "quantum computing"
  
  # Run scaling analysis
  python sciresearch_workflow_refactored.py --test-scaling --scaling-candidates "3,5,7,10"

CONFIGURATION:
  Configuration can be specified via --config JSON file or command line arguments.
  Command line arguments override configuration file settings.
        """
    )

    # Check if we're modifying existing
    modify_existing = '--modify-existing' in (argv or sys.argv)
    
    # Core research parameters (only required for new papers)
    if not modify_existing:
        parser.add_argument("--topic", required=False, help="Research topic")
        parser.add_argument("--field", required=False, help="Research field")
        parser.add_argument("--question", required=False, help="Research question")
        parser.add_argument("--document-type", choices=["research_paper", "engineering_paper", "finance_research", 
                          "equity_research", "survey_paper", "presentation_slides", "technical_report", 
                          "white_paper", "conference_paper", "journal_article", "auto"], 
                          default="auto", help="Type of document to generate (auto-detect if not specified)")

    # Basic workflow parameters
    parser.add_argument("--output-dir", default="output", help="Output directory root (contains project subfolder)")
    parser.add_argument("--model", default="gpt-4o", help="AI model to use (default: gpt-4o)")
    parser.add_argument("--request-timeout", type=int, default=3600, help="Per-request timeout seconds (0 means no timeout)")
    parser.add_argument("--max-retries", type=int, default=3, help="Max AI API retries")
    parser.add_argument("--max-iterations", type=int, default=4, help="Max review->revise iterations")
    parser.add_argument("--no-early-stopping", action="store_true", help="Disable early stopping for quality stagnation (run all max iterations)")
    parser.add_argument("--modify-existing", action="store_true", help="If output dir already has paper.tex, modify in place")
    parser.add_argument("--strict-singletons", action="store_true", default=True, help="Keep only paper.tex & simulation.py (others archived)")
    parser.add_argument("--python-exec", default=None, help="Python interpreter for running simulation.py")
    
    # Configuration file support
    parser.add_argument("--config", type=str, default=None, help="Path to configuration JSON file")
    parser.add_argument("--save-config", type=str, default=None, help="Save current configuration to JSON file")
    
    # Quality control parameters
    parser.add_argument("--quality-threshold", type=float, default=1.0, help="Minimum quality score required for acceptance (0.0-1.0)")
    parser.add_argument("--check-references", action="store_true", default=True, help="Enable external reference validation")
    parser.add_argument("--validate-figures", action="store_true", default=True, help="Enable figure generation validation")
    parser.add_argument("--skip-reference-check", action="store_true", help="Disable external reference validation (faster)")
    parser.add_argument("--skip-figure-validation", action="store_true", help="Disable figure generation validation (faster)")
    
    # PDF review parameters
    parser.add_argument("--enable-pdf-review", action="store_true", default=False, help="Send PDF files to AI models during review/revision")
    parser.add_argument("--disable-pdf-review", action="store_true", help="Disable PDF review (text-only review)")
    
    # Ideation parameters - ideation is enabled by default for new papers
    parser.add_argument("--skip-ideation", action="store_true", help="Skip research ideation phase (use original topic directly)")
    parser.add_argument("--specify-idea", type=str, default=None, help="Specify a research idea to use directly (skips ideation phase)")
    parser.add_argument("--num-ideas", type=int, default=15, help="Number of research ideas to generate (10-20)")
    
    # Custom prompt parameter
    parser.add_argument("--user-prompt", type=str, default=None, help="Custom prompt that takes priority over standard requirements")
    
    # Diff output parameters
    parser.add_argument("--no-output-diffs", action="store_true", help="Disable diff file saving for each review/revision cycle")
    parser.add_argument("--output-diffs", action="store_true", default=True, help="Save diff files for each review/revision cycle to track changes (default: enabled)")
    
    # Content protection parameters
    parser.add_argument("--disable-content-protection", action="store_true", help="Disable content protection against accidental deletions (DANGEROUS)")
    parser.add_argument("--auto-approve-changes", action="store_true", help="Automatically approve content changes that pass safety checks")
    parser.add_argument("--content-protection-threshold", type=float, default=0.15, help="Maximum allowed content reduction as fraction (default: 0.15 = 15%%)")
    
    # Test-time compute scaling parameters
    parser.add_argument("--test-scaling", action="store_true", help="Run test-time compute scaling analysis instead of normal workflow")
    parser.add_argument("--scaling-candidates", type=str, default="3,5,7,10", help="Comma-separated list of candidate counts to test (e.g., '3,5,7,10')")
    parser.add_argument("--scaling-timeout", type=int, default=1800, help="Base timeout for scaling tests (seconds)")
    parser.add_argument("--scaling-prompt", type=str, default=None, help="Custom prompt for scaling tests")
    
    # Test-time compute scaling for normal workflow
    parser.add_argument("--use-test-time-scaling", action="store_true", help="Enable test-time compute scaling during revision cycles")
    parser.add_argument("--revision-candidates", type=int, default=3, help="Number of revision candidates to generate when using test-time scaling")
    parser.add_argument("--draft-candidates", type=int, default=1, help="Number of initial draft candidates to generate")

    args = parser.parse_args(argv)
    
    # Handle skip flags and conflicts
    if args.skip_reference_check:
        args.check_references = False
    if args.skip_figure_validation:
        args.validate_figures = False
    
    if args.disable_pdf_review:
        args.enable_pdf_review = False
    if args.no_output_diffs:
        args.output_diffs = False
    if args.disable_content_protection:
        args.enable_content_protection = False
    else:
        args.enable_content_protection = True
    
    # Initialize topic, field, question, document_type attributes if they don't exist (modify-existing mode)
    if modify_existing:
        if not hasattr(args, 'topic'):
            args.topic = None
        if not hasattr(args, 'field'):
            args.field = None
        if not hasattr(args, 'question'):
            args.question = None
        if not hasattr(args, 'document_type'):
            args.document_type = "auto"
    
    return args


def main():
    """Main entry point."""
    args = parse_args()
    
    # Convert output_dir to Path if it's a string
    if isinstance(args.output_dir, str):
        args.output_dir = Path(args.output_dir)
    
    # Load or create configuration
    config = None
    if args.config:
        config = WorkflowConfig.from_file(Path(args.config))
    else:
        config = WorkflowConfig(
            max_iterations=args.max_iterations,
            quality_threshold=args.quality_threshold,
            request_timeout=args.request_timeout,
            max_retries=args.max_retries,
            enable_pdf_review=args.enable_pdf_review,
            reference_validation=args.check_references,
            figure_validation=args.validate_figures,
            diff_output_tracking=args.output_diffs,
            no_early_stopping=args.no_early_stopping,
            content_protection=args.enable_content_protection,
            content_protection_threshold=args.content_protection_threshold,
            auto_approve_changes=args.auto_approve_changes
        )
    
    # Save configuration if requested
    if args.save_config:
        config.save_to_file(Path(args.save_config))
        print(f"📝 Configuration saved to: {args.save_config}")
    
    try:
        # Handle test-time compute scaling analysis mode
        if args.test_scaling:
            from performance.scaling import test_time_compute_scaling
            
            candidates = [int(x.strip()) for x in args.scaling_candidates.split(',')]
            print(f"🧮 Running test-time compute scaling analysis with candidates: {candidates}")
            
            scaling_results = test_time_compute_scaling(
                config=config,
                candidates=candidates,
                timeout=args.scaling_timeout,
                custom_prompt=args.scaling_prompt,
                model=args.model
            )
            
            print(f"✅ Scaling analysis completed!")
            for candidate_count, metrics in scaling_results.items():
                print(f"📊 {candidate_count} candidates: Quality={metrics.get('quality', 0):.2f}, "
                      f"Time={metrics.get('time', 0):.1f}s")
            return
        
        # Normal workflow execution
        # Ideation is enabled by default, only disable if explicitly skipped or idea specified
        enable_ideation = not args.skip_ideation and args.specify_idea is None
        
        result_dir = run_workflow(
            output_dir=args.output_dir,
            topic=args.topic or "",
            field=args.field or "",
            question=args.question or "",
            document_type=getattr(args, 'document_type', 'auto'),
            model=args.model,
            modify_existing=args.modify_existing,
            user_prompt=args.user_prompt,
            config=config,
            python_exec=args.python_exec,
            enable_ideation=enable_ideation,
            specify_idea=args.specify_idea,
            num_ideas=args.num_ideas,
            use_test_time_scaling=args.use_test_time_scaling,
            revision_candidates=args.revision_candidates,
            draft_candidates=args.draft_candidates,
            strict_singletons=args.strict_singletons
        )
        
        print(f"✅ Workflow completed successfully!")
        print(f"📁 Results saved to: {result_dir}")
        
    except KeyboardInterrupt:
        print("\n⚠️ Workflow interrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"❌ Workflow failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
