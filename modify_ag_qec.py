#!/usr/bin/env python3
"""
Direct modification script for ag-qec paper using GPT-5 with max 1 iteration.
"""
import sys
import os
from pathlib import Path

# Add the current directory to Python path so we can import modules
sys.path.insert(0, str(Path(__file__).parent))

def modify_ag_qec_paper():
    """Modify the ag-qec paper using the refactored workflow components."""
    try:
        # Import necessary components
        from core.config import WorkflowConfig, setup_workflow_logging, _prepare_project_dir
        from ai.chat import _universal_chat
        from latex.compiler import _compile_latex_and_get_errors, _generate_pdf_for_review
        from evaluation.quality import _extract_quality_metrics, _calculate_quality_score, _validate_research_quality
        from generation.content import _save_iteration_diff, _extract_paper_metadata
        from workflow_steps.simulation import run_simulation_step
        from workflow_steps.review_revision import run_review_revision_step
        from utils.sim_runner import ensure_single_tex_py, extract_simulation_from_tex
        
        print("🚀 Starting ag-qec paper modification with GPT-5 (max 1 iteration)")
        
        # Set up configuration
        config = WorkflowConfig(
            max_iterations=1,
            quality_threshold=1.0,
            request_timeout=3600,
            max_retries=3,
            enable_pdf_review=True,
            reference_validation=True,
            figure_validation=True,
            diff_output_tracking=True
        )
        
        # Set up logging
        logger = setup_workflow_logging()
        logger.info("Starting ag-qec paper modification")
        
        # Prepare directories
        project_dir = Path("output/ag-qec")
        paper_path = project_dir / "paper.tex"
        sim_path = project_dir / "simulation.py"
        
        if not paper_path.exists():
            print("❌ Error: ag-qec paper.tex not found!")
            return False
            
        print(f"📁 Working with: {project_dir}")
        print(f"🤖 Using model: gpt-5")
        print(f"🔄 Max iterations: 1")
        
        # Read existing paper
        paper_content = paper_path.read_text(encoding="utf-8", errors="ignore")
        print(f"📄 Loaded paper ({len(paper_content)} characters)")
        
        # Extract metadata
        topic, field, question = _extract_paper_metadata(paper_content)
        print(f"📝 Detected - Topic: {topic[:50]}...")
        print(f"🎯 Field: {field}")
        
        # Ensure single tex/py structure  
        ensure_single_tex_py(project_dir)
        extract_simulation_from_tex(paper_path, sim_path)
        
        print("\\n🔄 === Starting Iteration 1 ===")
        
        # Run simulation
        print("🐍 Running simulation...")
        try:
            sim_summary, _ = run_simulation_step(paper_path, sim_path, project_dir, "gpt-5", 3600, "python")
            print(f"✅ Simulation completed")
        except Exception as e:
            print(f"⚠️ Simulation warning: {e}")
            sim_summary = "Simulation step encountered issues but proceeding with review."
        
        # Compile LaTeX
        print("📄 Compiling LaTeX...")
        latex_success, latex_errors = _compile_latex_and_get_errors(paper_path, timeout=180)
        
        if latex_success:
            print("✅ LaTeX compilation successful!")
        else:
            print(f"⚠️ LaTeX compilation issues detected")
            print(f"Last 10 lines of errors:\\n{latex_errors[-1000:]}")
        
        # Quality validation
        print("🔍 Validating research quality...")
        quality_issues = _validate_research_quality(paper_content, sim_summary)
        
        if quality_issues:
            print(f"⚠️ Quality issues detected ({len(quality_issues)} total):")
            for i, issue in enumerate(quality_issues[:5], 1):
                print(f"   {i}. {issue}")
            if len(quality_issues) > 5:
                print(f"   ... and {len(quality_issues) - 5} more issues")
        else:
            print("✅ No quality issues detected")
        
        # Calculate quality score
        current_metrics = _extract_quality_metrics(paper_content, sim_summary)
        quality_score = _calculate_quality_score(current_metrics, quality_issues)
        print(f"📊 Current quality score: {quality_score:.2f}")
        
        # Generate PDF for review if possible
        pdf_path = None
        if latex_success:
            print("📄 Generating PDF for AI review...")
            try:
                pdf_success, generated_pdf_path, pdf_error = _generate_pdf_for_review(paper_path, 180)
                if pdf_success and generated_pdf_path:
                    pdf_path = generated_pdf_path
                    file_size = pdf_path.stat().st_size
                    print(f"✅ PDF generated: {pdf_path.name} ({file_size:,} bytes)")
                else:
                    print(f"⚠️ PDF generation failed: {pdf_error}")
            except Exception as e:
                print(f"⚠️ PDF generation error: {e}")
        
        # Run review and revision with GPT-5
        print("🤖 Running AI review and revision with GPT-5...")
        old_content = paper_content
        
        try:
            review, decision = run_review_revision_step(
                paper_content, sim_summary, latex_errors, project_dir, None,
                1, "gpt-5", 3600, config, pdf_path, True,
                paper_path, quality_issues
            )
            
            print("✅ Review and revision completed!")
            
            # Save iteration diff
            if config.diff_output_tracking:
                new_content = paper_path.read_text(encoding="utf-8", errors="ignore")
                _save_iteration_diff(old_content, new_content, project_dir, 1)
                print("💾 Changes saved to diff file")
            
            # Final quality assessment
            final_content = paper_path.read_text(encoding="utf-8", errors="ignore")
            final_metrics = _extract_quality_metrics(final_content, sim_summary)
            final_quality_issues = _validate_research_quality(final_content, sim_summary)
            final_quality_score = _calculate_quality_score(final_metrics, final_quality_issues)
            
            print(f"\\n🎉 === Modification Complete ===")
            print(f"📊 Final quality score: {final_quality_score:.2f}")
            print(f"📈 Quality improvement: {final_quality_score - quality_score:+.2f}")
            print(f"📁 Results saved to: {project_dir}")
            
            return True
            
        except Exception as e:
            print(f"❌ Review/revision failed: {e}")
            return False
        
    except Exception as e:
        print(f"❌ Script failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = modify_ag_qec_paper()
    if success:
        print("\\n✅ ag-qec paper modification completed successfully!")
    else:
        print("\\n❌ ag-qec paper modification failed")
        sys.exit(1)
