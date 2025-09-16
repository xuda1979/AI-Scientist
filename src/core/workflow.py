"""
Main workflow orchestrator for SciResearch Workflow.
"""
from __future__ import annotations
import sys
import logging
from typing import Optional
from pathlib import Path

# Import modular components
from core.config import WorkflowConfig, DEFAULT_MODEL
from core.quality import QualityAssessment
from ai.chat import AIChat
from ai.prompts import PromptTemplates
from processing.latex import LaTeXProcessor
from processing.files import FileManager

# Import existing workflow functions from legacy workflow
import sys
from pathlib import Path
# Add the parent directory to the path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Import functions from legacy workflow
from legacy_monolithic_workflow import _initial_draft_prompt, run_simulation_step, run_optimized_review_revision_step

logger = logging.getLogger(__name__)


class SciResearchWorkflow:
    """Main workflow orchestrator."""
    
    def __init__(self, config: Optional[WorkflowConfig] = None):
        self.config = config or WorkflowConfig()
        self.chat = AIChat(DEFAULT_MODEL, self.config.fallback_models)
        self.prompts = PromptTemplates()
        self.latex_processor = LaTeXProcessor()
        self.file_manager = FileManager()
        self.quality_assessor = QualityAssessment(self.config)
    
    def run(
        self,
        topic: str,
        field: str,
        question: str,
        output_dir: Path,
        model: str = DEFAULT_MODEL,
        request_timeout: Optional[int] = 3600,
        max_iterations: int = 4,
        modify_existing: bool = False,
        strict_singletons: bool = True,
        python_exec: Optional[str] = None,
        user_prompt: Optional[str] = None,
        enable_ideation: bool = True,
        num_ideas: int = 15,
        output_diffs: bool = False,
        continuous: bool = False,
    ) -> Path:
        """Run the complete research workflow."""
        
        # Update configuration
        self.config.max_iterations = max_iterations
        self.chat.primary_model = model
        
        logger.info(f"Starting workflow: {topic} in {field}")
        logger.info("Starting SciResearch Workflow")
        logger.info(f"Topic: {topic}")
        logger.info(f"Field: {field}")
        logger.info(f"Model: {model}")
        logger.info(f"Max Iterations: {max_iterations}")
        
        # Setup project
        project_dir = self.file_manager.prepare_project_directory(output_dir, modify_existing)
        paper_path, sim_path = self.file_manager.ensure_single_tex_file(
            project_dir, 
            strict_singletons, 
            preserve_original_filename=modify_existing
        )
        
        logger.info(f"Working with: {paper_path.name}")
        
        # Get custom user prompt if not provided
        if user_prompt is None:
            user_prompt = self._get_user_prompt()
        
        # Check if new paper or existing content
        paper_content = paper_path.read_text(encoding="utf-8").strip()
        is_new_paper = (
            paper_content == "\\documentclass{article}\\begin{document}\\end{document}" or 
            len(paper_content) < 200
        )
        
        # Generate initial content for new papers
        if is_new_paper:
            self._create_initial_paper(
                topic, field, question, user_prompt, paper_path,
                enable_ideation, num_ideas, project_dir
            )
        else:
            logger.info("📖 Using existing paper (%d characters)", len(paper_content))
        
        # Main iteration loop
        return self._run_iteration_loop(
            paper_path, sim_path, project_dir, user_prompt,
            python_exec, output_diffs, request_timeout
        )
    
    def _get_user_prompt(self) -> Optional[str]:
        """Get custom user prompt interactively."""
        if not (sys.stdin and sys.stdin.isatty()):
            return None
        
        prompt_message = "\n".join(
            [
                "",
                "=" * 60,
                "CUSTOM PROMPT INPUT",
                "=" * 60,
                "Provide a custom prompt for AI interactions (optional):",
                "Examples:",
                "  - 'Focus on mathematical rigor and formal proofs'",
                "  - 'Emphasize practical applications'",
                "  - 'Use conversational writing style'",
                "-" * 60,
            ]
        )
        logger.info(prompt_message)
        
        user_input = input("Enter custom prompt (or press Enter to skip): ").strip()
        return user_input if user_input else None
    
    def _create_initial_paper(
        self, 
        topic: str, 
        field: str, 
        question: str,
        user_prompt: Optional[str],
        paper_path: Path,
        enable_ideation: bool,
        num_ideas: int,
        project_dir: Path
    ) -> None:
        """Create initial paper content."""
        logger.info("Creating initial paper...")
        
        if enable_ideation and self.config.enable_optimization:
            # Use optimized combined ideation + draft
            prompt = self.prompts.combined_ideation_draft(
                topic, field, question, num_ideas, user_prompt
            )
            
            response = self.chat.chat(
                [{"role": "user", "content": prompt}],
                prompt_type="combined_ideation_draft"
            )
            
            # Parse combined response
            draft = self._parse_combined_ideation_response(response, project_dir)
            
        else:
            # Use traditional separate approach
            draft = generate_initial_draft(
                topic, field, question, user_prompt,
                self.chat.primary_model, 3600, self.config
            )
        
        paper_path.write_text(draft, encoding="utf-8")
        logger.info("Initial paper created")
    
    def _parse_combined_ideation_response(self, response: str, project_dir: Path) -> str:
        """Parse combined ideation and draft response."""
        import re
        
        # Save the full response first for debugging
        ideation_file = project_dir / "ideation_analysis.txt"
        with open(ideation_file, 'w', encoding='utf-8') as f:
            f.write("COMBINED IDEATION AND DRAFT ANALYSIS\n")
            f.write("=" * 50 + "\n\n")
            f.write(response)
        
        logger.info(f"Ideation analysis saved to: {ideation_file.name}")
        
        # Try multiple patterns to extract LaTeX paper
        patterns = [
            r'##\s*COMPLETE\s+LATEX\s+PAPER\s*\n```(?:latex|tex)?\s*\n(.*?)\n```',
            r'```(?:latex|tex)\s*\n(.*?)\n```',
            r'\\documentclass.*?\\end\{document\}',
            r'\\begin\{document\}.*?\\end\{document\}'
        ]
        
        for pattern in patterns:
            draft_match = re.search(pattern, response, re.DOTALL | re.IGNORECASE)
            if draft_match:
                draft = draft_match.group(1).strip() if draft_match.groups() else draft_match.group(0).strip()
                logger.info(f"Successfully extracted LaTeX using pattern: {pattern[:50]}...")
                return draft
        
        # If no LaTeX found, try to extract anything that looks like a complete document
        if '\\documentclass' in response:
            # Extract everything from \documentclass to \end{document}
            start_idx = response.find('\\documentclass')
            end_idx = response.rfind('\\end{document}')
            if start_idx != -1 and end_idx != -1:
                draft = response[start_idx:end_idx + len('\\end{document}')].strip()
                logger.info("Extracted LaTeX by searching for document boundaries")
                return draft
        
        # If still no LaTeX found, create a basic template with the research info
        logger.warning("Could not extract LaTeX from response, creating basic template")
        return self._create_fallback_latex_template(response)
    
    def _create_fallback_latex_template(self, response: str) -> str:
        """Create a fallback LaTeX template when parsing fails."""
        import re
        
        # Extract title if possible
        title_match = re.search(r'\*\*Title\*\*:?\s*(.+)', response, re.IGNORECASE)
        title = title_match.group(1).strip() if title_match else "Research Paper"
        
        return f"""\\documentclass{{article}}
\\usepackage{{amsmath}}
\\usepackage{{amsthm}}
\\usepackage{{amssymb}}

\\begin{{document}}
\\title{{{title}}}
\\author{{AI Research}}
\\maketitle

\\begin{{abstract}}
This paper presents research findings on neural network optimization and training efficiency improvements.
\\end{{abstract}}

\\section{{Introduction}}
Neural network training efficiency is a critical area of research in artificial intelligence.

\\section{{Methodology}}
Our approach focuses on mathematical rigor and clear reasoning in optimization techniques.

\\section{{Results}}
Preliminary results show promising improvements in training efficiency.

\\section{{Conclusion}}
This work contributes to the understanding of neural network optimization strategies.

\\end{{document}}"""
    
    def _run_iteration_loop(
        self,
        paper_path: Path,
        sim_path: Path, 
        project_dir: Path,
        user_prompt: Optional[str],
        python_exec: Optional[str],
        output_diffs: bool,
        request_timeout: int
    ) -> Path:
        """Run the main iteration loop."""
        
        logger.info("🔄 Starting %d iteration(s)", self.config.max_iterations)

        for i in range(1, self.config.max_iterations + 1):
            logger.info("\n--- Iteration %d/%d ---", i, self.config.max_iterations)
            
            # Run simulation if code changed
            sim_summary = self._run_simulation_if_needed(
                paper_path, sim_path, project_dir, python_exec, request_timeout
            )
            
            # Compile LaTeX if needed
            latex_success, latex_errors = self._compile_latex_if_needed(paper_path)
            
            # Quality assessment
            current_tex = paper_path.read_text(encoding="utf-8", errors="ignore")
            quality_result = self.quality_assessor.assess_paper_quality(
                current_tex, sim_summary, latex_errors
            )
            
            self._print_iteration_summary(i, quality_result, latex_success)
            
            # Review and revision
            self._run_review_revision(
                paper_path, current_tex, sim_summary, latex_errors,
                quality_result["issues"], user_prompt, i, project_dir,
                output_diffs, request_timeout
            )
        
        # Final summary
        self._print_final_summary()
        return project_dir
    
    def _run_simulation_if_needed(
        self, 
        paper_path: Path, 
        sim_path: Path, 
        project_dir: Path,
        python_exec: Optional[str],
        request_timeout: int
    ) -> str:
        """Run simulation only if code changed."""
        if self.file_manager.check_file_changed(sim_path):
            logger.info("Running simulation (code changed)...")
            sim_summary, _ = run_simulation_step(
                paper_path, sim_path, project_dir,
                self.chat.primary_model, request_timeout, python_exec
            )
            return sim_summary
        else:
            logger.info("Reusing simulation results (code unchanged)")
            return getattr(self, '_last_sim_summary', "No simulation results")
    
    def _compile_latex_if_needed(self, paper_path: Path) -> tuple:
        """Compile LaTeX only if content changed."""
        timeout = self.latex_processor.calculate_dynamic_timeout(
            paper_path.read_text(encoding="utf-8", errors="ignore")
        )
        
        success, errors = self.latex_processor.compile_and_validate(paper_path, timeout)
        
        if success:
            logger.info("LaTeX compilation successful")
        else:
            logger.error("LaTeX compilation failed")
        
        return success, errors
    
    def _print_iteration_summary(self, iteration: int, quality_result: dict, latex_success: bool):
        """Print iteration summary."""
        logger.info(f"Iteration {iteration} Summary:")
        logger.info(f"Quality Score: {quality_result['quality_score']:.3f}")
        logger.info(f"LaTeX: {'Success' if latex_success else 'Failed'}")
        logger.info("Issues: %d", len(quality_result['issues']))
        if quality_result['issues']:
            logger.info("   Top Issues: %s", ", ".join(quality_result['issues'][:3]))
    
    def _run_review_revision(
        self,
        paper_path: Path,
        current_tex: str,
        sim_summary: str, 
        latex_errors: str,
        quality_issues: list,
        user_prompt: Optional[str],
        iteration: int,
        project_dir: Path,
        output_diffs: bool,
        request_timeout: int
    ) -> None:
        """Run optimized review and revision."""
        logger.info("🔍 Running comprehensive review & revision...")
        
        # Create comprehensive prompt
        prompt = self.prompts.comprehensive_review_revision(
            current_tex, sim_summary, latex_errors, quality_issues,
            user_prompt, iteration, self.config.max_iterations
        )
        
        # Run optimized review + revision
        review, decision = run_optimized_review_revision_step(
            prompt, project_dir, user_prompt, iteration,
            self.chat.primary_model, request_timeout, self.config,
            None, output_diffs, paper_path
        )
        
        logger.info("Review & revision completed")
    
    def _print_final_summary(self) -> None:
        """Print final workflow summary."""
        progress = self.quality_assessor.get_progress_summary()
        chat_stats = self.chat.get_stats()
        
        logger.info("\nWorkflow Complete!")
        logger.info(f"Quality Progression: {[f'{q:.2f}' for q in progress['quality_history']]}")
        logger.info(f"Best Score: {progress['best_score']:.3f}")
        logger.info(f"API Calls: {chat_stats['calls']} (Success: {chat_stats['success_rate']:.1%})")
        logger.info(f"Trend: {progress['improvement_trend']}")


def run_workflow(*args, **kwargs) -> Path:
    """Convenience function to run workflow."""
    workflow = SciResearchWorkflow()
    return workflow.run(*args, **kwargs)
