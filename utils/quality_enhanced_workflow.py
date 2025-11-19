"""
Research Quality Integration Module

This module integrates the research robustness validators and enhanced prompts
into the main AI-Scientist workflow, enabling automatic quality assessment
and iterative improvement of generated papers.
"""

import os
import sys
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Add utils directory to path for imports
utils_dir = Path(__file__).parent
if str(utils_dir) not in sys.path:
    sys.path.insert(0, str(utils_dir))

from research_robustness_validators import ComprehensiveResearchValidator, ValidationResult
from enhanced_research_prompts import get_enhanced_paper_prompt, get_revision_prompt

logger = logging.getLogger(__name__)

class QualityEnhancedWorkflow:
    """
    Enhanced AI-Scientist workflow with integrated quality validation and iterative improvement.
    """
    
    def __init__(self, quality_threshold: float = 0.6, max_iterations: int = 3):
        """
        Initialize quality-enhanced workflow.
        
        Args:
            quality_threshold: Minimum score required for each validator (0-1)
            max_iterations: Maximum number of revision iterations
        """
        self.validator = ComprehensiveResearchValidator()
        self.quality_threshold = quality_threshold
        self.max_iterations = max_iterations
        
    def generate_paper_with_quality_assurance(
        self,
        idea: str,
        paper_type: str = "theoretical",
        specific_requirements: str = "",
        output_dir: str = "output",
        model: str = "gpt-4"
    ) -> Tuple[bool, Dict[str, ValidationResult], str]:
        """
        Generate a paper with iterative quality improvement.
        
        Args:
            idea: Research idea description
            paper_type: Type of paper ("theoretical", "computational", "experimental")
            specific_requirements: Additional domain-specific requirements
            output_dir: Output directory for paper files
            model: LLM model to use
            
        Returns:
            Tuple of (success, final_validation_results, paper_path)
        """
        
        logger.info("Starting quality-enhanced paper generation")
        
        # Generate enhanced prompt
        enhanced_prompt = get_enhanced_paper_prompt(paper_type, specific_requirements)
        
        # Create full prompt combining idea and quality requirements
        full_prompt = f"""
Research Idea: {idea}

{enhanced_prompt}

Generate a complete research paper that addresses the above idea while meeting all quality requirements.
The paper should be publication-ready with rigorous theoretical foundations and comprehensive validation.
"""
        
        # Initial paper generation
        paper_text = self._generate_paper_with_model(full_prompt, model)
        if not paper_text:
            logger.error("Failed to generate initial paper")
            return False, {}, ""
        
        # Save initial paper
        paper_path = os.path.join(output_dir, "paper.tex")
        os.makedirs(output_dir, exist_ok=True)
        
        # Iterative improvement loop
        iteration = 0
        while iteration < self.max_iterations:
            iteration += 1
            logger.info(f"Quality validation iteration {iteration}")
            
            # Validate current paper
            validation_results = self.validator.validate_paper(paper_text)
            
            # Check if quality threshold is met
            if self._meets_quality_threshold(validation_results):
                logger.info("Paper meets quality threshold")
                self._save_paper_and_report(paper_text, validation_results, output_dir)
                return True, validation_results, paper_path
            
            # Generate revision prompt
            revision_prompt = get_revision_prompt(validation_results)
            
            # Create revision request
            revision_request = f"""
Here is the current paper:

{paper_text}

{revision_prompt}

Please provide a complete revised paper that addresses all the identified issues.
"""
            
            # Generate revised paper
            logger.info(f"Generating revision (iteration {iteration})")
            revised_paper = self._generate_paper_with_model(revision_request, model)
            
            if not revised_paper:
                logger.warning(f"Failed to generate revision in iteration {iteration}")
                break
                
            paper_text = revised_paper
        
        # Final validation and save
        final_validation = self.validator.validate_paper(paper_text)
        self._save_paper_and_report(paper_text, final_validation, output_dir)
        
        success = self._meets_quality_threshold(final_validation)
        if not success:
            logger.warning("Paper did not meet quality threshold after maximum iterations")
        
        return success, final_validation, paper_path
    
    def validate_existing_paper(self, paper_path: str) -> Tuple[Dict[str, ValidationResult], str]:
        """
        Validate an existing paper and generate improvement report.
        
        Args:
            paper_path: Path to existing paper file
            
        Returns:
            Tuple of (validation_results, report_text)
        """
        
        try:
            with open(paper_path, 'r', encoding='utf-8') as f:
                paper_text = f.read()
        except Exception as e:
            logger.error(f"Error reading paper file {paper_path}: {e}")
            return {}, f"Error reading file: {e}"
        
        # Run validation
        validation_results = self.validator.validate_paper(paper_text)
        
        # Generate report
        report = self.validator.generate_report(validation_results)
        
        # Save validation report
        report_path = paper_path.replace('.tex', '_quality_report.txt')
        try:
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write(report)
            logger.info(f"Quality report saved to {report_path}")
        except Exception as e:
            logger.error(f"Error saving report: {e}")
        
        return validation_results, report
    
    def improve_existing_paper(
        self,
        paper_path: str,
        model: str = "gpt-4",
        max_iterations: int = None
    ) -> Tuple[bool, Dict[str, ValidationResult]]:
        """
        Improve an existing paper through iterative revision.
        
        Args:
            paper_path: Path to existing paper file
            model: LLM model to use for revisions
            max_iterations: Override default max iterations
            
        Returns:
            Tuple of (success, final_validation_results)
        """
        
        if max_iterations is None:
            max_iterations = self.max_iterations
        
        try:
            with open(paper_path, 'r', encoding='utf-8') as f:
                paper_text = f.read()
        except Exception as e:
            logger.error(f"Error reading paper file {paper_path}: {e}")
            return False, {}
        
        # Initial validation
        validation_results = self.validator.validate_paper(paper_text)
        
        if self._meets_quality_threshold(validation_results):
            logger.info("Paper already meets quality threshold")
            return True, validation_results
        
        # Iterative improvement
        iteration = 0
        while iteration < max_iterations:
            iteration += 1
            logger.info(f"Improvement iteration {iteration}")
            
            # Generate revision prompt
            revision_prompt = get_revision_prompt(validation_results)
            
            # Create revision request
            revision_request = f"""
Here is the current paper:

{paper_text}

{revision_prompt}

Please provide a complete revised paper that addresses all the identified issues.
"""
            
            # Generate revised paper
            revised_paper = self._generate_paper_with_model(revision_request, model)
            
            if not revised_paper:
                logger.warning(f"Failed to generate revision in iteration {iteration}")
                break
            
            paper_text = revised_paper
            
            # Validate revision
            validation_results = self.validator.validate_paper(paper_text)
            
            # Check if quality threshold is met
            if self._meets_quality_threshold(validation_results):
                logger.info("Paper meets quality threshold after revision")
                
                # Save improved paper
                improved_path = paper_path.replace('.tex', '_improved.tex')
                try:
                    with open(improved_path, 'w', encoding='utf-8') as f:
                        f.write(paper_text)
                    logger.info(f"Improved paper saved to {improved_path}")
                except Exception as e:
                    logger.error(f"Error saving improved paper: {e}")
                
                # Save validation report
                output_dir = os.path.dirname(paper_path)
                self._save_paper_and_report(paper_text, validation_results, output_dir)
                
                return True, validation_results
        
        # Save final attempt even if not meeting threshold
        final_path = paper_path.replace('.tex', '_final_attempt.tex')
        try:
            with open(final_path, 'w', encoding='utf-8') as f:
                f.write(paper_text)
            logger.info(f"Final attempt saved to {final_path}")
        except Exception as e:
            logger.error(f"Error saving final attempt: {e}")
        
        return False, validation_results
    
    def _generate_paper_with_model(self, prompt: str, model: str) -> Optional[str]:
        """
        Generate paper using specified model (placeholder for actual implementation).
        This should be replaced with actual LLM integration.
        """
        # TODO: Integrate with actual LLM API (OpenAI, Anthropic, etc.)
        # For now, return placeholder
        logger.warning("Using placeholder LLM integration - implement actual model calls")
        return None
    
    def _meets_quality_threshold(self, validation_results: Dict[str, ValidationResult]) -> bool:
        """Check if validation results meet quality threshold"""
        if not validation_results:
            return False
        
        for result in validation_results.values():
            if result.score < self.quality_threshold:
                return False
        
        return True
    
    def _save_paper_and_report(
        self,
        paper_text: str,
        validation_results: Dict[str, ValidationResult],
        output_dir: str
    ):
        """Save paper and validation report"""
        
        # Save paper
        paper_path = os.path.join(output_dir, "paper.tex")
        try:
            with open(paper_path, 'w', encoding='utf-8') as f:
                f.write(paper_text)
            logger.info(f"Paper saved to {paper_path}")
        except Exception as e:
            logger.error(f"Error saving paper: {e}")
        
        # Save validation report
        report = self.validator.generate_report(validation_results)
        report_path = os.path.join(output_dir, "quality_validation_report.txt")
        try:
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write(report)
            logger.info(f"Validation report saved to {report_path}")
        except Exception as e:
            logger.error(f"Error saving validation report: {e}")
        
        # Save detailed results as JSON
        results_data = {
            name: {
                'score': result.score,
                'passed': result.passed,
                'severity': result.severity,
                'issues': result.issues,
                'suggestions': result.suggestions
            }
            for name, result in validation_results.items()
        }
        
        json_path = os.path.join(output_dir, "validation_results.json")
        try:
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(results_data, f, indent=2)
            logger.info(f"Detailed results saved to {json_path}")
        except Exception as e:
            logger.error(f"Error saving JSON results: {e}")


def main():
    """Example usage of quality-enhanced workflow"""
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # Initialize workflow
    workflow = QualityEnhancedWorkflow(quality_threshold=0.6, max_iterations=3)
    
    # Example: Validate existing paper
    paper_path = "../output/sample_paper/paper.tex"
    if os.path.exists(paper_path):
        print("=== VALIDATING EXISTING PAPER ===")
        validation_results, report = workflow.validate_existing_paper(paper_path)
        print(report)
        
        # Check if improvement is needed
        needs_improvement = not workflow._meets_quality_threshold(validation_results)
        if needs_improvement:
            print("\n=== PAPER NEEDS IMPROVEMENT ===")
            print("Consider running improvement process with actual LLM integration")
    else:
        print(f"Paper not found at {paper_path}")
    
    # Example: Generate new paper (requires LLM integration)
    # success, results, path = workflow.generate_paper_with_quality_assurance(
    #     idea="Novel approach to computational optimization using advanced algorithms",
    #     paper_type="theoretical"
    # )

if __name__ == "__main__":
    main()