#!/usr/bin/env python3
"""
Quality validation module for automated paper quality assessment.
"""

import re
import json
from typing import List, Dict, Optional, Tuple, Any
from pathlib import Path
import logging

from document_types import DocumentTemplate, DocumentType, get_document_template
from .novelty_vetting import NoveltyVetter

# Import review-driven enhancements
try:
    from prompts.review_driven_enhancements import detect_review_issues
    REVIEW_ENHANCEMENTS_AVAILABLE = True
except ImportError:
    REVIEW_ENHANCEMENTS_AVAILABLE = False
    print("Warning: Review-driven enhancements not available")

# Import fair scoring system
try:
    from scoring.fair_scoring_system import FairPaperScorer
    FAIR_SCORING_AVAILABLE = True
except ImportError:
    FAIR_SCORING_AVAILABLE = False
    print("Warning: Fair scoring system not available")

logger = logging.getLogger(__name__)

class PaperQualityValidator:
    """Automated validator for paper quality and experimental rigor."""
    
    def __init__(self, config: Optional[Dict] = None):
        """Initialize validator with configuration."""
        self.config = config or self._default_config()
        self._active_template: Optional[DocumentTemplate] = None
        self._active_doc_type: Optional[DocumentType] = None
        self._active_field: str = ""
        self._novelty_vetter: Optional[NoveltyVetter] = None
        
    def _default_config(self) -> Dict:
        """Default validation configuration."""
        return {
            "experimental_design": {
                "min_random_seeds": 5,
                "statistical_tests": ["t-test", "wilcoxon", "confidence_intervals"],
                "confidence_level": 0.95,
                "min_baselines": 3,
                "effect_size_reporting": True,
                "multiple_testing_correction": True
            },
            "evaluation_metrics": {
                "primary_metrics": ["accuracy", "precision", "recall", "f1", "auc"],
                "secondary_metrics": ["training_time", "inference_time", "memory_usage"],
                "statistical_measures": ["mean", "std", "confidence_intervals", "p_values"],
                "min_evaluation_samples": 1000
            },
            "methodology_requirements": {
                "algorithm_pseudocode": True,
                "hyperparameter_documentation": True,
                "reproducibility_info": True,
                "preprocessing_details": True,
                "architecture_specification": True
            },
            "literature_review": {
                "min_recent_refs": 10,
                "recent_years": 3,
                "min_total_refs": 15,
                "comparison_analysis": True
            },
            "novelty": {
                "global_threshold": 0.55,
                "domain_thresholds": {
                    "machine learning": 0.60,
                    "computer vision": 0.60,
                    "natural language processing": 0.60,
                    "robotics": 0.55,
                    "security": 0.50,
                    "theory": 0.45,
                },
                "fair_score_threshold": 0.55,
            },
            "poker_specific": {
                "required_metrics": ["exploitability", "nash_convergence", "win_rate"],
                "opponent_types": 3,
                "game_formats": 2,
                "min_evaluation_hands": 50000
            }
        }
    
    def validate_paper(self, paper_content: str, simulation_output: str = "", 
                      metadata: Optional[Dict] = None) -> Tuple[List[str], Dict[str, float]]:
        """
        Comprehensive paper validation.
        
        Returns:
            Tuple of (issues_list, quality_scores)
        """
        issues = []
        quality_scores = {}

        doc_type: Optional[DocumentType] = None
        field = ""
        if metadata:
            doc_type_value = metadata.get("doc_type") or metadata.get("document_type")
            field = metadata.get("field", "")
            if isinstance(doc_type_value, DocumentType):
                doc_type = doc_type_value
            elif isinstance(doc_type_value, str):
                normalized = doc_type_value.lower()
                for candidate in DocumentType:
                    if normalized in (candidate.value, candidate.name.lower()):
                        doc_type = candidate
                        break

        template = get_document_template(doc_type) if doc_type else None

        previous_template = self._active_template
        previous_doc_type = self._active_doc_type
        previous_field = self._active_field

        self._active_template = template
        self._active_doc_type = doc_type
        self._active_field = field

        try:
            # Statistical rigor validation
            stat_issues, stat_score = self._validate_statistical_rigor(paper_content, simulation_output)
            issues.extend(stat_issues)
            quality_scores['statistical_rigor'] = stat_score

            # Experimental design validation
            exp_issues, exp_score = self._validate_experimental_design(paper_content, simulation_output)
            issues.extend(exp_issues)
            quality_scores['experimental_design'] = exp_score

            # Methodology clarity validation
            method_issues, method_score = self._validate_methodology_clarity(paper_content)
            issues.extend(method_issues)
            quality_scores['methodology_clarity'] = method_score

            # Literature review validation
            lit_issues, lit_score = self._validate_literature_review(paper_content)
            issues.extend(lit_issues)
            quality_scores['literature_review'] = lit_score

            # Innovation articulation validation
            innovation_issues, innovation_score = self._validate_innovation_focus(paper_content)
            issues.extend(innovation_issues)
            quality_scores['innovation_focus'] = innovation_score

            # Results presentation validation
            results_issues, results_score = self._validate_results_presentation(paper_content)
            issues.extend(results_issues)
            quality_scores['results_presentation'] = results_score

            # Domain-specific validation (poker AI if detected)
            if self._is_poker_related(paper_content):
                poker_issues, poker_score = self._validate_poker_specific(paper_content, simulation_output)
                issues.extend(poker_issues)
                quality_scores['poker_specific'] = poker_score

            # LaTeX and formatting validation
            format_issues, format_score = self._validate_formatting(paper_content)
            issues.extend(format_issues)
            quality_scores['formatting'] = format_score

            # Novelty validation using retrieval-backed analysis
            novelty_issues, novelty_score = self._validate_novelty(
                paper_content,
                field or self._active_field,
            )
            issues.extend(novelty_issues)
            quality_scores['novelty'] = novelty_score

            # Review-driven issue detection (based on actual reviewer feedback)
            if REVIEW_ENHANCEMENTS_AVAILABLE:
                review_issues = detect_review_issues(
                    paper_content,
                    doc_type=self._active_doc_type,
                    field=self._active_field,
                )
                issues.extend(review_issues)
                # Calculate review quality score based on critical issues
                critical_count = sum(1 for issue in review_issues if "CRITICAL" in issue)
                warning_count = sum(1 for issue in review_issues if "WARNING" in issue)
                review_score = max(0.0, 1.0 - (critical_count * 0.3) - (warning_count * 0.1))
                quality_scores['review_readiness'] = review_score

            # Hard acceptance gate using FairPaperScorer when available
            fair_issues, fair_score, fair_metrics = self._enforce_fair_acceptance(
                paper_content,
                simulation_output,
                metadata,
            )
            if fair_issues:
                issues.extend(fair_issues)
            if fair_score is not None:
                quality_scores['fair_final'] = fair_score
            if fair_metrics is not None:
                quality_scores['fair_penalty_adjusted'] = fair_metrics.penalty_adjusted_score

            # Overall quality score
            quality_scores['overall'] = sum(quality_scores.values()) / len(quality_scores)

            return issues, quality_scores
        finally:
            self._active_template = previous_template
            self._active_doc_type = previous_doc_type
            self._active_field = previous_field
    
    def get_fair_quality_assessment(
        self,
        paper_content: str,
        simulation_output: str = "",
        metadata: Optional[Dict] = None,
    ) -> Tuple[float, Dict, List[str], str]:
        """
        Get comprehensive quality assessment using the fair scoring system.
        
        Returns:
            Tuple of (final_score, detailed_metrics, issues, report)
        """
        if not FAIR_SCORING_AVAILABLE:
            # Fallback to original validation
            issues, scores = self.validate_paper(paper_content, simulation_output)
            overall_score = scores.get('overall', 0.5)
            return overall_score, scores, issues, "Fair scoring system not available"
        
        # Use fair scoring system
        fair_scorer = FairPaperScorer()
        metrics, issues = fair_scorer.score_paper(paper_content, simulation_output, metadata=metadata)
        
        # Generate detailed report
        report = fair_scorer.generate_detailed_report(metrics, issues)
        
        # Convert metrics to dictionary format
        detailed_metrics = metrics.to_dict()
        
        return metrics.final_score, detailed_metrics, issues, report
    
    def _validate_statistical_rigor(self, content: str, simulation: str) -> Tuple[List[str], float]:
        """Validate statistical rigor and significance testing."""
        issues = []
        score = 1.0

        template = self._active_template
        if template and not template.requires_simulation:
            return self._validate_theoretical_rigor(content)

        # First, determine if this paper involves numerical experiments
        experimental_paper = self._is_experimental_paper(content, simulation)

        if not experimental_paper:
            # For theoretical/survey papers, different validation criteria
            return self._validate_theoretical_rigor(content)
        
        # Check for p-value reporting (only for experimental papers)
        p_value_patterns = [r'p\s*[<>=]\s*0\.\d+', r'p-value', r'statistically significant']
        if not any(re.search(pattern, content, re.IGNORECASE) for pattern in p_value_patterns):
            issues.append("Missing statistical significance testing and p-value reporting")
            score -= 0.2
        
        # Check for confidence intervals
        ci_patterns = [r'confidence interval', r'\d+%\s*ci', r'95%\s*confidence']
        if not any(re.search(pattern, content, re.IGNORECASE) for pattern in ci_patterns):
            issues.append("Missing confidence interval reporting")
            score -= 0.15
        
        # Check for multiple runs/seeds
        seed_patterns = [r'random seed', r'multiple runs', r'\d+\s*seeds?', r'independent runs']
        if not any(re.search(pattern, content, re.IGNORECASE) for pattern in seed_patterns):
            issues.append("No evidence of multiple experimental runs with different seeds")
            score -= 0.2
        
        # Check for effect size reporting
        effect_patterns = [r'effect size', r'cohen\'?s d', r'eta squared', r'practical significance']
        if not any(re.search(pattern, content, re.IGNORECASE) for pattern in effect_patterns):
            issues.append("Missing effect size reporting - statistical significance without practical significance")
            score -= 0.1
        
        # Check for multiple testing correction
        correction_patterns = [r'bonferroni', r'fdr', r'multiple.*correction', r'family.*wise']
        if not any(re.search(pattern, content, re.IGNORECASE) for pattern in correction_patterns):
            issues.append("Missing multiple testing correction when comparing multiple methods/conditions")
            score -= 0.1
        
        # Check simulation for proper statistical analysis
        if simulation:
            if 'np.random.seed' not in simulation and 'random.seed' not in simulation:
                issues.append("Simulation code lacks proper random seed control")
                score -= 0.1
            
            if 'scipy.stats' not in simulation and 't-test' not in simulation.lower():
                issues.append("Simulation lacks statistical testing implementation")
                score -= 0.1
        
        return issues, max(0.0, score)
    
    def _is_experimental_paper(self, content: str, simulation: str) -> bool:
        """Determine if paper involves numerical experiments."""
        # Keywords that suggest experimental work
        experimental_keywords = [
            'experiment', 'evaluation', 'performance', 'accuracy', 'baseline',
            'dataset', 'test', 'validation', 'benchmark', 'empirical',
            'simulation', 'results', 'metrics', 'comparison'
        ]
        
        # Keywords that suggest theoretical work
        theoretical_keywords = [
            'theoretical', 'proof', 'theorem', 'lemma', 'proposition',
            'survey', 'review', 'analysis', 'framework', 'formalism',
            'position paper', 'perspective', 'opinion', 'discussion'
        ]

        template = self._active_template
        if template:
            if not template.requires_simulation:
                return False
            # Bias towards experimental if template explicitly requires it
            if template.requires_simulation:
                experimental_keywords.extend(['simulation', 'benchmark', 'empirical'])

        content_lower = content.lower()

        # Count experimental vs theoretical indicators
        exp_count = sum(1 for keyword in experimental_keywords if keyword in content_lower)
        theo_count = sum(1 for keyword in theoretical_keywords if keyword in content_lower)
        
        # If has simulation code, likely experimental
        if simulation and len(simulation.strip()) > 100:
            exp_count += 2
        
        # If paper has results/evaluation section, likely experimental
        if re.search(r'\\section\{.*results.*\}|\\section\{.*evaluation.*\}', content, re.IGNORECASE):
            exp_count += 2
            
        # If paper has methodology but no experiments section, might be theoretical
        if (re.search(r'\\section\{.*method.*\}', content, re.IGNORECASE) and
            not re.search(r'\\section\{.*experiment.*\}|\\section\{.*evaluation.*\}', content, re.IGNORECASE)):
            theo_count += 1
        
        return exp_count > theo_count
    
    def _validate_theoretical_rigor(self, content: str) -> Tuple[List[str], float]:
        """Validate rigor for theoretical papers."""
        issues = []
        score = 1.0
        
        # Check for proper mathematical formalization
        math_patterns = [r'\\begin\{theorem\}', r'\\begin\{lemma\}', r'\\begin\{proposition\}', 
                        r'\\begin\{definition\}', r'\\begin\{proof\}']
        if not any(re.search(pattern, content, re.IGNORECASE) for pattern in math_patterns):
            # Only flag if paper claims to be theoretical
            if any(keyword in content.lower() for keyword in ['theorem', 'proof', 'theoretical']):
                issues.append("Theoretical claims lack proper mathematical formalization")
                score -= 0.1
        
        # Check for literature analysis (important for surveys/reviews)
        analysis_patterns = [r'comprehensive', r'systematic', r'comparative analysis', r'taxonomy']
        if not any(re.search(pattern, content, re.IGNORECASE) for pattern in analysis_patterns):
            if any(keyword in content.lower() for keyword in ['survey', 'review', 'overview']):
                issues.append("Survey/review paper lacks systematic analysis framework")
                score -= 0.1
        
        return issues, score
    
    def _validate_experimental_design(self, content: str, simulation: str) -> Tuple[List[str], float]:
        """Validate experimental design quality."""
        issues = []
        score = 1.0
        
        # Check if this is an experimental paper
        if not self._is_experimental_paper(content, simulation):
            # For non-experimental papers, validate theoretical design
            return self._validate_theoretical_design(content)
        
        # Check for baseline comparisons (only for experimental papers)
        baseline_patterns = [r'baseline', r'state.of.the.art', r'compared? to', r'versus', r'outperform']
        baseline_count = sum(1 for pattern in baseline_patterns 
                           if re.search(pattern, content, re.IGNORECASE))
        
        if baseline_count < 2:
            issues.append("Insufficient baseline comparison discussion - need at least 3 relevant baselines")
            score -= 0.25
        
        # Check for cross-validation or proper train/test splits
        validation_patterns = [r'cross.validation', r'k.fold', r'train.*test.*split', r'holdout']
        if not any(re.search(pattern, content, re.IGNORECASE) for pattern in validation_patterns):
            issues.append("Missing proper validation methodology (cross-validation or train/test splits)")
            score -= 0.2
        
        # Check for hyperparameter tuning discussion
        hyperparam_patterns = [r'hyperparameter', r'grid search', r'bayesian optimization', r'parameter tuning']
        if not any(re.search(pattern, content, re.IGNORECASE) for pattern in hyperparam_patterns):
            issues.append("Missing hyperparameter tuning methodology and fairness discussion")
            score -= 0.15
        
        # Check for computational complexity analysis
        complexity_patterns = [r'computational complexity', r'time complexity', r'space complexity', r'big o']
        if not any(re.search(pattern, content, re.IGNORECASE) for pattern in complexity_patterns):
            issues.append("Missing computational complexity analysis")
            score -= 0.1
        
        # Check for ablation studies
        ablation_patterns = [r'ablation', r'component analysis', r'contribution.*component']
        if not any(re.search(pattern, content, re.IGNORECASE) for pattern in ablation_patterns):
            issues.append("Missing ablation studies to analyze component contributions")
            score -= 0.1
        
        return issues, max(0.0, score)
    
    def _validate_theoretical_design(self, content: str) -> Tuple[List[str], float]:
        """Validate design for theoretical papers."""
        issues = []
        score = 1.0
        
        # Check for logical structure and argumentation
        structure_patterns = [r'framework', r'approach', r'methodology', r'systematic']
        if not any(re.search(pattern, content, re.IGNORECASE) for pattern in structure_patterns):
            issues.append("Theoretical paper lacks clear methodological framework")
            score -= 0.1
        
        # Check for related work comparison (still important for theoretical papers)
        comparison_patterns = [r'compared? to', r'versus', r'unlike', r'differs? from']
        if not any(re.search(pattern, content, re.IGNORECASE) for pattern in comparison_patterns):
            issues.append("Missing comparison with existing theoretical approaches")
            score -= 0.15
        
        # Check for contribution clarity
        contribution_patterns = [r'contribution', r'novel', r'new', r'propose', r'introduce']
        if not any(re.search(pattern, content, re.IGNORECASE) for pattern in contribution_patterns):
            issues.append("Theoretical contributions not clearly articulated")
            score -= 0.1
        
        return issues, score
    
    def _validate_methodology_clarity(self, content: str) -> Tuple[List[str], float]:
        """Validate methodology documentation clarity."""
        issues = []
        score = 1.0

        template = self._active_template
        requires_algorithms = True if template is None else template.requires_algorithms

        # Check for algorithm pseudocode
        if requires_algorithms:
            if '\\begin{algorithm}' not in content and '\\begin{algorithmic}' not in content:
                issues.append("Missing algorithmic pseudocode for novel methods")
                score -= 0.2

        # Check for detailed architecture description
        arch_patterns = [r'architecture', r'layer.*dimension', r'hidden.*unit', r'network.*structure']
        if not any(re.search(pattern, content, re.IGNORECASE) for pattern in arch_patterns):
            issues.append("Missing detailed model architecture specification")
            score -= 0.15
        
        # Check for preprocessing details
        preproc_patterns = [r'preprocessing', r'normalization', r'feature.*engineering', r'data.*augmentation']
        if not any(re.search(pattern, content, re.IGNORECASE) for pattern in preproc_patterns):
            issues.append("Missing preprocessing and data preparation details")
            score -= 0.1
        
        # Check for reproducibility information
        reprod_patterns = [r'reproducib', r'implementation.*detail', r'code.*available', r'open.*source']
        if not any(re.search(pattern, content, re.IGNORECASE) for pattern in reprod_patterns):
            issues.append("Missing reproducibility information and implementation details")
            score -= 0.15
        
        # Check for training procedure details
        training_patterns = [r'learning rate', r'batch size', r'optimization', r'convergence.*criteria']
        if not any(re.search(pattern, content, re.IGNORECASE) for pattern in training_patterns):
            issues.append("Missing detailed training procedure specification")
            score -= 0.1
        
        return issues, max(0.0, score)
    
    def _validate_literature_review(self, content: str) -> Tuple[List[str], float]:
        """Validate literature review comprehensiveness."""
        issues = []
        score = 1.0

        template = self._active_template
        min_required_refs = self.config['literature_review']['min_total_refs']
        if template:
            if not template.requires_embedded_references:
                issues.append(
                    f"INFO: Extensive bibliography requirements relaxed for {template.doc_type.value} template"
                )
                return issues, score
            min_required_refs = max(template.min_references, min_required_refs)

        # Count references
        ref_matches = re.findall(r'\\cite\{[^}]+\}', content)
        ref_count = len(set(ref_matches))  # Unique citations

        if ref_count < min_required_refs:
            issues.append(f"Insufficient references: {ref_count} < {min_required_refs}")
            score -= 0.2

        # Check for recent work citation
        recent_patterns = [r'20(1[9-9]|2[0-4])', r'recent', r'state.of.the.art']
        if not any(re.search(pattern, content, re.IGNORECASE) for pattern in recent_patterns):
            issues.append("Missing recent work from the last 2-3 years")
            score -= 0.15

        # Check for related work integration
        if template is None or template.enforce_strict_sections:
            if '\\section{Related Work}' not in content and '\\section{Literature Review}' not in content:
                issues.append("Missing dedicated related work or literature review section")
                score -= 0.1
        else:
            issues.append(
                f"INFO: Dedicated related work section requirement skipped for {template.doc_type.value} template"
            )

        # Check for limitation discussion of existing work
        limitation_patterns = [r'limitation', r'shortcoming', r'drawback', r'problem.*existing']
        if not any(re.search(pattern, content, re.IGNORECASE) for pattern in limitation_patterns):
            issues.append("Missing discussion of limitations in existing approaches")
            score -= 0.1
        
        return issues, max(0.0, score)
    
    def _validate_innovation_focus(self, content: str) -> Tuple[List[str], float]:
        """Ensure the manuscript foregrounds innovation and differentiation."""
        issues: List[str] = []
        score = 0.0

        template = self._active_template
        if template:
            innovation_doc_types = {
                DocumentType.RESEARCH_PAPER,
                DocumentType.CONFERENCE_PAPER,
                DocumentType.JOURNAL_ARTICLE,
                DocumentType.ENGINEERING_PAPER,
                DocumentType.TECHNICAL_REPORT,
                DocumentType.WHITE_PAPER,
            }
            if template.doc_type not in innovation_doc_types:
                return issues, 1.0

        lowered = content.lower()

        # Dedicated contributions section is critical
        contributions_match = re.search(r'\\section\\*?\{[^}]*contribution[^}]*\}', content, re.IGNORECASE)
        if not contributions_match:
            contributions_match = re.search(r'\\subsection\\*?\{[^}]*contribution[^}]*\}', content, re.IGNORECASE)

        if contributions_match:
            score += 0.35
            section_body = content[contributions_match.end():]
            next_section_match = re.search(r'\\section\\*?\{', section_body)
            if next_section_match:
                section_body = section_body[:next_section_match.start()]

            has_bullets = bool(
                re.search(r'\\begin\{itemize\}.*?\\end\{itemize\}', section_body, re.DOTALL)
                or re.search(r'\\begin\{enumerate\}.*?\\end\{enumerate\}', section_body, re.DOTALL)
            )
            if has_bullets:
                score += 0.2
            else:
                issues.append(
                    "WARNING: Contributions section should enumerate innovations with bullet or numbered lists for clarity."
                )
        else:
            issues.append(
                "CRITICAL: Missing a dedicated Contributions section that summarises the paper's novel innovations."
            )

        # Prior art differentiation subsection
        prior_art_match = re.search(
            r'\\(?:sub)?section\\*?\{[^}]*(prior\\s*art|differentiation|novelty analysis)[^}]*\}',
            content,
            re.IGNORECASE,
        )
        if prior_art_match:
            score += 0.2
        else:
            issues.append(
                "WARNING: Add a labelled prior art differentiation subsection contrasting against closest existing work."
            )

        # Measure explicit innovation language
        novelty_terms = re.findall(
            r'\b(novel|innovation|innovative|original|differentiation|breakthrough)\b',
            lowered,
        )
        if len(novelty_terms) >= 6:
            score += 0.15
        elif len(novelty_terms) >= 3:
            score += 0.1
        else:
            issues.append(
                "WARNING: Manuscript minimally references novelty; strengthen innovation framing across sections."
            )

        # Innovation hooks or validation checklists
        hook_patterns = [
            r'innovation\s+hooks?',
            r'innovation\s+checklist',
            r'differentiation\s+matrix',
            r'novelty\s+validation',
        ]
        if any(re.search(pattern, lowered) for pattern in hook_patterns):
            score += 0.1
        else:
            issues.append(
                "INFO: Document does not reference an innovation hooks checklist tying novel elements to evaluation evidence."
            )

        # Forward-looking vision and impact
        vision_patterns = [r'broader impact', r'future work', r'vision', r'roadmap']
        if any(re.search(pattern, lowered) for pattern in vision_patterns):
            score += 0.1
        else:
            issues.append(
                "INFO: Consider adding a forward-looking impact or vision statement to contextualise the innovations."
            )

        return issues, max(0.0, min(1.0, score))

    def _validate_results_presentation(self, content: str) -> Tuple[List[str], float]:
        """Validate results presentation quality."""
        issues = []
        score = 1.0

        template = self._active_template
        requires_figures = True
        requires_tables = True
        if template:
            requires_figures = template.requires_figures
            requires_tables = template.requires_tables

        if not requires_figures and not requires_tables:
            issues.append(
                f"INFO: Visual results requirements relaxed for {template.doc_type.value} template"
                if template
                else "INFO: Visual results requirements relaxed"
            )
            return issues, score

        # Check for figures
        if requires_figures:
            figure_count = len(re.findall(r'\\begin{figure}', content))
            if figure_count < 2:
                issues.append("Insufficient figures for results presentation - need at least 2-3 figures")
                score -= 0.15

        # Check for tables
        if requires_tables:
            table_count = len(re.findall(r'\\begin{table}', content))
            if table_count < 1:
                issues.append("Missing tables for numerical results presentation")
                score -= 0.1

        # Check for error bars or uncertainty quantification
        error_patterns = [r'error.*bar', r'standard.*deviation', r'\\pm', r'uncertainty', r'variance']
        if not any(re.search(pattern, content, re.IGNORECASE) for pattern in error_patterns):
            issues.append("Missing error bars or uncertainty quantification in results")
            score -= 0.15
        
        # Check for statistical comparison
        comparison_patterns = [r'statistically.*significant', r'p.*value', r'significance.*test']
        if not any(re.search(pattern, content, re.IGNORECASE) for pattern in comparison_patterns):
            issues.append("Missing statistical significance testing for method comparisons")
            score -= 0.1
        
        # Check for failure case analysis
        failure_patterns = [r'failure.*case', r'limitation', r'error.*analysis', r'worst.*case']
        if not any(re.search(pattern, content, re.IGNORECASE) for pattern in failure_patterns):
            issues.append("Missing failure case analysis and method limitations")
            score -= 0.1
        
        return issues, max(0.0, score)
    
    def _is_poker_related(self, content: str) -> bool:
        """Check if paper is poker/game theory related."""
        poker_keywords = ['poker', 'texas hold', 'cfr', 'game theory', 'nash equilibrium', 
                         'regret minimization', 'exploitability']
        return any(keyword in content.lower() for keyword in poker_keywords)
    
    def _validate_poker_specific(self, content: str, simulation: str) -> Tuple[List[str], float]:
        """Validate poker-specific experimental requirements."""
        issues = []
        score = 1.0
        
        # Check for exploitability analysis
        if 'exploitability' not in content.lower():
            issues.append("Missing exploitability analysis for poker AI evaluation")
            score -= 0.2
        
        # Check for Nash equilibrium discussion
        nash_patterns = [r'nash.*equilibrium', r'nash.*convergence', r'game.*theoretic.*optimal']
        if not any(re.search(pattern, content, re.IGNORECASE) for pattern in nash_patterns):
            issues.append("Missing Nash equilibrium analysis for game-theoretic optimality")
            score -= 0.15
        
        # Check for diverse opponent evaluation
        opponent_patterns = [r'opponent.*type', r'playing.*style', r'tight.*aggressive', r'loose.*passive']
        if not any(re.search(pattern, content, re.IGNORECASE) for pattern in opponent_patterns):
            issues.append("Missing evaluation against diverse opponent types and playing styles")
            score -= 0.1
        
        # Check for sufficient hand evaluation
        if simulation and 'hands' in simulation.lower():
            hand_numbers = re.findall(r'(\d+).*hands?', simulation.lower())
            if hand_numbers:
                max_hands = max(int(num) for num in hand_numbers)
                if max_hands < 50000:
                    issues.append(f"Insufficient evaluation hands: {max_hands} < 50,000")
                    score -= 0.1
        
        # Check for betting pattern analysis
        betting_patterns = [r'betting.*pattern', r'bluff.*frequenc', r'aggression.*factor']
        if not any(re.search(pattern, content, re.IGNORECASE) for pattern in betting_patterns):
            issues.append("Missing betting pattern and strategic behavior analysis")
            score -= 0.1
        
        return issues, max(0.0, score)
    
    def _validate_formatting(self, content: str) -> Tuple[List[str], float]:
        """Validate LaTeX formatting and structure."""
        issues = []
        score = 1.0

        template = self._active_template
        enforce_sections = True if template is None else template.enforce_strict_sections

        # Check for proper document structure
        if '\\documentclass' not in content:
            issues.append("Missing proper LaTeX document class declaration")
            score -= 0.1

        # Check for abstract
        if enforce_sections and '\\begin{abstract}' not in content:
            issues.append("Missing abstract section")
            score -= 0.1

        # Check for introduction
        if enforce_sections and '\\section{Introduction}' not in content and '\\section{Introduction}' not in content:
            issues.append("Missing introduction section")
            score -= 0.1

        # Check for conclusion
        if enforce_sections and '\\section{Conclusion}' not in content and '\\section{Conclusions}' not in content:
            issues.append("Missing conclusion section")
            score -= 0.05

        # Check for references section
        if enforce_sections and '\\begin{thebibliography}' not in content and '\\bibliography' not in content:
            issues.append("Missing references section")
            score -= 0.15
        
        # Check for equation formatting issues (long lines)
        equation_blocks = re.findall(r'\\begin{equation}.*?\\end{equation}', content, re.DOTALL)
        for eq in equation_blocks:
            if len(max(eq.split('\n'), key=len)) > 100:  # Rough estimate for line length
                issues.append("Potential equation overflow - long equations should use align or split environments")
                score -= 0.05
                break

        return issues, max(0.0, score)

    def _get_novelty_vetter(self) -> NoveltyVetter:
        if self._novelty_vetter is None:
            self._novelty_vetter = NoveltyVetter()
        return self._novelty_vetter

    def _lookup_novelty_threshold(self, field: str) -> float:
        novelty_cfg = self.config.get('novelty', {})
        domain_thresholds = novelty_cfg.get('domain_thresholds', {})
        field_lower = (field or '').lower()
        for domain, threshold in domain_thresholds.items():
            if domain in field_lower:
                return float(threshold)
        return float(novelty_cfg.get('global_threshold', 0.55))

    def _validate_novelty(self, content: str, field: str) -> Tuple[List[str], float]:
        if 'novelty' not in self.config:
            return [], 0.5

        try:
            assessment = self._get_novelty_vetter().assess_manuscript(content, field or self._active_field)
        except Exception as exc:
            logger.warning(f"Novelty assessment failed: {exc}")
            return [f"WARNING: Novelty assessment unavailable ({exc})"], 0.4

        issues = self._get_novelty_vetter().revision_diagnostics(assessment)
        threshold = self._lookup_novelty_threshold(field or self._active_field)
        if assessment.novelty_score < threshold:
            issues.append(
                f"CRITICAL: Novelty score {assessment.novelty_score:.2f} below baseline {threshold:.2f}"
            )

        return issues, assessment.novelty_score

    def _enforce_fair_acceptance(
        self,
        content: str,
        simulation_output: str,
        metadata: Optional[Dict],
    ) -> Tuple[List[str], Optional[float], Any]:
        if not FAIR_SCORING_AVAILABLE:
            return [], None, None

        try:
            scorer = FairPaperScorer()
            metrics, issues = scorer.score_paper(content, simulation_output, metadata=metadata)
        except Exception as exc:
            logger.warning(f"Fair scoring failed: {exc}")
            return [f"WARNING: Fair scoring failed ({exc})"], None, None

        novelty_cfg = self.config.get('novelty', {})
        threshold = float(novelty_cfg.get('fair_score_threshold', 0.55))

        reported_issues = [f"FAIR: {issue}" for issue in issues[:10]] if issues else []
        if metrics.final_score < threshold:
            reported_issues.append(
                f"CRITICAL: FairPaperScorer final score {metrics.final_score:.2f} below acceptance threshold {threshold:.2f}"
            )

        return reported_issues, metrics.final_score, metrics

    def generate_quality_report(self, issues: List[str], quality_scores: Dict[str, float]) -> str:
        """Generate comprehensive quality assessment report."""
        report = "📊 AUTOMATED QUALITY ASSESSMENT REPORT\n"
        report += "=" * 50 + "\n\n"
        
        # Overall score
        overall_score = quality_scores.get('overall', 0.0)
        report += f"🎯 OVERALL QUALITY SCORE: {overall_score:.2f}/1.0 ({overall_score*100:.1f}%)\n\n"
        
        # Score breakdown
        report += "📈 SCORE BREAKDOWN:\n"
        for category, score in quality_scores.items():
            if category != 'overall':
                status = "✅" if score >= 0.8 else "⚠️" if score >= 0.6 else "❌"
                report += f"  {status} {category.replace('_', ' ').title()}: {score:.2f}\n"
        
        report += "\n"
        
        # Issues summary
        if issues:
            report += f"⚠️ IDENTIFIED ISSUES ({len(issues)}):\n"
            for i, issue in enumerate(issues, 1):
                report += f"{i}. {issue}\n"
        else:
            report += "✅ No significant quality issues detected!\n"
        
        report += "\n"
        
        # Recommendations
        report += "💡 RECOMMENDATIONS:\n"
        if overall_score < 0.6:
            report += "- Paper requires major revision to meet publication standards\n"
            report += "- Address all identified issues before resubmission\n"
        elif overall_score < 0.8:
            report += "- Paper shows promise but needs improvement in identified areas\n"
            report += "- Focus on strengthening experimental rigor and methodology\n"
        else:
            report += "- Paper meets high quality standards\n"
            report += "- Minor improvements in identified areas will enhance impact\n"
        
        return report
    
    def save_quality_report(self, issues: List[str], quality_scores: Dict[str, float], 
                           output_path: Path) -> None:
        """Save quality assessment to file."""
        report_data = {
            "quality_scores": quality_scores,
            "issues": issues,
            "overall_assessment": {
                "score": quality_scores.get('overall', 0.0),
                "grade": self._get_quality_grade(quality_scores.get('overall', 0.0)),
                "recommendation": self._get_recommendation(quality_scores.get('overall', 0.0))
            },
            "generated_at": str(Path.cwd()),
            "config": self.config
        }
        
        # Save JSON data
        json_path = output_path.with_suffix('.json')
        with open(json_path, 'w') as f:
            json.dump(report_data, f, indent=2)
        
        # Save human-readable report
        txt_path = output_path.with_suffix('.txt')
        with open(txt_path, 'w') as f:
            f.write(self.generate_quality_report(issues, quality_scores))
        
        logger.info(f"Quality report saved to {json_path} and {txt_path}")
    
    def _get_quality_grade(self, score: float) -> str:
        """Convert quality score to letter grade."""
        if score >= 0.9: return "A+"
        elif score >= 0.85: return "A"
        elif score >= 0.8: return "A-"
        elif score >= 0.75: return "B+"
        elif score >= 0.7: return "B"
        elif score >= 0.65: return "B-"
        elif score >= 0.6: return "C+"
        elif score >= 0.55: return "C"
        else: return "F"
    
    def _get_recommendation(self, score: float) -> str:
        """Get publication recommendation based on score."""
        if score >= 0.8: return "Accept with minor revisions"
        elif score >= 0.7: return "Major revisions required"
        elif score >= 0.6: return "Reject with resubmission opportunity"
        else: return "Reject"