#!/usr/bin/env python3
"""
Enhanced experimental rigor prompts for improving paper quality and review outcomes.
"""

import re
from typing import List, Dict, Optional

# Core experimental rigor requirements (for papers with numerical experiments)
EXPERIMENTAL_RIGOR_PROMPTS = [
    "STATISTICAL SIGNIFICANCE (if conducting experiments): Ensure all experiments include proper statistical significance testing with p-values < 0.05. Report exact p-values, not just 'significant'.",
    
    "CONFIDENCE INTERVALS (if conducting experiments): Include 95% confidence intervals for all reported metrics, especially main results and comparisons.",
    
    "MULTIPLE SEEDS (if conducting experiments): Use minimum 5 different random seeds and report mean ± standard deviation for all experimental results. Show seed-level variance.",
    
    "CROSS-VALIDATION (if conducting experiments): Implement proper k-fold cross-validation (k≥5) or holdout testing methodology with clear train/validation/test splits.",
    
    "SAMPLE SIZE JUSTIFICATION (if conducting experiments): Provide statistical power analysis or justify sample sizes used in experiments. Ensure adequate statistical power (≥0.8).",
    
    "EFFECT SIZE (if conducting experiments): Report effect sizes (Cohen's d, eta-squared, etc.) in addition to significance tests to demonstrate practical significance.",
    
    "MULTIPLE TESTING CORRECTION (if conducting experiments): Apply Bonferroni, FDR, or other appropriate corrections when conducting multiple statistical comparisons.",
    
    "ASSUMPTION VALIDATION (if conducting experiments): Verify statistical test assumptions (normality, homoscedasticity, independence) and use appropriate non-parametric alternatives when violated."
]

# Theoretical rigor requirements (for papers without numerical experiments)
THEORETICAL_RIGOR_PROMPTS = [
    "MATHEMATICAL FORMALIZATION: Provide proper mathematical definitions, theorems, lemmas, and proofs using LaTeX environments (\\begin{theorem}, \\begin{proof}, etc.).",
    
    "LOGICAL ARGUMENTATION: Present clear, logically structured arguments with proper justification for all claims and assertions.",
    
    "SYSTEMATIC ANALYSIS: For survey/review papers, provide systematic taxonomy, comprehensive coverage, and comparative analysis framework.",
    
    "FORMAL PROOFS: Include complete mathematical proofs for all theoretical claims, not just proof sketches or intuitive explanations.",
    
    "CONSISTENCY CHECKING: Ensure all definitions, assumptions, and theoretical constructs are internally consistent throughout the paper.",
    
    "RELATED WORK POSITIONING: Clearly position theoretical contributions within existing theoretical frameworks and highlight novel aspects.",
    
    "CONTRIBUTION CLARITY: Explicitly articulate theoretical contributions and their significance to the field."
]

BASELINE_COMPARISON_PROMPTS = [
    "COMPREHENSIVE BASELINES (if conducting experiments): Compare against at least 3-5 relevant state-of-the-art baselines from recent literature (last 2-3 years).",
    
    "SIMPLE BASELINES (if conducting experiments): Include both sophisticated baselines and simple ones (random, majority class, heuristic methods) to demonstrate meaningful improvement.",
    
    "FAIR COMPARISON (if conducting experiments): Ensure identical evaluation protocols, datasets, metrics, and computational resources across all baseline comparisons.",
    
    "HYPERPARAMETER FAIRNESS (if conducting experiments): Use grid search or Bayesian optimization for all methods, not just your proposed approach. Report hyperparameter sensitivity.",
    
    "COMPUTATIONAL COMPLEXITY: Report training time, inference time, memory usage, and computational complexity (Big O notation) for all methods.",
    
    "IMPLEMENTATION DETAILS: Clearly specify implementation details, libraries, and versions used for baseline reproductions.",
    
    "STATISTICAL COMPARISON (if conducting experiments): Use paired statistical tests (paired t-test, Wilcoxon signed-rank) when comparing methods on same data splits.",
    
    "THEORETICAL COMPARISON (for theoretical papers): Compare theoretical properties, assumptions, guarantees, and applicability with existing theoretical approaches."
]

METHODOLOGY_CLARITY_PROMPTS = [
    "ALGORITHMIC PSEUDOCODE: Provide complete, executable pseudocode for all novel algorithms using proper algorithm environments in LaTeX.",
    
    "HYPERPARAMETER DOCUMENTATION: Document ALL hyperparameters with exact values, ranges tested, and selection methodology (grid search, random search, Bayesian optimization).",
    
    "MODEL ARCHITECTURE: Specify exact model architectures including layer types, dimensions, activation functions, regularization, and initialization schemes.",
    
    "PREPROCESSING PIPELINE: Document all data preprocessing steps, normalization schemes, feature engineering, and data augmentation techniques with exact parameters.",
    
    "REPRODUCIBILITY CHECKLIST: Provide comprehensive reproducibility information including software versions, hardware specifications, random seeds, and execution environment.",
    
    "TRAINING PROCEDURES: Detail optimization algorithms, learning rate schedules, batch sizes, early stopping criteria, and convergence thresholds.",
    
    "EVALUATION PROTOCOLS: Specify exact evaluation procedures, metrics computation, aggregation methods, and statistical testing procedures."
]

def get_experimental_rigor_system_prompt() -> str:
    """Generate system prompt emphasizing experimental rigor."""
    return f"""
🔬 PAPER TYPE ADAPTIVE REQUIREMENTS:

FOR EXPERIMENTAL PAPERS (with numerical experiments/evaluations):
{chr(10).join('- ' + prompt for prompt in EXPERIMENTAL_RIGOR_PROMPTS)}

FOR THEORETICAL PAPERS (without numerical experiments):
{chr(10).join('- ' + prompt for prompt in THEORETICAL_RIGOR_PROMPTS)}

UNIVERSAL REQUIREMENTS (all paper types):
{chr(10).join('- ' + prompt for prompt in BASELINE_COMPARISON_PROMPTS)}

METHODOLOGY DOCUMENTATION (all paper types):
{chr(10).join('- ' + prompt for prompt in METHODOLOGY_CLARITY_PROMPTS)}

NOTE: The system will automatically detect whether your paper is experimental (with numerical results) 
or theoretical (mathematical analysis, surveys, position papers) and apply appropriate requirements.
For experimental papers, statistical rigor is mandatory. For theoretical papers, mathematical 
formalization and logical argumentation are emphasized instead.

CRITICAL: Every claim must be supported by appropriate evidence - statistical for experimental 
papers, mathematical proof for theoretical papers. Reviewers will specifically look for these 
elements based on your paper type.
"""

def detect_paper_type(content: str, simulation_output: str = "") -> str:
    """Detect if paper is experimental, theoretical, survey, or position paper."""
    content_lower = content.lower()
    
    # Strong indicators for different paper types
    experimental_indicators = [
        'experiment', 'evaluation', 'performance', 'accuracy', 'dataset',
        'baseline', 'benchmark', 'test', 'validation', 'empirical',
        'simulation', 'results', 'metrics'
    ]
    
    theoretical_indicators = [
        'theorem', 'proof', 'lemma', 'proposition', 'mathematical',
        'formal', 'theoretical', 'analysis', 'framework'
    ]
    
    survey_indicators = [
        'survey', 'review', 'overview', 'comprehensive', 'systematic',
        'taxonomy', 'classification', 'state-of-the-art'
    ]
    
    position_indicators = [
        'position', 'perspective', 'opinion', 'discussion', 'commentary',
        'viewpoint', 'position paper', 'workshop'
    ]
    
    # Count indicators
    exp_score = sum(1 for ind in experimental_indicators if ind in content_lower)
    theo_score = sum(1 for ind in theoretical_indicators if ind in content_lower)
    survey_score = sum(1 for ind in survey_indicators if ind in content_lower)
    position_score = sum(1 for ind in position_indicators if ind in content_lower)
    
    # Boost experimental score if has simulation code
    if simulation_output and len(simulation_output.strip()) > 100:
        exp_score += 3
    
    # Boost experimental score if has results/evaluation sections
    if re.search(r'\\section\{.*results.*\}|\\section\{.*evaluation.*\}', content, re.IGNORECASE):
        exp_score += 2
    
    # Determine paper type based on highest score
    scores = {
        'experimental': exp_score,
        'theoretical': theo_score,
        'survey': survey_score,
        'position': position_score
    }
    
    paper_type = max(scores.items(), key=lambda x: x[1])[0]
    
    # If scores are too close or too low, default to experimental
    if max(scores.values()) < 2 or (exp_score > 0 and abs(exp_score - max(scores.values())) <= 1):
        paper_type = 'experimental'
    
    return paper_type

def get_paper_type_specific_prompts(paper_type: str, domain: Optional[str] = None) -> List[str]:
    """Get prompts specific to paper type."""
    if paper_type == 'experimental':
        prompts = EXPERIMENTAL_RIGOR_PROMPTS + BASELINE_COMPARISON_PROMPTS
    elif paper_type == 'theoretical':
        prompts = THEORETICAL_RIGOR_PROMPTS + [p for p in BASELINE_COMPARISON_PROMPTS if 'theoretical' in p.lower() or 'computational complexity' in p.lower()]
    elif paper_type == 'survey':
        prompts = [
            "COMPREHENSIVE COVERAGE: Provide systematic coverage of the field with clear inclusion/exclusion criteria.",
            "TAXONOMY DEVELOPMENT: Develop clear taxonomies or classification schemes for organizing the literature.",
            "COMPARATIVE ANALYSIS: Include detailed comparative analysis tables or frameworks.",
            "TREND IDENTIFICATION: Identify and discuss key trends, gaps, and future directions.",
            "SYSTEMATIC METHODOLOGY: Use systematic review methodology with clear search strategy."
        ]
    elif paper_type == 'position':
        prompts = [
            "CLEAR POSITION: Articulate your position or perspective clearly and early in the paper.",
            "EVIDENCE-BASED ARGUMENTATION: Support your position with evidence from literature or logical reasoning.",
            "COUNTERARGUMENT ADDRESSING: Acknowledge and address potential counterarguments.",
            "COMMUNITY IMPACT: Discuss implications for the research community or field.",
            "ACTIONABLE RECOMMENDATIONS: Provide concrete recommendations or calls to action."
        ]
    else:
        prompts = EXPERIMENTAL_RIGOR_PROMPTS  # Default fallback
    
    # Add domain-specific prompts if applicable
    if domain == "poker":
        prompts.extend(get_poker_specific_prompts())
    
    return prompts

def get_poker_specific_prompts() -> List[str]:
    """Get poker AI specific experimental requirements."""
    return [
        "EXPLOITABILITY ANALYSIS: Measure and report exploitability metrics against known optimal strategies (when available) or approximate Nash equilibria.",
        
        "NASH EQUILIBRIUM CONVERGENCE: Analyze convergence to Nash equilibrium using appropriate game-theoretic metrics (Nash-conv, exploitability bounds).",
        
        "OPPONENT DIVERSITY: Test against diverse opponent types including tight-aggressive, loose-passive, and exploitative strategies with different playing styles.",
        
        "GAME FORMAT ROBUSTNESS: Evaluate performance across different poker formats (cash games, tournaments, heads-up, multi-table) and stack depth variations.",
        
        "BETTING PATTERN ANALYSIS: Analyze and report betting frequencies, bluffing rates, and strategic patterns compared to game-theoretically optimal play.",
        
        "REAL-TIME CONSTRAINTS: Evaluate performance under realistic time constraints and computational budgets typical in actual poker environments.",
        
        "SAMPLE COMPLEXITY: Report sample efficiency in terms of hands played to reach various performance milestones compared to existing approaches.",
        
        "GENERALIZATION TESTING: Test on held-out game scenarios, opponent types, or rule variations not seen during training to demonstrate generalization."
    ]
    """Get poker AI specific experimental requirements."""
    return [
        "EXPLOITABILITY ANALYSIS: Measure and report exploitability metrics against known optimal strategies (when available) or approximate Nash equilibria.",
        
        "NASH EQUILIBRIUM CONVERGENCE: Analyze convergence to Nash equilibrium using appropriate game-theoretic metrics (Nash-conv, exploitability bounds).",
        
        "OPPONENT DIVERSITY: Test against diverse opponent types including tight-aggressive, loose-passive, and exploitative strategies with different playing styles.",
        
        "GAME FORMAT ROBUSTNESS: Evaluate performance across different poker formats (cash games, tournaments, heads-up, multi-table) and stack depth variations.",
        
        "BETTING PATTERN ANALYSIS: Analyze and report betting frequencies, bluffing rates, and strategic patterns compared to game-theoretically optimal play.",
        
        "REAL-TIME CONSTRAINTS: Evaluate performance under realistic time constraints and computational budgets typical in actual poker environments.",
        
        "SAMPLE COMPLEXITY: Report sample efficiency in terms of hands played to reach various performance milestones compared to existing approaches.",
        
        "GENERALIZATION TESTING: Test on held-out game scenarios, opponent types, or rule variations not seen during training to demonstrate generalization."
    ]

def get_training_stability_prompts() -> List[str]:
    """Get training stability monitoring requirements."""
    return [
        "CONVERGENCE MONITORING: Track and report training stability metrics including loss convergence, gradient norms, and parameter update magnitudes.",
        
        "LEARNING CURVES: Include comprehensive learning curves with error bars across multiple independent training runs, showing both training and validation performance.",
        
        "VARIANCE ANALYSIS: Report final model performance variance across multiple training runs and analyze sources of variance (initialization, data sampling, etc.).",
        
        "CURRICULUM ANALYSIS: If using curriculum learning, document progression stages and analyze the impact of curriculum design on final performance.",
        
        "HYPERPARAMETER SENSITIVITY: Conduct sensitivity analysis for key hyperparameters and report robustness to hyperparameter choices.",
        
        "EARLY STOPPING VALIDATION: If using early stopping, justify criteria and analyze impact on generalization performance.",
        
        "COMPUTATIONAL SCALABILITY: Report how training time and performance scale with dataset size, model complexity, and computational resources."
    ]

def enhance_prompt_with_rigor(base_prompt: str, domain_specific: Optional[str] = None) -> str:
    """Enhance any base prompt with experimental rigor requirements."""
    rigor_additions = get_experimental_rigor_system_prompt()
    
    if domain_specific == "poker":
        poker_prompts = get_poker_specific_prompts()
        rigor_additions += f"""

🎰 POKER-SPECIFIC EXPERIMENTAL REQUIREMENTS:
{chr(10).join('- ' + prompt for prompt in poker_prompts)}
"""
    
    training_prompts = get_training_stability_prompts()
    rigor_additions += f"""

📈 TRAINING STABILITY REQUIREMENTS:
{chr(10).join('- ' + prompt for prompt in training_prompts)}
"""
    
    return base_prompt + "\n\n" + rigor_additions

def get_results_presentation_prompts() -> List[str]:
    """Get requirements for high-quality results presentation."""
    return [
        "PUBLICATION FIGURES: Create clear, publication-quality figures with proper axis labels, legends, error bars, and consistent styling.",
        
        "STATISTICAL REPORTING: Use appropriate statistical tests and report exact p-values, confidence intervals, and effect sizes for all comparisons.",
        
        "RESULT AGGREGATION: Provide both aggregate results across all conditions and detailed per-scenario/per-condition breakdowns.",
        
        "ERROR ANALYSIS: Include comprehensive error analysis, failure case discussions, and limitations of the proposed approach.",
        
        "SIGNIFICANCE TESTING: Ensure all performance claims are supported by appropriate statistical significance tests with multiple testing corrections.",
        
        "VISUAL CLARITY: Use clear, colorblind-friendly color schemes and ensure all figures are interpretable in grayscale for print publications.",
        
        "NUMERICAL PRECISION: Report results with appropriate precision (not excessive decimal places) and include uncertainty quantification."
    ]

def get_literature_review_prompts() -> List[str]:
    """Get requirements for comprehensive literature reviews."""
    return [
        "RECENT COVERAGE: Include comprehensive coverage of recent works from top-tier venues (NeurIPS, ICML, ICLR, AAAI, IJCAI) from the last 2-3 years.",
        
        "TAXONOMY POSITIONING: Position your work clearly within existing literature taxonomy and identify the specific gap being addressed.",
        
        "LIMITATION ANALYSIS: Identify and discuss specific limitations of previous approaches that motivate your proposed solution.",
        
        "CONTRIBUTION CLARITY: Highlight specific novel contributions that go beyond incremental improvements, with clear technical differentiation.",
        
        "RELATED WORK INTEGRATION: Integrate related work discussion throughout the paper, not just in a single section, showing connections to your contributions.",
        
        "CITATION COMPLETENESS: Ensure citations are complete, accurate, and formatted consistently. Verify all referenced papers exist and are correctly cited.",
        
        "COMPARATIVE ANALYSIS: Provide detailed comparative analysis of your approach versus existing methods, highlighting advantages and trade-offs."
    ]

# Configuration for automated quality checking
QUALITY_CHECK_CONFIG = {
    "experimental_design": {
        "min_random_seeds": 5,
        "required_statistical_tests": ["t-test", "confidence_intervals"],
        "confidence_level": 0.95,
        "min_baselines": 3,
        "effect_size_reporting": True
    },
    "evaluation_metrics": {
        "primary_metrics": ["accuracy", "precision", "recall", "f1"],
        "secondary_metrics": ["training_time", "inference_time", "memory_usage"],
        "statistical_measures": ["mean", "std", "confidence_intervals", "p_values"]
    },
    "poker_specific": {
        "required_metrics": ["exploitability", "nash_convergence", "win_rate"],
        "opponent_types": ["tight_aggressive", "loose_passive", "exploitative"],
        "game_formats": ["cash", "tournament", "heads_up"],
        "evaluation_hands": 100000
    }
}

def validate_experimental_rigor(paper_content: str, results_data: Dict) -> List[str]:
    """Validate paper content for experimental rigor requirements."""
    issues = []
    
    # Check for statistical significance reporting
    if "p-value" not in paper_content.lower() and "p<" not in paper_content.lower():
        issues.append("Missing statistical significance testing and p-value reporting")
    
    # Check for confidence intervals
    if "confidence interval" not in paper_content.lower() and "ci" not in paper_content.lower():
        issues.append("Missing confidence interval reporting")
    
    # Check for multiple seeds/runs
    if "seed" not in paper_content.lower() and "run" not in paper_content.lower():
        issues.append("No evidence of multiple experimental runs or seeds")
    
    # Check for baseline comparisons
    baseline_indicators = ["baseline", "state-of-the-art", "comparison", "versus"]
    if not any(indicator in paper_content.lower() for indicator in baseline_indicators):
        issues.append("Insufficient baseline comparison discussion")
    
    # Check results data if provided
    if results_data:
        if "seeds" in results_data and len(results_data["seeds"]) < 5:
            issues.append(f"Insufficient random seeds: {len(results_data['seeds'])} < 5")
        
        if "baselines" in results_data and len(results_data["baselines"]) < 3:
            issues.append(f"Insufficient baseline comparisons: {len(results_data['baselines'])} < 3")
    
    return issues


def get_statistical_rigor_requirements() -> str:
    """Get formatted statistical rigor requirements."""
    return "\n".join([f"- {req}" for req in EXPERIMENTAL_RIGOR_PROMPTS])


def get_theoretical_rigor_requirements() -> str:
    """Get formatted theoretical rigor requirements."""
    return "\n".join([f"- {req}" for req in THEORETICAL_RIGOR_PROMPTS])