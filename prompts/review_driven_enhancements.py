#!/usr/bin/env python3
"""
Review-driven prompt enhancements based on actual reviewer feedback patterns.
Addresses specific issues that cause papers to be rejected or receive weak reviews.
"""

from typing import List, Dict, Optional, Tuple
import re

# CRITICAL EMPIRICAL VALIDATION REQUIREMENTS
EMPIRICAL_VALIDATION_PROMPTS = [
    "MANDATORY REAL EXPERIMENTS: You MUST include actual experiments on real benchmarks, not just analytical/simulated results. For ML/AI papers, use standard benchmarks like GSM8K (math), MATH (competition math), HumanEval (code), or domain-appropriate alternatives.",
    
    "MEASURED PERFORMANCE METRICS: Report actual measured numbers from running your system, including accuracy, latency (p50/p95/p99 percentiles), throughput, cost analysis, and resource utilization. NO purely theoretical or simulated results as primary evidence.",
    
    "BENCHMARK COMPARISON: Include comparison with established baselines on named benchmarks with reported numbers. For reasoning papers: compare against CoT, self-consistency, tool-augmented reasoning. For systems papers: compare against existing system implementations.",
    
    "CORRELATION QUANTIFICATION: If claiming correlation-aware improvements, measure and report actual pairwise agreement matrices, Cohen's kappa, route diversity metrics, and entropy. Show how performance changes as correlation varies.",
    
    "COST-BENEFIT ANALYSIS: Report accuracy vs computational budget curves, showing tokens consumed, API calls made, wall-clock time, and dollar cost. Include efficiency frontiers and trade-off analysis.",
    
    "ABLATION STUDIES: Include systematic ablations removing each component to demonstrate contribution. For ensemble methods: show impact of each route, verification steps, aggregation methods.",
    
    "STATISTICAL VALIDATION: All experimental claims must include statistical significance testing, confidence intervals, and effect sizes. Use appropriate statistical tests for your evaluation methodology."
]

# MANDATORY FIGURE AND VISUALIZATION REQUIREMENTS  
FIGURE_REQUIREMENTS_PROMPTS = [
    "MANDATORY FIGURES: Your paper MUST include at least 3-4 figures. Papers with 0 figures are automatically flagged for rejection. Required figure types based on paper type:",
    
    "SYSTEMS PAPERS FIGURES: (1) System architecture diagram showing components and data flow, (2) Performance curves (accuracy vs budget, latency CDFs, throughput), (3) Comparison plots showing your method vs baselines, (4) Ablation results visualization.",
    
    "ML ALGORITHM PAPERS FIGURES: (1) Algorithm schematic or flowchart, (2) Performance comparison plots (accuracy, F1, AUC), (3) Learning curves or convergence plots, (4) Visualization of learned representations or attention patterns.",
    
    "THEORETICAL PAPERS FIGURES: (1) Conceptual diagram of theoretical framework, (2) Proof sketch or theorem relationship diagram, (3) Simulation results supporting theory, (4) Parameter sensitivity analysis plots.",
    
    "FIGURE QUALITY STANDARDS: All figures must be vector-based (TikZ, PGF, or high-resolution), have clear axis labels, legends, and captions that fully explain the content. Use consistent color schemes and font sizes readable in print.",
    
    "DATA VISUALIZATION BEST PRACTICES: Include error bars, confidence intervals, or distribution plots. Use appropriate plot types for data (box plots for distributions, line plots for trends, bar plots for comparisons). Avoid misleading scales or cherry-picked results.",
    
    "FIGURE PLACEMENT AND REFERENCING: Each figure must be referenced in the main text before it appears. Use proper LaTeX figure environments with [htbp] positioning. Captions should be comprehensive and self-contained."
]

# CONCRETE IMPLEMENTATION REQUIREMENTS
CONCRETE_IMPLEMENTATION_PROMPTS = [
    "CONCRETE ALGORITHMS: Replace abstract descriptions with detailed algorithms, pseudocode, and specific implementation details. For 'verification gates' or 'validators', specify exact checking procedures, validation rules, and error handling.",
    
    "SPECIFIC VALIDATORS: For different domains: Math problems (equation solving, unit checking, numerical verification), Code (syntax checking, test execution, type validation), Logic (proof checking, consistency validation), Text (factual verification, coherence checking).",
    
    "PERFORMANCE METRICS: Report specific metrics like True Positive Rate (TPR), False Positive Rate (FPR), precision, recall, calibration error (ECE), Brier score, and coverage-risk curves for abstention systems.",
    
    "IMPLEMENTATION DETAILS: Include hyperparameters, model sizes, hardware requirements, runtime complexity analysis, memory usage, and scalability considerations. Provide enough detail for replication.",
    
    "TASK-SPECIFIC INSTANTIATION: Don't leave concepts abstract. For routing: specify features used, classification methods, training procedures. For aggregation: specify weighting schemes, combination rules, confidence estimation methods.",
    
    "ERROR ANALYSIS: Provide concrete examples of failure cases, error propagation analysis, robustness testing, and edge case handling. Show specific inputs where methods fail and why.",
    
    "REPRODUCIBILITY CODE: Include detailed pseudocode or reference to code repository with specific function signatures, data structures, and algorithms used. Ensure complete reproducibility."
]

# COMPARISON AND POSITIONING REQUIREMENTS
COMPARISON_POSITIONING_PROMPTS = [
    "CLEAR NOVELTY ARTICULATION: Include a dedicated section clearly stating what is novel versus prior work. Use comparison tables showing your method vs existing approaches across key dimensions.",
    
    "COMPREHENSIVE RELATED WORK: Don't just cite papers - explain how your work differs. For ensemble methods: compare to majority voting, Dawid-Skene, mixture of experts, self-consistency. For routing: compare to learned routing, static policies, bandit approaches.",
    
    "POSITIONING TABLE: Create a table comparing methods across dimensions like: uses multiple models/routes, includes verification, correlation-aware, compute-conscious, handles abstention, security considerations.",
    
    "BASELINE COMPARISON: Compare against strongest possible baselines, not strawmen. Include recent state-of-the-art methods, commercial systems where applicable, and standard benchmark results from leaderboards.",
    
    "CONTRIBUTION CLARITY: Start with a contributions section listing 3-4 specific, measurable contributions. Avoid vague claims like 'we propose a novel framework' - be specific about algorithmic, empirical, or theoretical contributions.",
    
    "PRIOR ART GAPS: Clearly identify what existing methods cannot do that yours can. Don't claim novelty for combinations unless the combination itself creates new capabilities or insights.",
    
    "EXPERIMENTAL POSITIONING: Show that your experimental setup is stronger than prior work - more benchmarks, stronger baselines, more comprehensive metrics, better statistical analysis."
]

# EXPERIMENTAL DESIGN REQUIREMENTS
EXPERIMENTAL_DESIGN_PROMPTS = [
    "COMPREHENSIVE BASELINES: Include at least 3-5 strong baselines: (1) Best single method, (2) Simple ensemble (majority vote), (3) State-of-the-art existing method, (4) Oracle upper bound where applicable, (5) Random baseline for calibration.",
    
    "MULTI-BENCHMARK EVALUATION: Test on at least 2-3 diverse benchmarks to show generalization. For reasoning: GSM8K + MATH + HumanEval. For NLP: multiple datasets from different domains. For vision: standard benchmarks with different characteristics.",
    
    "SYSTEMATIC ABLATIONS: Remove each component systematically: (1) w/o verification, (2) w/o routing, (3) w/o correlation correction, (4) w/o abstention, (5) different aggregation methods. Show contribution of each component.",
    
    "STATISTICAL RIGOR: Use proper train/validation/test splits, multiple random seeds (≥5), significance testing, confidence intervals, and effect size reporting. Address multiple testing correction when doing many comparisons.",
    
    "SCALING ANALYSIS: Show how methods perform with different computational budgets, ensemble sizes, or model scales. Include efficiency analysis and computational complexity discussion.",
    
    "FAILURE ANALYSIS: Analyze when and why methods fail. Include examples of failure cases, error propagation analysis, and robustness testing under distribution shift or adversarial inputs.",
    
    "REPRODUCIBILITY STANDARDS: Provide hyperparameters, random seeds, model versions, evaluation protocols, and statistical testing procedures. Include confidence intervals and variance reporting across runs."
]

# LATEX AND PRESENTATION REQUIREMENTS  
PRESENTATION_QUALITY_PROMPTS = [
    "PROPER LATEX FORMATTING: Use proper LaTeX environments. Replace ALL '- item' text with \\begin{itemize}\\item ... \\end{itemize}. Use proper equation environments, algorithm blocks, and table formatting with booktabs package.",
    
    "TERMINOLOGY CONSISTENCY: Maintain consistent terminology throughout. Define technical terms on first use. Use hyphens consistently (test-time, multi-class, correlation-aware). Maintain consistent notation and variable naming.",
    
    "STRUCTURE ORGANIZATION: Follow standard paper structure for your field. Include clear contributions section, comprehensive related work, detailed methodology, thorough experiments, limitations discussion, and concrete conclusions.",
    
    "PROVENANCE AND ATTRIBUTION: Never include sections titled 'Original Extracted Content' or similar. All content must be original or properly attributed with clear citations. Paraphrase rather than copying.",
    
    "WRITING CLARITY: Use active voice, clear topic sentences, and logical paragraph flow. Avoid jargon without definition. Each section should have clear purpose and contribution to the overall narrative.",
    
    "PROFESSIONAL PRESENTATION: Use proper citation formats, consistent formatting, appropriate academic tone, and clear figure/table captions. Ensure double-blind compliance if required.",
    
    "COMPLETENESS CHECKS: Ensure all references resolve, figures are referenced in text, equations are numbered and referenced, algorithms are complete and consistent, and all claims are supported."
]

def detect_review_issues(paper_content: str) -> List[str]:
    """
    Detect specific issues that lead to negative reviews based on the analysis.
    """
    issues = []
    
    # Check for empirical validation issues
    if "analytical" in paper_content.lower() and "measured" not in paper_content.lower():
        issues.append("CRITICAL: Appears to rely on analytical/simulated results without real measurements")
    
    if not re.search(r'GSM8K|MATH|HumanEval|benchmark', paper_content, re.IGNORECASE):
        issues.append("CRITICAL: No standard benchmarks mentioned for evaluation")
        
    # Check for figure requirements
    figure_count = len(re.findall(r'\\begin{figure}', paper_content))
    if figure_count == 0:
        issues.append("CRITICAL: No figures found - papers with 0 figures are flagged for rejection")
    elif figure_count < 3:
        issues.append("WARNING: Only {} figures found - need at least 3-4 for strong paper".format(figure_count))
    
    # Check for concrete implementation
    if "abstract" in paper_content.lower() and "concrete" not in paper_content.lower():
        issues.append("WARNING: May contain abstract descriptions without concrete implementation details")
    
    # Check for comparison issues  
    if "novel" in paper_content.lower() and "comparison" not in paper_content.lower():
        issues.append("WARNING: Claims novelty but lacks comparison table or positioning against prior work")
    
    # Check for LaTeX formatting issues
    if re.search(r'\n\s*-\s+', paper_content):
        issues.append("FORMATTING: Found plain text bullets '- ' that should be \\begin{itemize}\\item")
    
    # Check for provenance issues
    if "extracted content" in paper_content.lower() or "provenance note" in paper_content.lower():
        issues.append("CRITICAL: Contains provenance/extracted content sections that may indicate copied material")
    
    return issues

def enhance_prompt_for_review_quality(base_prompt: str, paper_type: str = "research") -> str:
    """
    Enhance prompts to address common review issues.
    """
    enhancement_sections = []
    
    # Add empirical validation requirements
    enhancement_sections.append("🔬 EMPIRICAL VALIDATION REQUIREMENTS (CRITICAL FOR ACCEPTANCE):\n" + 
                               "\n".join([f"• {req}" for req in EMPIRICAL_VALIDATION_PROMPTS]))
    
    # Add figure requirements
    enhancement_sections.append("📊 MANDATORY FIGURE REQUIREMENTS (0 FIGURES = REJECTION):\n" + 
                               "\n".join([f"• {req}" for req in FIGURE_REQUIREMENTS_PROMPTS]))
    
    # Add concrete implementation requirements
    enhancement_sections.append("⚙️ CONCRETE IMPLEMENTATION REQUIREMENTS:\n" + 
                               "\n".join([f"• {req}" for req in CONCRETE_IMPLEMENTATION_PROMPTS]))
    
    # Add comparison and positioning
    enhancement_sections.append("🎯 COMPARISON AND POSITIONING REQUIREMENTS:\n" + 
                               "\n".join([f"• {req}" for req in COMPARISON_POSITIONING_PROMPTS]))
    
    # Add experimental design requirements
    enhancement_sections.append("🧪 EXPERIMENTAL DESIGN REQUIREMENTS:\n" + 
                               "\n".join([f"• {req}" for req in EXPERIMENTAL_DESIGN_PROMPTS]))
    
    # Add presentation quality requirements
    enhancement_sections.append("✍️ PRESENTATION QUALITY REQUIREMENTS:\n" + 
                               "\n".join([f"• {req}" for req in PRESENTATION_QUALITY_PROMPTS]))
    
    # Combine with base prompt
    enhanced_prompt = base_prompt + "\n\n" + "\n\n".join(enhancement_sections)
    
    # Add final emphasis
    enhanced_prompt += """

🚨 CRITICAL SUCCESS FACTORS:
These requirements are based on actual reviewer feedback that led to paper rejections. Papers failing these criteria typically receive negative reviews citing:
- "Insufficient empirical validation" 
- "Missing figures make evaluation difficult"
- "Abstract descriptions without concrete implementation"
- "Unclear novelty positioning"
- "Weak experimental design"
- "Poor presentation quality"

Your paper must demonstrate CONCRETE, MEASURED results on REAL benchmarks with proper statistical analysis and clear visual presentation to receive positive reviews."""
    
    return enhanced_prompt

def get_review_driven_requirements(paper_type: str) -> Dict[str, List[str]]:
    """
    Get specific requirements based on paper type and review feedback.
    """
    return {
        "empirical_validation": EMPIRICAL_VALIDATION_PROMPTS,
        "figure_requirements": FIGURE_REQUIREMENTS_PROMPTS, 
        "concrete_implementation": CONCRETE_IMPLEMENTATION_PROMPTS,
        "comparison_positioning": COMPARISON_POSITIONING_PROMPTS,
        "experimental_design": EXPERIMENTAL_DESIGN_PROMPTS,
        "presentation_quality": PRESENTATION_QUALITY_PROMPTS
    }