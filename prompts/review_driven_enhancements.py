#!/usr/bin/env python3
"""
Review-driven prompt enhancements based on actual reviewer feedback patterns.
Addresses specific issues that cause papers to be rejected or receive weak reviews.
"""

from typing import List, Dict, Optional, Tuple
import re
from dataclasses import dataclass, replace

from document_types import DocumentType, get_document_template


@dataclass(frozen=True)
class DomainReviewProfile:
    name: str
    empirical_validation: List[str]
    figure_requirements: List[str]
    baseline_expectations: List[str]
    benchmark_terms: List[str]
    baseline_keywords: List[str]
    dataset_terms: List[str]
    min_figures: int = 3
    min_tables: int = 1


# Baseline prompt sets shared across domains
DEFAULT_EMPIRICAL_VALIDATION_PROMPTS = [
    "Provide empirical or analytical evidence using methods that are standard for your field (e.g., laboratory experiments, simulations, user studies, archival datasets).",
    "Report measured performance metrics with appropriate units—accuracy, error rates, latency, ROI, effect sizes, or other domain-relevant indicators—rather than purely qualitative claims.",
    "Compare against credible baselines or prior work using benchmarks, case studies, or historical data that your community recognizes as meaningful.",
    "Summarize uncertainty with confidence intervals, statistical tests, or sensitivity analyses that match the methodology of your discipline."
]

DEFAULT_FIGURE_REQUIREMENTS_PROMPTS = [
    "Include at least three visuals (figures or tables) that explain the methodology and highlight the most important results.",
    "Use visualization styles that suit the domain—schematics for methods, plots for quantitative trends, maps or diagrams for spatial data—with clear legends and captions.",
    "Reference each visual in the surrounding text and describe the insight it conveys so readers can interpret the evidence quickly."
]

DEFAULT_BASELINE_EXPECTATIONS = [
    "Select comparison points that represent the state of practice in your field and explain why they are relevant.",
    "Quantify the improvement over each baseline using domain-appropriate metrics.",
    "When direct baselines are unavailable, use ablations or historical data to contextualize performance claims."
]

ML_EMPIRICAL_VALIDATION_PROMPTS = [
    "Report results on widely recognised machine-learning benchmarks relevant to the task (e.g., MMLU, GSM8K, HumanEval, ImageNet, or comparable datasets).",
    "Provide measured accuracy, calibration, latency, or cost metrics from actual model executions rather than hypothetical estimates.",
    "Compare against strong baselines such as recent published models, established prompting techniques, or open-source reference implementations.",
    "Include ablation studies or component analyses to demonstrate the contribution of each part of the system."
]

ML_FIGURE_REQUIREMENTS_PROMPTS = [
    "Show architecture diagrams or pipeline schematics that clarify the learning setup.",
    "Plot quantitative performance (accuracy, F1, BLEU, etc.) across datasets or compute budgets, including error bars where possible.",
    "Visualise qualitative outputs or attention/embedding plots that help readers understand model behaviour."
]

ML_BASELINE_EXPECTATIONS = [
    "Compare against strong published baselines, including recent large models or fine-tuned systems.",
    "Report results on at least two benchmarks that the target community recognises.",
    "Document training and inference settings (batch size, hardware, token budgets) alongside baseline comparisons."
]

FINANCE_EMPIRICAL_VALIDATION_PROMPTS = [
    "Backtest strategies on historical market data that span multiple regimes and instruments.",
    "Report risk-adjusted performance metrics (Sharpe, Sortino, maximum drawdown) together with turnover and transaction costs.",
    "Evaluate against representative benchmarks such as broad market indices, factor portfolios, or peer strategies.",
    "Stress-test the method under adverse market conditions and include scenario or sensitivity analyses."
]

FINANCE_FIGURE_REQUIREMENTS_PROMPTS = [
    "Include return or drawdown curves that illustrate performance over time.",
    "Provide bar or heatmap visualisations of factor exposures, sector allocations, or scenario outcomes.",
    "Use tables to summarise valuation metrics, risk decomposition, or comparative performance across benchmarks."
]

FINANCE_BASELINE_EXPECTATIONS = [
    "Benchmark against indices such as the S&P 500, MSCI World, or relevant sector ETFs.",
    "Discuss how the strategy performs relative to buy-and-hold, risk-parity, or factor-investing baselines.",
    "Explain data sources, cleaning steps, and look-ahead bias controls so results are auditable."
]

MEDICAL_EMPIRICAL_VALIDATION_PROMPTS = [
    "Describe study design and patient cohorts clearly, including inclusion/exclusion criteria.",
    "Report clinically meaningful metrics (sensitivity, specificity, AUROC, hazard ratios) with confidence intervals.",
    "Compare against the standard of care, experienced clinicians, or prior clinical studies when possible.",
    "Discuss ethical approvals, data governance, and limitations affecting clinical deployment."
]

MEDICAL_FIGURE_REQUIREMENTS_PROMPTS = [
    "Include study CONSORT-style diagrams or workflow charts describing patient/data flow.",
    "Visualise diagnostic performance (ROC curves, calibration plots) and subgroup analyses.",
    "Provide tables that summarise cohort characteristics or outcome distributions."
]

MEDICAL_BASELINE_EXPECTATIONS = [
    "Compare against standard-of-care treatments or widely accepted clinical risk scores.",
    "Highlight differences relative to published clinical trials or registries.",
    "Report statistical significance using tests common in medical research (e.g., log-rank, chi-squared)."
]

SURVEY_EMPIRICAL_VALIDATION_PROMPTS = [
    "Ensure the survey covers the breadth of literature across seminal and recent works.",
    "Synthesize findings into taxonomies, comparative tables, or timelines that highlight trends.",
    "Discuss evaluation criteria reported in the literature and identify gaps or inconsistencies.",
    "Highlight open problems, emerging themes, and consensus viewpoints."
]

SURVEY_FIGURE_REQUIREMENTS_PROMPTS = [
    "Provide taxonomy diagrams or conceptual maps organising the literature.",
    "Include summary tables comparing methods, datasets, or evaluation criteria.",
    "Use timeline or trend plots to show the evolution of the field."
]

SURVEY_BASELINE_EXPECTATIONS = [
    "Contrast representative methods or categories across consistent dimensions (e.g., data, methodology, evaluation).",
    "Document the datasets and benchmarks most frequently used in the literature.",
    "Discuss strengths, weaknesses, and adoption barriers for each major approach."
]

PRESENTATION_EMPIRICAL_VALIDATION_PROMPTS = [
    "Summarise the evidence base succinctly—datasets, experiments, or case studies that underpin the recommendations.",
    "Highlight key quantitative takeaways (metrics, financial impact, patient outcomes) using bullet-friendly phrasing.",
    "Reference supporting sources or appendices so interested readers can verify claims."
]

PRESENTATION_FIGURE_REQUIREMENTS_PROMPTS = [
    "Use visually engaging charts, diagrams, or illustrations on most slides to maintain audience attention.",
    "Keep tables lightweight—focus on headline numbers or comparisons that reinforce the narrative.",
    "Include a concluding summary slide that visualises the main call-to-action or roadmap."
]

PRESENTATION_BASELINE_EXPECTATIONS = [
    "Briefly contrast the proposal with current practice or competing options so the audience understands differentiation.",
    "Highlight constraints, risks, or mitigation steps that decision makers care about.",
    "Point to appendices or backup material for deeper technical or financial validation if needed."
]

ENGINEERING_EMPIRICAL_VALIDATION_PROMPTS = [
    "Measure throughput, latency, resource utilisation, and reliability under realistic workloads.",
    "Benchmark against previous-generation systems, open-source alternatives, or published engineering baselines.",
    "Provide reproducible configuration details (hardware specs, dataset characteristics, load profiles).",
    "Include stress tests or failure mode analyses covering edge cases relevant to deployment."
]

ENGINEERING_FIGURE_REQUIREMENTS_PROMPTS = [
    "Present system architecture diagrams that show component interactions and data flow.",
    "Include performance plots (throughput vs latency, scaling curves) with annotations for key observations.",
    "Use tables to summarise configuration parameters or benchmark comparisons across hardware setups."
]

ENGINEERING_BASELINE_EXPECTATIONS = [
    "Compare against baseline implementations or industry-standard benchmarks (e.g., SPEC, TPC, MLPerf).",
    "Quantify efficiency gains such as cost-per-operation, energy usage, or resource savings.",
    "Document workload characteristics and tuning steps so others can reproduce the benchmarks."
]
# Domain profile lookup tables

DEFAULT_REVIEW_PROFILE = DomainReviewProfile(
    name="General Research",
    empirical_validation=DEFAULT_EMPIRICAL_VALIDATION_PROMPTS,
    figure_requirements=DEFAULT_FIGURE_REQUIREMENTS_PROMPTS,
    baseline_expectations=DEFAULT_BASELINE_EXPECTATIONS,
    benchmark_terms=["benchmark", "dataset", "evaluation"],
    baseline_keywords=["baseline", "comparison", "state-of-the-art"],
    dataset_terms=["dataset", "corpus", "measurement"],
    min_figures=3,
    min_tables=1,
)

ML_REVIEW_PROFILE = DomainReviewProfile(
    name="Machine Learning",
    empirical_validation=ML_EMPIRICAL_VALIDATION_PROMPTS,
    figure_requirements=ML_FIGURE_REQUIREMENTS_PROMPTS,
    baseline_expectations=ML_BASELINE_EXPECTATIONS,
    benchmark_terms=["GSM8K", "HumanEval", "MMLU", "ImageNet", "CIFAR", "SQuAD", "GLUE", "benchmark"],
    baseline_keywords=["baseline", "SOTA", "state-of-the-art", "fine-tuned", "reference model"],
    dataset_terms=["dataset", "corpus", "benchmark"],
    min_figures=3,
    min_tables=1,
)

FINANCE_REVIEW_PROFILE = DomainReviewProfile(
    name="Financial Research",
    empirical_validation=FINANCE_EMPIRICAL_VALIDATION_PROMPTS,
    figure_requirements=FINANCE_FIGURE_REQUIREMENTS_PROMPTS,
    baseline_expectations=FINANCE_BASELINE_EXPECTATIONS,
    benchmark_terms=["S&P 500", "MSCI", "Russell", "Fama-French", "Bloomberg", "market index"],
    baseline_keywords=["benchmark index", "buy-and-hold", "market portfolio", "risk-free", "alpha"],
    dataset_terms=["historical price", "return series", "macroeconomic", "financial data"],
    min_figures=3,
    min_tables=2,
)

MEDICAL_REVIEW_PROFILE = DomainReviewProfile(
    name="Medical/Clinical Research",
    empirical_validation=MEDICAL_EMPIRICAL_VALIDATION_PROMPTS,
    figure_requirements=MEDICAL_FIGURE_REQUIREMENTS_PROMPTS,
    baseline_expectations=MEDICAL_BASELINE_EXPECTATIONS,
    benchmark_terms=["clinical trial", "patient cohort", "MIMIC", "UK Biobank", "registry"],
    baseline_keywords=["standard of care", "control group", "baseline cohort", "placebo"],
    dataset_terms=["patient", "cohort", "clinical dataset", "registry"],
    min_figures=3,
    min_tables=2,
)

SURVEY_REVIEW_PROFILE = DomainReviewProfile(
    name="Survey Paper",
    empirical_validation=SURVEY_EMPIRICAL_VALIDATION_PROMPTS,
    figure_requirements=SURVEY_FIGURE_REQUIREMENTS_PROMPTS,
    baseline_expectations=SURVEY_BASELINE_EXPECTATIONS,
    benchmark_terms=[],
    baseline_keywords=["taxonomy", "comparative", "overview", "survey"],
    dataset_terms=["dataset", "benchmark", "corpus"],
    min_figures=2,
    min_tables=1,
)

PRESENTATION_REVIEW_PROFILE = DomainReviewProfile(
    name="Presentation",
    empirical_validation=PRESENTATION_EMPIRICAL_VALIDATION_PROMPTS,
    figure_requirements=PRESENTATION_FIGURE_REQUIREMENTS_PROMPTS,
    baseline_expectations=PRESENTATION_BASELINE_EXPECTATIONS,
    benchmark_terms=[],
    baseline_keywords=["current state", "option", "alternative"],
    dataset_terms=["case study", "metric", "result"],
    min_figures=3,
    min_tables=0,
)

ENGINEERING_REVIEW_PROFILE = DomainReviewProfile(
    name="Engineering/System Paper",
    empirical_validation=ENGINEERING_EMPIRICAL_VALIDATION_PROMPTS,
    figure_requirements=ENGINEERING_FIGURE_REQUIREMENTS_PROMPTS,
    baseline_expectations=ENGINEERING_BASELINE_EXPECTATIONS,
    benchmark_terms=["throughput", "latency", "benchmark", "SPEC", "TPC", "MLPerf"],
    baseline_keywords=["baseline system", "previous generation", "reference implementation", "throughput"],
    dataset_terms=["workload", "benchmark", "trace", "traffic"],
    min_figures=3,
    min_tables=1,
)


DOMAIN_REVIEW_PROFILES: Dict[DocumentType, DomainReviewProfile] = {
    DocumentType.FINANCE_RESEARCH: FINANCE_REVIEW_PROFILE,
    DocumentType.EQUITY_RESEARCH: FINANCE_REVIEW_PROFILE,
    DocumentType.PRESENTATION_SLIDES: PRESENTATION_REVIEW_PROFILE,
    DocumentType.SURVEY_PAPER: SURVEY_REVIEW_PROFILE,
    DocumentType.ENGINEERING_PAPER: ENGINEERING_REVIEW_PROFILE,
}


FIELD_KEYWORD_PROFILES: List[Tuple[Tuple[str, ...], DomainReviewProfile]] = [
    (("machine learning", "artificial intelligence", "deep learning", "nlp", "computer vision"), ML_REVIEW_PROFILE),
    (("finance", "financial", "portfolio", "investment", "trading", "market"), FINANCE_REVIEW_PROFILE),
    (("medical", "clinical", "health", "biomedical", "patient", "healthcare"), MEDICAL_REVIEW_PROFILE),
    (("survey", "literature review", "taxonomy"), SURVEY_REVIEW_PROFILE),
]


def get_domain_review_profile(
    doc_type: Optional[DocumentType],
    field: str = "",
) -> DomainReviewProfile:
    """Select a review profile based on document type and field heuristics."""

    profile = DEFAULT_REVIEW_PROFILE

    if doc_type and doc_type in DOMAIN_REVIEW_PROFILES:
        profile = DOMAIN_REVIEW_PROFILES[doc_type]
    else:
        field_lower = field.lower()
        for keywords, candidate_profile in FIELD_KEYWORD_PROFILES:
            if any(keyword in field_lower for keyword in keywords):
                profile = candidate_profile
                break

    return replace(profile)


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

def detect_review_issues(
    paper_content: str,
    doc_type: Optional[DocumentType] = None,
    field: str = "",
) -> List[str]:
    """Detect review issues using domain-aware expectations."""

    issues: List[str] = []
    content_lower = paper_content.lower()
    template = get_document_template(doc_type) if doc_type else None
    profile = get_domain_review_profile(doc_type, field)

    requires_empirical = True
    if template and not template.requires_simulation:
        requires_empirical = False

    if requires_empirical:
        if "analytical" in content_lower and "measured" not in content_lower:
            issues.append("CRITICAL: Appears to rely on analytical/simulated results without measured evidence")

        if profile.benchmark_terms:
            if not any(re.search(term, paper_content, re.IGNORECASE) for term in profile.benchmark_terms):
                example = profile.benchmark_terms[0]
                issues.append(
                    f"CRITICAL: No domain benchmarks mentioned (e.g., {example})"
                )
        else:
            if not any(keyword in content_lower for keyword in ["benchmark", "evaluation", "dataset", "experiment"]):
                issues.append("CRITICAL: No evaluation benchmarks or datasets mentioned")

        if profile.baseline_keywords and not any(
            keyword in content_lower for keyword in profile.baseline_keywords
        ):
            baseline_hint = profile.baseline_expectations[0] if profile.baseline_expectations else "Provide baseline comparisons"
            issues.append(f"WARNING: Missing clear baseline comparisons. {baseline_hint}")
    else:
        issues.append(
            f"INFO: Empirical benchmark requirements relaxed for {doc_type.value if doc_type else 'this'} template"
        )

    # Figure expectations
    figure_count = len(re.findall(r'\\begin{figure}', paper_content))
    min_figures = profile.min_figures if (template is None or template.requires_figures) else 0
    if min_figures > 0 and template and not template.requires_figures:
        min_figures = 0

    if min_figures > 0:
        if figure_count == 0:
            issues.append("CRITICAL: No figures found despite template requirements")
        elif figure_count < min_figures:
            issues.append(
                f"WARNING: Only {figure_count} figures found - {profile.name} documents typically include {min_figures}+ visuals"
            )
    else:
        if template and not template.requires_figures:
            issues.append(
                f"INFO: Figure count requirement skipped for {doc_type.value if doc_type else 'this'} template"
            )

    # Table expectations when relevant
    table_count = len(re.findall(r'\\begin{table}', paper_content))
    min_tables = profile.min_tables if (template is None or template.requires_tables) else 0
    if min_tables > 0 and table_count < min_tables:
        issues.append(
            f"WARNING: Only {table_count} tables found - expected at least {min_tables} to summarise quantitative results"
        )

    # Concrete implementation
    if "abstract" in content_lower and "concrete" not in content_lower:
        issues.append("WARNING: May contain abstract descriptions without concrete implementation details")

    # Comparison issues
    if "novel" in content_lower and "comparison" not in content_lower and profile.baseline_expectations:
        issues.append("WARNING: Claims novelty but lacks explicit comparison against prior work")

    # LaTeX formatting issues
    if re.search(r'\n\s*-\s+', paper_content):
        issues.append("FORMATTING: Found plain text bullets '- ' that should be \\begin{itemize}\\item")

    # Provenance issues
    if "extracted content" in content_lower or "provenance note" in content_lower:
        issues.append("CRITICAL: Contains provenance/extracted content sections that may indicate copied material")

    return issues

def enhance_prompt_for_review_quality(
    base_prompt: str,
    paper_type: str = "research",
    doc_type: Optional[DocumentType] = None,
    field: str = "",
) -> str:
    """Enhance prompts to address common review issues in a domain-aware way."""

    profile = get_domain_review_profile(doc_type, field)
    enhancement_sections = []

    enhancement_sections.append(
        "🔬 EMPIRICAL VALIDATION REQUIREMENTS (ALIGN WITH REVIEWERS):\n"
        + "\n".join([f"• {req}" for req in profile.empirical_validation])
    )

    enhancement_sections.append(
        "📊 VISUAL AND TABLE EXPECTATIONS:\n"
        + "\n".join([f"• {req}" for req in profile.figure_requirements])
    )

    if profile.baseline_expectations:
        enhancement_sections.append(
            "⚖️ BASELINE & COMPARISON GUIDELINES:\n"
            + "\n".join([f"• {req}" for req in profile.baseline_expectations])
        )

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
    enhanced_prompt += (
        "\n\n🚨 CRITICAL SUCCESS FACTORS:\n"
        "Reviewers expect evidence that matches the norms of the "
        f"{profile.name.lower()} community. Highlight rigorous validation, clear visuals, and transparent comparisons "
        "so the paper survives rebuttal."
    )

    return enhanced_prompt

def get_review_driven_requirements(
    paper_type: str,
    doc_type: Optional[DocumentType] = None,
    field: str = "",
) -> Dict[str, List[str]]:
    """Get specific review requirements tailored to the detected domain."""

    profile = get_domain_review_profile(doc_type, field)

    return {
        "empirical_validation": profile.empirical_validation,
        "figure_requirements": profile.figure_requirements,
        "baseline_expectations": profile.baseline_expectations,
        "expected_benchmarks": profile.benchmark_terms,
        "concrete_implementation": CONCRETE_IMPLEMENTATION_PROMPTS,
        "comparison_positioning": COMPARISON_POSITIONING_PROMPTS,
        "experimental_design": EXPERIMENTAL_DESIGN_PROMPTS,
        "presentation_quality": PRESENTATION_QUALITY_PROMPTS,
    }