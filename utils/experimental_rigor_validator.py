"""
Experimental Rigor Validator

This module detects common weaknesses in research papers that lead to poor reviews,
specifically addressing issues like:
1. Reliance on synthetic-only evaluation
2. Unvalidated assumptions
3. Missing complexity analysis
4. Excessive algorithms without empirical validation
5. Poor calibration and missing real-world applicability
"""

import re
from typing import List, Tuple, Optional, Dict
from pathlib import Path


def detect_synthetic_only_evaluation(paper_content: str, sim_summary: str) -> List[str]:
    """
    Detect if paper relies entirely on synthetic evaluation without real-world validation.
    
    Returns:
        List of issues found
    """
    issues = []
    
    # Check for synthetic/toy dataset indicators
    synthetic_indicators = [
        r'synthetic\s+(?:data|dataset|benchmark|evaluation|experiment)',
        r'toy\s+(?:problem|example|dataset)',
        r'simulated\s+(?:data|environment|scenario)',
        r'randomly\s+generated\s+(?:data|instances|problems)',
        r'artificial\s+(?:data|dataset|benchmark)',
    ]
    
    synthetic_count = 0
    for pattern in synthetic_indicators:
        synthetic_count += len(re.findall(pattern, paper_content, re.IGNORECASE))
    
    # Check for real-world dataset indicators
    real_world_indicators = [
        r'real-world\s+(?:data|dataset|benchmark|application|task)',
        r'(?:ImageNet|MNIST|CIFAR|COCO|SQuAD|GLUE|SuperGLUE|WikiText|Penn\s+Treebank)',
        r'standard\s+benchmark',
        r'public\s+dataset',
        r'real\s+(?:data|application|problem|task)',
    ]
    
    real_world_count = 0
    for pattern in real_world_indicators:
        real_world_count += len(re.findall(pattern, paper_content, re.IGNORECASE))
    
    # Issue if only synthetic evaluation found
    if synthetic_count > 2 and real_world_count == 0:
        issues.append(
            "CRITICAL: Paper appears to rely entirely on synthetic/toy evaluation. "
            "Add real-world benchmark evaluation to demonstrate practical applicability. "
            "Reviewers expect validation on standard benchmarks (e.g., ImageNet, GLUE, WikiText) "
            "to show the method works beyond idealized synthetic settings."
        )
    
    # Check if limitations section discusses synthetic evaluation
    has_limitations = bool(re.search(r'\\section\*?\{(?:Limitations|Discussion)\}', paper_content))
    if synthetic_count > 0 and has_limitations:
        limitations_section = re.search(
            r'\\section\*?\{(?:Limitations|Discussion)\}(.*?)(?:\\section|\\bibliography|\\end\{document\})',
            paper_content,
            re.DOTALL | re.IGNORECASE
        )
        if limitations_section:
            limitations_text = limitations_section.group(1)
            if 'synthetic' not in limitations_text.lower() and 'real-world' not in limitations_text.lower():
                issues.append(
                    "WEAKNESS: Limitations section does not acknowledge the use of synthetic evaluation "
                    "or discuss real-world applicability concerns. Reviewers expect honest discussion "
                    "of whether synthetic results will generalize."
                )
    
    return issues


def detect_unvalidated_assumptions(paper_content: str) -> List[str]:
    """
    Detect if paper makes critical assumptions without empirical validation.
    
    Returns:
        List of issues found
    """
    issues = []
    
    # Find assumption statements
    assumption_patterns = [
        r'we\s+assume\s+(?:that\s+)?([^.!?]{10,150})',
        r'under\s+the\s+assumption\s+(?:that\s+)?([^.!?]{10,150})',
        r'assuming\s+(?:that\s+)?([^.!?]{10,150})',
        r'\\textbf\{Assumption\s+\d+\}[:\s]*([^.!?]{10,150})',
        r'Assumption\s+\d+[:\s]*([^.!?]{10,150})',
    ]
    
    assumptions = []
    for pattern in assumption_patterns:
        matches = re.findall(pattern, paper_content, re.IGNORECASE)
        assumptions.extend(matches)
    
    if len(assumptions) > 2:
        # Check if there's an experimental validation section
        has_validation = bool(re.search(
            r'(?:validat|verif|test|empirical)(?:ing|e|ion)\s+(?:the\s+)?assumption',
            paper_content,
            re.IGNORECASE
        ))
        
        if not has_validation:
            issues.append(
                f"CRITICAL: Paper makes {len(assumptions)} explicit assumptions but does not appear "
                "to empirically validate them. Reviewers expect ablation studies or experiments "
                "testing whether key assumptions (e.g., submodularity, monotonicity) actually hold "
                "in practice. Add validation experiments or discuss assumption violations."
            )
    
    # Check for specific problematic assumptions
    submodularity_assumption = bool(re.search(
        r'(?:submodular|diminishing\s+returns|monotone)',
        paper_content,
        re.IGNORECASE
    ))
    
    if submodularity_assumption:
        has_submodularity_test = bool(re.search(
            r'(?:test|verif|measure|check).*?(?:submodular|diminishing\s+returns)',
            paper_content,
            re.IGNORECASE
        ))
        
        if not has_submodularity_test:
            issues.append(
                "WEAKNESS: Paper assumes submodularity/diminishing returns but does not empirically "
                "verify this holds. Reviewers will question whether 'Eureka moments' or sudden "
                "breakthroughs violate this assumption. Add experiments measuring submodularity "
                "or discuss when assumption may fail."
            )
    
    return issues


def detect_missing_complexity_analysis(paper_content: str, sim_summary: str) -> List[str]:
    """
    Detect if paper proposes algorithms without complexity/overhead analysis.
    
    Returns:
        List of issues found
    """
    issues = []
    
    # Count proposed algorithms
    algorithm_count = len(re.findall(r'\\begin\{algorithm\}', paper_content))
    algorithm_count += len(re.findall(r'Algorithm\s+\d+:', paper_content))
    
    # Check for complexity analysis
    has_complexity = bool(re.search(
        r'(?:time\s+complexity|space\s+complexity|computational\s+complexity|overhead|runtime\s+analysis)',
        paper_content,
        re.IGNORECASE
    ))
    
    has_big_o = bool(re.search(r'O\([^)]+\)', paper_content))
    
    if algorithm_count > 0 and not (has_complexity or has_big_o):
        issues.append(
            f"CRITICAL: Paper proposes {algorithm_count} algorithm(s) but lacks complexity analysis. "
            "Reviewers expect runtime/space complexity bounds (Big-O notation) and overhead "
            "measurements. Add theoretical complexity analysis and empirical overhead benchmarks "
            "comparing to baseline methods."
        )
    
    # Check for overhead measurements in experiments
    has_overhead_measurement = bool(re.search(
        r'(?:overhead|latency|inference\s+time|runtime|execution\s+time|wall-clock\s+time)',
        paper_content,
        re.IGNORECASE
    ))
    
    if algorithm_count > 0 and not has_overhead_measurement:
        issues.append(
            "WEAKNESS: No empirical overhead measurements found. Reviewers expect timing "
            "comparisons showing the computational cost of proposed methods vs. baselines, "
            "especially for low-latency applications. Add runtime benchmarks."
        )
    
    # Check if metareasoning/scheduling overhead is discussed
    has_metareasoning = bool(re.search(
        r'(?:metareasoning|scheduler|scheduling|compute\s+allocation)',
        paper_content,
        re.IGNORECASE
    ))
    
    if has_metareasoning and not has_overhead_measurement:
        issues.append(
            "CRITICAL: Paper proposes metareasoning/scheduling but does not measure the "
            "scheduler's own computational overhead. Reviewers will question if the "
            "metareasoning cost outweighs the benefits. Add overhead analysis of the "
            "scheduling mechanism itself."
        )
    
    return issues


def detect_algorithm_density_issues(paper_content: str) -> List[str]:
    """
    Detect if paper proposes too many algorithms without sufficient validation.
    
    Returns:
        List of issues found
    """
    issues = []
    
    # Count algorithms
    algorithm_count = len(re.findall(r'\\begin\{algorithm\}', paper_content))
    algorithm_count += len(re.findall(r'Algorithm\s+\d+:', paper_content))
    
    # Count named algorithms/methods
    named_methods = set()
    
    # Look for method names (e.g., "BPF", "MVC", "IGD")
    acronyms = re.findall(r'\b([A-Z]{3,})\b', paper_content)
    # Filter to likely algorithm names (mentioned multiple times)
    for acronym in acronyms:
        if acronyms.count(acronym) > 3 and acronym not in ['LLM', 'NLP', 'PDF', 'URL']:
            named_methods.add(acronym)
    
    total_methods = max(algorithm_count, len(named_methods))
    
    if total_methods > 5:
        # Check how many are empirically evaluated
        results_section = re.search(
            r'\\section\*?\{(?:Results|Experiments|Evaluation)\}(.*?)(?:\\section|\\bibliography|\\end\{document\})',
            paper_content,
            re.DOTALL | re.IGNORECASE
        )
        
        evaluated_methods = set()
        if results_section:
            results_text = results_section.group(1)
            # Count which methods appear in results
            for method in named_methods:
                if method in results_text:
                    evaluated_methods.add(method)
        
        evaluated_count = len(evaluated_methods)
        unevaluated_count = total_methods - evaluated_count
        
        if unevaluated_count > total_methods * 0.5:  # More than 50% unevaluated
            issues.append(
                f"CRITICAL: Paper proposes {total_methods} methods/algorithms but only "
                f"{evaluated_count} appear to be empirically evaluated. Reviewers will find "
                "this overwhelming and the analysis too shallow. Either reduce the number of "
                "proposed methods or provide comprehensive experimental validation for all. "
                "Consider moving some methods to appendix as future work."
            )
        
        if total_methods >= 8:
            issues.append(
                f"WEAKNESS: Excessive density - {total_methods} algorithms/methods in one paper. "
                "Reviewers may find this unfocused. Consider: (1) focusing on 2-3 core methods "
                "with deep analysis, or (2) framing as a survey/framework paper with different "
                "evaluation standards."
            )
    
    return issues


def detect_poor_calibration_issues(paper_content: str) -> List[str]:
    """
    Detect if paper relies on LLM calibration/self-evaluation without addressing known issues.
    
    Returns:
        List of issues found
    """
    issues = []
    
    # Check for reliance on self-evaluation or confidence
    calibration_indicators = [
        r'self-evaluation',
        r'confidence\s+(?:score|estimate|level)',
        r'expected\s+(?:value|improvement|gain)',
        r'(?:LLM|model)\s+calibration',
        r'posterior\s+(?:distribution|probability)',
    ]
    
    uses_calibration = False
    for pattern in calibration_indicators:
        if re.search(pattern, paper_content, re.IGNORECASE):
            uses_calibration = True
            break
    
    if uses_calibration:
        # Check if calibration challenges are discussed
        discusses_calibration = bool(re.search(
            r'(?:calibration|confidence).*?(?:challenge|problem|limitation|poor|inaccurate|unreliable)',
            paper_content,
            re.IGNORECASE | re.DOTALL
        ))
        
        if not discusses_calibration:
            issues.append(
                "CRITICAL: Paper relies on LLM confidence/self-evaluation but does not discuss "
                "calibration challenges. Reviewers know LLMs are poorly calibrated. You must: "
                "(1) acknowledge this limitation, (2) show empirical calibration analysis, or "
                "(3) describe calibration techniques used (e.g., temperature scaling, Platt scaling). "
                "Otherwise, reviewers will question if the whole method breaks with miscalibrated estimates."
            )
        
        # Check if calibration is measured
        measures_calibration = bool(re.search(
            r'(?:calibration\s+(?:error|metric|curve)|ECE|expected\s+calibration\s+error|Brier\s+score)',
            paper_content,
            re.IGNORECASE
        ))
        
        if not measures_calibration:
            issues.append(
                "WEAKNESS: Paper uses confidence/calibration but does not measure calibration quality. "
                "Add calibration metrics (ECE, calibration curves, Brier score) to validate that "
                "the model's confidence estimates are reliable."
            )
    
    return issues


def validate_experimental_rigor(
    paper_content: str,
    sim_summary: str,
    project_dir: Optional[Path] = None
) -> Tuple[List[str], List[str]]:
    """
    Comprehensive validation of experimental rigor.
    
    Returns:
        Tuple of (critical_issues, warnings)
    """
    critical_issues = []
    warnings = []
    
    # Run all validators
    synthetic_issues = detect_synthetic_only_evaluation(paper_content, sim_summary)
    assumption_issues = detect_unvalidated_assumptions(paper_content)
    complexity_issues = detect_missing_complexity_analysis(paper_content, sim_summary)
    density_issues = detect_algorithm_density_issues(paper_content)
    calibration_issues = detect_poor_calibration_issues(paper_content)
    
    # Categorize by severity
    all_issues = (
        synthetic_issues + 
        assumption_issues + 
        complexity_issues + 
        density_issues + 
        calibration_issues
    )
    
    for issue in all_issues:
        if issue.startswith("CRITICAL:"):
            critical_issues.append(issue)
        else:
            warnings.append(issue)
    
    return critical_issues, warnings


def generate_rigor_improvement_prompt(critical_issues: List[str], warnings: List[str]) -> str:
    """
    Generate a prompt section for improving experimental rigor.
    
    Returns:
        Formatted prompt text
    """
    if not critical_issues and not warnings:
        return ""
    
    prompt = "\n🔬 EXPERIMENTAL RIGOR REQUIREMENTS:\n"
    prompt += "Address the following experimental rigor issues to meet publication standards:\n\n"
    
    if critical_issues:
        prompt += "CRITICAL ISSUES (MUST FIX):\n"
        for i, issue in enumerate(critical_issues, 1):
            prompt += f"{i}. {issue}\n\n"
    
    if warnings:
        prompt += "WARNINGS (SHOULD ADDRESS):\n"
        for i, issue in enumerate(warnings, 1):
            prompt += f"{i}. {issue}\n\n"
    
    prompt += (
        "GENERAL EXPERIMENTAL RIGOR GUIDELINES:\n"
        "1. REAL-WORLD VALIDATION: Include experiments on standard benchmarks, not just synthetic data\n"
        "2. ASSUMPTION TESTING: Empirically validate key assumptions with ablation studies\n"
        "3. COMPLEXITY ANALYSIS: Provide Big-O notation and measure empirical overhead\n"
        "4. FOCUSED SCOPE: Limit to 2-4 core methods with deep evaluation, not 10+ shallow ones\n"
        "5. CALIBRATION AWARENESS: If using confidence/self-evaluation, measure calibration quality\n"
        "6. HONEST LIMITATIONS: Discuss when methods may fail or assumptions break\n"
        "7. STATISTICAL RIGOR: Report mean, std dev, significance tests across multiple runs/seeds\n"
        "8. REPRODUCIBILITY: Ensure all experiments can be reproduced with provided code\n\n"
    )
    
    return prompt
