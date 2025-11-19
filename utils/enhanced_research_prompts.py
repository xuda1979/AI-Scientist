"""
Enhanced Research Quality Prompts

This module provides enhanced prompts that systematically address common research weaknesses
identified through reviewer feedback analysis. These prompts enforce robustness, depth,
and experimental feasibility in AI-generated research papers.
"""

ROBUSTNESS_REQUIREMENTS = """

=== ASSUMPTION ROBUSTNESS REQUIREMENTS ===

Your paper must include comprehensive analysis of all assumptions:

1. **ASSUMPTION IDENTIFICATION**: For every major assumption, explicitly state:
   - What assumption is being made
   - Why it's necessary for the analysis
   - How realistic/unrealistic it is

2. **SENSITIVITY ANALYSIS**: For each key assumption, provide:
   - Quantitative analysis of how results change when assumption is relaxed
   - Bounds on how much assumption can be violated while preserving conclusions
   - Discussion of what happens in realistic scenarios

3. **ROBUSTNESS VALIDATION**: Include dedicated sections on:
   - Robustness to parameter variations
   - Stability under perturbations
   - Sensitivity to model choices
   - Performance under realistic conditions

4. **LIMITATION ACKNOWLEDGMENT**: Explicitly discuss:
   - Where assumptions break down
   - Parameter regimes where analysis fails
   - Realistic constraints that may limit applicability
   - Future work needed to address assumption limitations

CRITICAL: Do not make strong assumptions (perfect isolation, infinite precision, 
typical behavior, idealized conditions) without dedicated robustness analysis. Every "assume", "suppose", 
"for simplicity", "idealized" statement must be followed by sensitivity discussion.
"""

MICROSCOPIC_DERIVATION_REQUIREMENTS = """

=== MICROSCOPIC DERIVATION REQUIREMENTS ===

Your paper must provide rigorous theoretical foundations:

1. **FIRST-PRINCIPLES DERIVATIONS**: For all major results, provide:
   - Step-by-step derivations from fundamental principles
   - Clear mathematical progression with all steps shown
   - Connection to established theoretical frameworks and foundational principles
   - No "it can be shown" or "straightforward calculation" without details

2. **MATHEMATICAL RIGOR**: Include:
   - Formal theorems and proofs for theoretical claims
   - Rigorous mathematical definitions
   - Clear statement of assumptions and their necessity
   - Proper mathematical notation and structure

3. **THEORETICAL FOUNDATIONS**: Connect to fundamental principles:
   - Derive effective theories from underlying models
   - Show connection to established theoretical frameworks where relevant
   - Provide rigorous mathematical formulations
   - Ground phenomenological models in theoretical principles

4. **THEORETICAL COMPLETENESS**: Ensure:
   - No gaps in logical progression
   - All mathematical steps are justified
   - Approximations are clearly identified and bounded
   - Alternative derivation approaches are considered

CRITICAL: Avoid pure phenomenological approaches without microscopic justification.
Every effective model must be connected to underlying fundamental theory.
"""

EXPERIMENTAL_FEASIBILITY_REQUIREMENTS = """

=== EXPERIMENTAL FEASIBILITY REQUIREMENTS ===

Your paper must thoroughly address experimental/observational aspects:

1. **EXPERIMENTAL PREDICTIONS**: Provide:
   - Clear, testable predictions with specific numbers
   - Observable signatures that distinguish your theory
   - Scaling laws and parameter dependencies
   - Comparison with existing experimental bounds

2. **MEASUREMENT FEASIBILITY**: Analyze:
   - Required experimental precision and sensitivity
   - Current technological limitations and capabilities
   - Signal-to-noise ratio estimates
   - Systematic error considerations

3. **SCALE ANALYSIS**: For small effects, discuss:
   - Quantitative estimates of effect sizes
   - Enhancement mechanisms or sensitive observables
   - Cumulative effects or resonance conditions
   - Statistical significance requirements

4. **ALTERNATIVE APPROACHES**: Consider:
   - Analogue systems and simplified experimental setups
   - Indirect measurement strategies
   - Large-scale and systematic observables
   - High-precision measurement techniques

5. **PRACTICAL IMPLEMENTATION**: Address:
   - Experimental setups and methodologies
   - Required technologies and timescales
   - Cost and feasibility considerations
   - Collaboration requirements

CRITICAL: If effects are small or suppressed (exponentially small, fundamental-scale),
you MUST provide detailed analysis of measurement strategies and alternative approaches.
"""

COMPUTATIONAL_LIMITATION_REQUIREMENTS = """

=== COMPUTATIONAL LIMITATION REQUIREMENTS ===

Your paper must rigorously address all computational and approximation limitations:

1. **CONVERGENCE ANALYSIS**: For all numerical methods, provide:
   - Systematic convergence tests with varying parameters
   - Scaling behavior analysis
   - Extrapolation to infinite precision/size limits
   - Error estimates as function of computational parameters

2. **APPROXIMATION VALIDATION**: For each approximation, include:
   - Quantitative error bounds
   - Comparison with exact results where available
   - Parameter regimes where approximation is valid
   - Systematic improvement strategies

3. **COMPUTATIONAL BOUNDS**: Discuss:
   - Fundamental computational limitations
   - Memory and time complexity scaling
   - Finite-size effects and boundary conditions
   - Discretization errors and their control

4. **VALIDATION STUDIES**: Provide:
   - Benchmarking against known exact results
   - Cross-validation with independent methods
   - Consistency checks and sanity tests
   - Comparison with experimental data where available

5. **SYSTEMATIC ERRORS**: Analyze:
   - All sources of systematic uncertainty
   - Propagation of errors through calculations
   - Statistical vs systematic error separation
   - Bias correction and uncertainty quantification

CRITICAL: Every finite parameter, cutoff, truncation, or approximation
must be validated with convergence studies and error bounds.
"""

COMPREHENSIVE_QUALITY_PROMPT = f"""
You are tasked with generating a high-quality research paper that meets rigorous academic standards. 
Your paper will be automatically validated against common research weaknesses that lead to reviewer criticism.

{ROBUSTNESS_REQUIREMENTS}

{MICROSCOPIC_DERIVATION_REQUIREMENTS}

{EXPERIMENTAL_FEASIBILITY_REQUIREMENTS}

{COMPUTATIONAL_LIMITATION_REQUIREMENTS}

=== OVERALL STRUCTURE REQUIREMENTS ===

Your paper must include these sections with substantial content:

1. **INTRODUCTION** (500+ words): Clear motivation, context, and contribution
2. **THEORETICAL FOUNDATION** (800+ words): Rigorous derivations from first principles
3. **METHODOLOGY** (600+ words): Detailed methods with validation
4. **RESULTS** (700+ words): Comprehensive analysis with uncertainty quantification
5. **ROBUSTNESS ANALYSIS** (400+ words): Dedicated section on assumption sensitivity
6. **EXPERIMENTAL PROSPECTS** (400+ words): Dedicated section on observational feasibility
7. **LIMITATIONS AND FUTURE WORK** (300+ words): Honest assessment of limitations
8. **CONCLUSION** (200+ words): Clear summary and implications

=== QUALITY METRICS ===

Your paper will be scored on:
- Assumption robustness: Must score >0.6/1.0 (sufficient sensitivity analysis)
- Microscopic derivations: Must score >0.7/1.0 (rigorous theoretical foundation)
- Experimental feasibility: Must score >0.6/1.0 (adequate experimental discussion)
- Computational limitations: Must score >0.7/1.0 (proper validation and error analysis)

=== FORBIDDEN PATTERNS ===

DO NOT include these problematic patterns:
- "For simplicity, we assume..." without robustness analysis
- "It can be shown that..." without showing the derivation
- "The effect is small but..." without feasibility discussion
- "We use finite [parameter]..." without convergence analysis
- "Typical" or "generic" behavior without validation
- "Phenomenological model" without microscopic connection

Your paper should be publication-ready with rigorous theoretical foundations,
comprehensive validation, and clear experimental prospects.
"""

def get_enhanced_paper_prompt(paper_type="theoretical", specific_requirements=""):
    """
    Get enhanced prompt for paper generation with quality requirements.
    
    Args:
        paper_type: Type of paper ("theoretical", "computational", "experimental")
        specific_requirements: Additional domain-specific requirements
    
    Returns:
        Enhanced prompt string with quality requirements
    """
    
    base_prompt = COMPREHENSIVE_QUALITY_PROMPT
    
    # Add paper-type specific requirements
    if paper_type == "computational":
        base_prompt += """
        
=== COMPUTATIONAL PAPER SPECIFIC REQUIREMENTS ===
- Include detailed algorithm descriptions with complexity analysis
- Provide convergence proofs or empirical convergence studies
- Compare multiple computational approaches
- Include performance benchmarking and scaling analysis
- Validate against analytical results where possible
        """
    
    elif paper_type == "experimental":
        base_prompt += """
        
=== EXPERIMENTAL PAPER SPECIFIC REQUIREMENTS ===
- Include detailed experimental setup descriptions
- Provide error analysis and systematic uncertainty quantification
- Compare with theoretical predictions
- Include statistical analysis and significance testing
- Discuss reproducibility and replication considerations
        """
    
    elif paper_type == "theoretical":
        base_prompt += """
        
=== THEORETICAL PAPER SPECIFIC REQUIREMENTS ===
- Provide complete mathematical derivations
- Include formal proofs for major theoretical claims
- Connect to established theoretical frameworks
- Discuss physical interpretation and implications
- Address mathematical consistency and completeness
        """
    
    # Add specific requirements if provided
    if specific_requirements:
        base_prompt += f"""
        
=== DOMAIN-SPECIFIC REQUIREMENTS ===
{specific_requirements}
        """
    
    return base_prompt

def get_revision_prompt(validation_results):
    """
    Generate revision prompt based on validation results.
    
    Args:
        validation_results: Dict of ValidationResult objects
    
    Returns:
        Targeted revision prompt addressing specific weaknesses
    """
    
    revision_prompt = """
Based on quality validation, your paper has been identified as needing improvements in the following areas:

"""
    
    for validator_name, result in validation_results.items():
        if not result.passed:
            revision_prompt += f"\n=== {validator_name.upper().replace('_', ' ')} ISSUES ===\n"
            revision_prompt += f"Score: {result.score:.2f}/1.00 (Target: >0.6)\n"
            revision_prompt += f"Severity: {result.severity.upper()}\n\n"
            
            if result.issues:
                revision_prompt += "Issues to address:\n"
                for i, issue in enumerate(result.issues, 1):
                    revision_prompt += f"{i}. {issue}\n"
                revision_prompt += "\n"
            
            if result.suggestions:
                revision_prompt += "Required improvements:\n"
                for i, suggestion in enumerate(result.suggestions, 1):
                    revision_prompt += f"{i}. {suggestion}\n"
                revision_prompt += "\n"
    
    revision_prompt += """
Please revise your paper to address ALL the above issues. Focus particularly on:
1. Adding missing sections (robustness analysis, experimental prospects, limitations)
2. Providing detailed derivations and proofs
3. Including sensitivity analysis for assumptions
4. Discussing experimental feasibility and measurement requirements
5. Validating computational methods with convergence studies

Your revised paper will be re-evaluated against these quality metrics.
"""
    
    return revision_prompt

# Example usage
if __name__ == "__main__":
    # Test prompt generation
    print("=== THEORETICAL PAPER PROMPT ===")
    print(get_enhanced_paper_prompt("theoretical"))
    
    print("\n" + "="*60 + "\n")
    
    # Test revision prompt
    from research_robustness_validators import ValidationResult
    
    mock_results = {
        'assumption_robustness': ValidationResult(
            passed=False,
            score=0.3,
            issues=["Heavy assumption usage without robustness analysis"],
            suggestions=["Add sensitivity analysis", "Include robustness section"],
            severity='high'
        )
    }
    
    print("=== REVISION PROMPT ===")
    print(get_revision_prompt(mock_results))