"""
Research Robustness Validators

This module provides validators to detect common weaknesses in AI-generated research papers
that lead to reviewer criticism. Based on systematic analysis of reviewer feedback patterns.
"""

import re
import logging
from typing import Dict, List, Tuple, Any
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

@dataclass
class ValidationResult:
    """Result of a validation check"""
    passed: bool
    score: float  # 0-1, higher is better
    issues: List[str]
    suggestions: List[str]
    severity: str  # 'low', 'medium', 'high', 'critical'

class AssumptionRobustnessValidator:
    """
    Validates that papers adequately address the robustness of their core assumptions.
    
    Common issues detected:
    - Strong assumptions without sensitivity analysis
    - Idealized conditions without realistic constraints
    - Missing discussion of assumption limitations
    - No robustness or deviation studies
    """
    
    def __init__(self):
        # Keywords indicating strong assumptions
        self.assumption_keywords = [
            'assume', 'assuming', 'assumption', 'suppose', 'let us assume',
            'for simplicity', 'idealized', 'perfect', 'typical', 'random',
            'uniform', 'generic', 'approximate', 'simplified'
        ]
        
        # Keywords indicating robustness analysis
        self.robustness_keywords = [
            'robust', 'robustness', 'sensitivity', 'deviation', 'perturbation',
            'stability', 'error analysis', 'tolerance', 'realistic', 'practical',
            'finite', 'approximate', 'bounded', 'convergence', 'validation'
        ]
        
        # Keywords for limitation discussions
        self.limitation_keywords = [
            'limitation', 'constraint', 'challenge', 'difficulty', 'issue',
            'problem', 'caveat', 'restriction', 'bound', 'approximation'
        ]

    def validate(self, paper_text: str) -> ValidationResult:
        """Validate assumption robustness in paper"""
        issues = []
        suggestions = []
        
        # Count assumption statements
        assumption_count = self._count_patterns(paper_text, self.assumption_keywords)
        robustness_count = self._count_patterns(paper_text, self.robustness_keywords)
        limitation_count = self._count_patterns(paper_text, self.limitation_keywords)
        
        # Analyze assumption-to-robustness ratio
        if assumption_count > 5:
            ratio = robustness_count / assumption_count if assumption_count > 0 else 0
            
            if ratio < 0.3:
                issues.append(f"High assumption usage ({assumption_count}) with insufficient robustness analysis ({robustness_count})")
                suggestions.append("Add sensitivity analysis for key assumptions")
                suggestions.append("Include discussion of realistic deviations from idealized conditions")
            
            if ratio < 0.1:
                issues.append("Critical lack of assumption validation")
                suggestions.append("Perform quantitative robustness studies")
        
        # Check for specific problematic patterns
        self._check_idealization_patterns(paper_text, issues, suggestions)
        self._check_assumption_sections(paper_text, issues, suggestions)
        
        # Calculate score
        score = min(1.0, (robustness_count + limitation_count) / max(1, assumption_count))
        severity = self._determine_severity(score, len(issues))
        
        return ValidationResult(
            passed=score > 0.4,
            score=score,
            issues=issues,
            suggestions=suggestions,
            severity=severity
        )
    
    def _count_patterns(self, text: str, keywords: List[str]) -> int:
        """Count occurrences of keyword patterns"""
        count = 0
        text_lower = text.lower()
        for keyword in keywords:
            count += len(re.findall(rf'\b{re.escape(keyword)}\b', text_lower))
        return count
    
    def _check_idealization_patterns(self, text: str, issues: List[str], suggestions: List[str]):
        """Check for problematic idealization patterns"""
        problematic_patterns = [
            (r'assume.*?(?:perfect|ideal|infinite|exact)', 
             "Strong idealization without discussing realistic constraints"),
            (r'for simplicity.*?assume', 
             "Simplifying assumptions may need robustness validation"),
            (r'typical.*?(?:random|generic)', 
             "Typicality assumptions should be validated against realistic conditions")
        ]
        
        for pattern, issue in problematic_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                issues.append(issue)
                suggestions.append("Add analysis of how results change under realistic conditions")
    
    def _check_assumption_sections(self, text: str, issues: List[str], suggestions: List[str]):
        """Check if paper has dedicated sections for assumption analysis"""
        has_robustness_section = bool(re.search(r'section.*?(?:robust|sensitivity|validation)', text, re.IGNORECASE))
        has_limitation_section = bool(re.search(r'section.*?(?:limitation|discussion|future)', text, re.IGNORECASE))
        
        if not has_robustness_section:
            issues.append("No dedicated section for robustness/sensitivity analysis")
            suggestions.append("Add a robustness analysis section")
        
        if not has_limitation_section:
            issues.append("No dedicated section discussing limitations")
            suggestions.append("Add a limitations/discussion section")
    
    def _determine_severity(self, score: float, issue_count: int) -> str:
        """Determine severity based on score and issue count"""
        if score < 0.2 or issue_count > 5:
            return 'critical'
        elif score < 0.4 or issue_count > 3:
            return 'high'
        elif score < 0.6 or issue_count > 1:
            return 'medium'
        else:
            return 'low'


class MicroscopicDerivationValidator:
    """
    Validates that papers provide adequate microscopic foundations for their results.
    
    Common issues detected:
    - Missing first-principles derivations
    - Lack of microscopic models
    - Insufficient theoretical foundations
    - Over-reliance on phenomenological models
    """
    
    def __init__(self):
        # Keywords indicating microscopic/fundamental approaches
        self.microscopic_keywords = [
            'microscopic', 'fundamental', 'first principles', 'ab initio',
            'from scratch', 'derive', 'derivation', 'proof', 'theorem',
            'lemma', 'theoretical framework', 'mathematical foundation'
        ]
        
        # Keywords indicating phenomenological approaches
        self.phenomenological_keywords = [
            'phenomenological', 'effective', 'model', 'ansatz', 'heuristic',
            'approximate', 'simplified', 'toy model', 'schematic', 'effective theory'
        ]
        
        # Keywords for mathematical rigor
        self.rigor_keywords = [
            'proof', 'theorem', 'lemma', 'corollary', 'proposition',
            'rigorous', 'exact', 'formal', 'mathematical', 'derivation'
        ]

    def validate(self, paper_text: str) -> ValidationResult:
        """Validate microscopic derivation depth"""
        issues = []
        suggestions = []
        
        # Count different types of approaches
        microscopic_count = self._count_patterns(paper_text, self.microscopic_keywords)
        phenomenological_count = self._count_patterns(paper_text, self.phenomenological_keywords)
        rigor_count = self._count_patterns(paper_text, self.rigor_keywords)
        
        # Check for imbalance toward phenomenological approaches
        total_approach = microscopic_count + phenomenological_count
        if total_approach > 0:
            pheno_ratio = phenomenological_count / total_approach
            if pheno_ratio > 0.7:
                issues.append("Heavy reliance on phenomenological models without microscopic foundations")
                suggestions.append("Provide first-principles derivations for key results")
                suggestions.append("Connect phenomenological models to underlying microscopic theory")
        
        # Check for mathematical rigor
        if rigor_count < 3:
            issues.append("Insufficient mathematical rigor (few proofs/theorems)")
            suggestions.append("Add formal proofs for main theoretical claims")
            suggestions.append("Include rigorous mathematical derivations")
        
        # Check for specific derivation patterns
        self._check_derivation_quality(paper_text, issues, suggestions)
        self._check_theoretical_foundation(paper_text, issues, suggestions)
        
        # Calculate score
        micro_score = min(1.0, microscopic_count / 5)
        rigor_score = min(1.0, rigor_count / 3)
        pheno_penalty = max(0, (phenomenological_count - microscopic_count) / 10)
        score = max(0, (micro_score + rigor_score) / 2 - pheno_penalty)
        
        severity = self._determine_severity(score, len(issues))
        
        return ValidationResult(
            passed=score > 0.5,
            score=score,
            issues=issues,
            suggestions=suggestions,
            severity=severity
        )
    
    def _count_patterns(self, text: str, keywords: List[str]) -> int:
        """Count occurrences of keyword patterns"""
        count = 0
        text_lower = text.lower()
        for keyword in keywords:
            count += len(re.findall(rf'\b{re.escape(keyword)}\b', text_lower))
        return count
    
    def _check_derivation_quality(self, text: str, issues: List[str], suggestions: List[str]):
        """Check quality of derivations"""
        # Look for step-by-step derivations
        has_derivation_steps = bool(re.search(r'(?:step|equation|derive).*?(?:\d+|next|then)', text, re.IGNORECASE))
        
        if not has_derivation_steps:
            issues.append("Lack of detailed step-by-step derivations")
            suggestions.append("Provide complete derivation steps for main equations")
        
        # Check for missing fundamental connections
        fundamental_patterns = [
            r'without.*?derivation',
            r'it can be shown',
            r'straightforward.*?calculation',
            r'standard.*?result'
        ]
        
        for pattern in fundamental_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                issues.append("Key results presented without adequate derivation")
                suggestions.append("Provide explicit derivations for claimed results")
    
    def _check_theoretical_foundation(self, text: str, issues: List[str], suggestions: List[str]):
        """Check strength of theoretical foundations"""
        # Look for theoretical foundations
        theoretical_foundations = [
            'theoretical framework', 'mathematical foundation', 'established theory',
            'fundamental principles', 'scientific method', 'rigorous approach',
            'systematic analysis', 'formal treatment', 'comprehensive theory'
        ]
        
        foundation_count = sum(1 for term in theoretical_foundations 
                             if term in text.lower())
        
        if foundation_count < 2:
            issues.append("Weak connection to established theoretical principles")
            suggestions.append("Ground results in established theoretical frameworks")
    
    def _determine_severity(self, score: float, issue_count: int) -> str:
        """Determine severity based on score and issue count"""
        if score < 0.3 or issue_count > 4:
            return 'critical'
        elif score < 0.5 or issue_count > 2:
            return 'high'
        elif score < 0.7 or issue_count > 1:
            return 'medium'
        else:
            return 'low'


class ObservationalFeasibilityValidator:
    """
    Validates that papers adequately address experimental/observational feasibility.
    
    Common issues detected:
    - Effects too small to observe
    - No experimental predictions
    - Missing measurement feasibility analysis
    - No alternative testing approaches
    """
    
    def __init__(self):
        # Keywords for experimental/observational content
        self.experimental_keywords = [
            'experiment', 'observation', 'measure', 'measurement', 'detect',
            'detection', 'observable', 'signature', 'signal', 'test', 'verify',
            'validation', 'empirical', 'data', 'evidence'
        ]
        
        # Keywords for feasibility analysis
        self.feasibility_keywords = [
            'feasible', 'practical', 'realistic', 'achievable', 'accessible',
            'challenging', 'difficult', 'impossible', 'scale', 'scaling',
            'sensitivity', 'precision', 'accuracy', 'resolution'
        ]
        
        # Keywords indicating scale problems
        self.scale_keywords = [
            'small', 'tiny', 'negligible', 'suppressed', 'weak', 'subtle',
            'order of magnitude', 'exponentially small', 'fundamental scale'
        ]

    def validate(self, paper_text: str) -> ValidationResult:
        """Validate observational feasibility"""
        issues = []
        suggestions = []
        
        # Count experimental content
        exp_count = self._count_patterns(paper_text, self.experimental_keywords)
        feasibility_count = self._count_patterns(paper_text, self.feasibility_keywords)
        scale_issue_count = self._count_patterns(paper_text, self.scale_keywords)
        
        # Check for adequate experimental discussion
        total_length = len(paper_text.split())
        exp_ratio = exp_count / max(1, total_length / 1000)  # per 1000 words
        
        if exp_ratio < 5:
            issues.append("Insufficient discussion of experimental/observational aspects")
            suggestions.append("Add section on experimental predictions and tests")
            suggestions.append("Discuss measurement strategies and requirements")
        
        # Check for scale/feasibility issues
        if scale_issue_count > 3 and feasibility_count < 2:
            issues.append("Mentions small/suppressed effects without feasibility analysis")
            suggestions.append("Analyze measurement sensitivity requirements")
            suggestions.append("Propose alternative experimental approaches")
        
        # Check for specific observational problems
        self._check_observational_challenges(paper_text, issues, suggestions)
        self._check_alternative_approaches(paper_text, issues, suggestions)
        
        # Calculate score
        exp_score = min(1.0, exp_ratio / 10)
        feasibility_score = min(1.0, feasibility_count / 5)
        scale_penalty = min(0.5, scale_issue_count / 10)
        score = max(0, (exp_score + feasibility_score) - scale_penalty)
        
        severity = self._determine_severity(score, len(issues))
        
        return ValidationResult(
            passed=score > 0.4,
            score=score,
            issues=issues,
            suggestions=suggestions,
            severity=severity
        )
    
    def _count_patterns(self, text: str, keywords: List[str]) -> int:
        """Count occurrences of keyword patterns"""
        count = 0
        text_lower = text.lower()
        for keyword in keywords:
            count += len(re.findall(rf'\b{re.escape(keyword)}\b', text_lower))
        return count
    
    def _check_observational_challenges(self, text: str, issues: List[str], suggestions: List[str]):
        """Check for unaddressed observational challenges"""
        challenge_patterns = [
            (r'suppressed.*?by', "Effects with suppression factors without feasibility discussion"),
            (r'exponentially.*?small', "Exponentially small effects without measurement strategy"),
            (r'(?:planck|fundamental).*?scale', "Fundamental scale physics without observational pathway"),
            (r'extremely.*?(?:small|weak|subtle)', "Extremely weak effects need feasibility analysis")
        ]
        
        for pattern, issue in challenge_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                issues.append(issue)
                suggestions.append("Discuss measurement precision requirements")
                suggestions.append("Propose enhanced sensitivity techniques")
    
    def _check_alternative_approaches(self, text: str, issues: List[str], suggestions: List[str]):
        """Check for alternative experimental approaches"""
        has_analogues = bool(re.search(r'analogue.*?system', text, re.IGNORECASE))
        has_alternatives = bool(re.search(r'alternative.*?(?:test|experiment|approach)', text, re.IGNORECASE))
        has_indirect = bool(re.search(r'indirect.*?(?:measurement|signature)', text, re.IGNORECASE))
        
        if not (has_analogues or has_alternatives or has_indirect):
            issues.append("No discussion of alternative experimental approaches")
            suggestions.append("Consider analogue systems or simplified models")
            suggestions.append("Discuss indirect measurement strategies")
    
    def _determine_severity(self, score: float, issue_count: int) -> str:
        """Determine severity based on score and issue count"""
        if score < 0.2 or issue_count > 4:
            return 'critical'
        elif score < 0.4 or issue_count > 2:
            return 'high'
        elif score < 0.6 or issue_count > 1:
            return 'medium'
        else:
            return 'low'


class ComputationalLimitationValidator:
    """
    Validates that papers adequately address computational and approximation limitations.
    
    Common issues detected:
    - Missing convergence analysis
    - Unvalidated approximations
    - No discussion of computational bounds
    - Insufficient error analysis
    """
    
    def __init__(self):
        # Keywords for computational methods
        self.computational_keywords = [
            'numerical', 'simulation', 'computation', 'algorithm', 'monte carlo',
            'finite', 'discretize', 'approximate', 'truncate', 'iteration'
        ]
        
        # Keywords for limitation analysis
        self.limitation_keywords = [
            'convergence', 'error', 'accuracy', 'precision', 'bound', 'limit',
            'truncation', 'approximation', 'finite size', 'systematic error',
            'numerical error', 'stability', 'validation'
        ]
        
        # Keywords for approximation methods
        self.approximation_keywords = [
            'cutoff', 'finite', 'truncated', 'approximate',
            'perturbative', 'leading order', 'lowest order', 'first order'
        ]

    def validate(self, paper_text: str) -> ValidationResult:
        """Validate computational limitation analysis"""
        issues = []
        suggestions = []
        
        # Count computational content
        comp_count = self._count_patterns(paper_text, self.computational_keywords)
        limit_count = self._count_patterns(paper_text, self.limitation_keywords)
        approx_count = self._count_patterns(paper_text, self.approximation_keywords)
        
        # Check balance between methods and limitations
        if comp_count > 5:
            limit_ratio = limit_count / comp_count
            if limit_ratio < 0.3:
                issues.append("Heavy computational content without adequate limitation analysis")
                suggestions.append("Add convergence analysis for computational methods")
                suggestions.append("Discuss systematic errors and approximation bounds")
        
        # Check for approximation validation
        if approx_count > 3 and limit_count < 2:
            issues.append("Multiple approximations without validation studies")
            suggestions.append("Validate key approximations with exact results where possible")
            suggestions.append("Quantify approximation errors")
        
        # Check for specific computational issues
        self._check_convergence_analysis(paper_text, issues, suggestions)
        self._check_error_bounds(paper_text, issues, suggestions)
        
        # Calculate score
        if comp_count == 0:
            score = 1.0  # No computational methods, no issues
        else:
            comp_score = min(1.0, limit_count / max(1, comp_count))
            validation_score = self._assess_validation_quality(paper_text)
            score = (comp_score + validation_score) / 2
        
        severity = self._determine_severity(score, len(issues))
        
        return ValidationResult(
            passed=score > 0.5,
            score=score,
            issues=issues,
            suggestions=suggestions,
            severity=severity
        )
    
    def _count_patterns(self, text: str, keywords: List[str]) -> int:
        """Count occurrences of keyword patterns"""
        count = 0
        text_lower = text.lower()
        for keyword in keywords:
            count += len(re.findall(rf'\b{re.escape(keyword)}\b', text_lower))
        return count
    
    def _check_convergence_analysis(self, text: str, issues: List[str], suggestions: List[str]):
        """Check for convergence analysis"""
        has_convergence = bool(re.search(r'convergen[ct]e.*?(?:analysis|study|test)', text, re.IGNORECASE))
        has_scaling = bool(re.search(r'scaling.*?(?:analysis|behavior|law)', text, re.IGNORECASE))
        
        if not (has_convergence or has_scaling):
            issues.append("No convergence analysis for computational methods")
            suggestions.append("Include convergence tests with varying parameters")
            suggestions.append("Study scaling behavior of computational methods")
    
    def _check_error_bounds(self, text: str, issues: List[str], suggestions: List[str]):
        """Check for error bound analysis"""
        has_error_bounds = bool(re.search(r'error.*?bound', text, re.IGNORECASE))
        has_uncertainty = bool(re.search(r'uncertain[ty].*?(?:analysis|quantification)', text, re.IGNORECASE))
        
        if not (has_error_bounds or has_uncertainty):
            issues.append("No quantitative error bounds for approximations")
            suggestions.append("Provide error bounds for key approximations")
            suggestions.append("Quantify uncertainty in computational results")
    
    def _assess_validation_quality(self, text: str) -> float:
        """Assess quality of validation studies"""
        validation_indicators = [
            r'benchmark.*?(?:test|comparison)',
            r'exact.*?(?:result|solution)',
            r'analytical.*?comparison',
            r'independent.*?validation',
            r'cross.*?check'
        ]
        
        score = 0
        for indicator in validation_indicators:
            if re.search(indicator, text, re.IGNORECASE):
                score += 0.2
        
        return min(1.0, score)
    
    def _determine_severity(self, score: float, issue_count: int) -> str:
        """Determine severity based on score and issue count"""
        if score < 0.3 or issue_count > 3:
            return 'critical'
        elif score < 0.5 or issue_count > 2:
            return 'high'
        elif score < 0.7 or issue_count > 1:
            return 'medium'
        else:
            return 'low'


class ComprehensiveResearchValidator:
    """
    Comprehensive validator that combines all individual validators for complete research quality assessment.
    """
    
    def __init__(self):
        self.validators = {
            'assumption_robustness': AssumptionRobustnessValidator(),
            'microscopic_derivation': MicroscopicDerivationValidator(),
            'observational_feasibility': ObservationalFeasibilityValidator(),
            'computational_limitation': ComputationalLimitationValidator()
        }
    
    def validate_paper(self, paper_text: str) -> Dict[str, ValidationResult]:
        """Run all validators on paper text"""
        results = {}
        
        for name, validator in self.validators.items():
            try:
                results[name] = validator.validate(paper_text)
                logger.info(f"Completed {name} validation")
            except Exception as e:
                logger.error(f"Error in {name} validation: {e}")
                results[name] = ValidationResult(
                    passed=False,
                    score=0.0,
                    issues=[f"Validation error: {str(e)}"],
                    suggestions=["Review validator implementation"],
                    severity='critical'
                )
        
        return results
    
    def generate_report(self, results: Dict[str, ValidationResult]) -> str:
        """Generate comprehensive validation report"""
        report = []
        report.append("=" * 60)
        report.append("COMPREHENSIVE RESEARCH QUALITY VALIDATION REPORT")
        report.append("=" * 60)
        report.append("")
        
        # Overall summary
        total_score = sum(result.score for result in results.values()) / len(results)
        total_issues = sum(len(result.issues) for result in results.values())
        
        report.append(f"OVERALL SCORE: {total_score:.2f}/1.00")
        report.append(f"TOTAL ISSUES FOUND: {total_issues}")
        report.append("")
        
        # Individual validator results
        for name, result in results.items():
            report.append(f"{name.upper().replace('_', ' ')} VALIDATION")
            report.append("-" * 40)
            report.append(f"Score: {result.score:.2f}/1.00")
            report.append(f"Status: {'PASSED' if result.passed else 'FAILED'}")
            report.append(f"Severity: {result.severity.upper()}")
            
            if result.issues:
                report.append("\nIssues Found:")
                for i, issue in enumerate(result.issues, 1):
                    report.append(f"  {i}. {issue}")
            
            if result.suggestions:
                report.append("\nSuggestions:")
                for i, suggestion in enumerate(result.suggestions, 1):
                    report.append(f"  {i}. {suggestion}")
            
            report.append("")
        
        # Priority recommendations
        critical_issues = [(name, result) for name, result in results.items() 
                          if result.severity == 'critical']
        
        if critical_issues:
            report.append("CRITICAL ISSUES REQUIRING IMMEDIATE ATTENTION")
            report.append("=" * 50)
            for name, result in critical_issues:
                report.append(f"• {name.replace('_', ' ').title()}: {result.issues[0] if result.issues else 'Critical validation failure'}")
            report.append("")
        
        return "\n".join(report)
    
    def validate_paper_file(self, paper_path: str) -> Dict[str, ValidationResult]:
        """Validate paper from file path"""
        try:
            with open(paper_path, 'r', encoding='utf-8') as f:
                paper_text = f.read()
            return self.validate_paper(paper_text)
        except Exception as e:
            logger.error(f"Error reading paper file {paper_path}: {e}")
            return {}


# Example usage and testing
if __name__ == "__main__":
    # Example paper text with various issues
    example_paper = """
    We assume that the interactions are random, which is a strong idealization.
    The system is perfectly isolated and we suppose thermal equilibrium.
    For simplicity, we assume infinite precision in our calculations.
    
    The effects are suppressed by a large factor which makes observation challenging.
    Our numerical simulations use finite cutoff parameter χ = 100.
    The truncation errors are assumed to be negligible.
    
    We use effective field theory without deriving from first principles.
    The phenomenological model captures the essential physics.
    """
    
    # Test comprehensive validator
    validator = ComprehensiveResearchValidator()
    results = validator.validate_paper(example_paper)
    report = validator.generate_report(results)
    print(report)