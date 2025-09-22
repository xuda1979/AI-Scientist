#!/usr/bin/env python3
"""
Fair Paper Quality Scoring System

A comprehensive, realistic scoring system that provides accurate quality assessments
based on academic standards and reviewer expectations. This system is designed to
give lower, more realistic scores that better reflect actual paper quality.
"""

import re
import math
from typing import List, Dict, Optional, Tuple, Union
from dataclasses import dataclass
from pathlib import Path
import json


@dataclass
class QualityMetrics:
    """Container for all quality metrics and sub-scores."""
    # Core metrics (0.0 - 1.0)
    empirical_validation: float = 0.0
    figure_quality: float = 0.0  
    methodology_rigor: float = 0.0
    experimental_design: float = 0.0
    writing_clarity: float = 0.0
    technical_depth: float = 0.0
    novelty_significance: float = 0.0
    reproducibility: float = 0.0
    
    # Penalty metrics (subtract from total)
    critical_issues: int = 0
    major_issues: int = 0
    minor_issues: int = 0
    
    # Overall scores
    raw_score: float = 0.0
    penalty_adjusted_score: float = 0.0
    final_score: float = 0.0
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            'empirical_validation': self.empirical_validation,
            'figure_quality': self.figure_quality,
            'methodology_rigor': self.methodology_rigor,
            'experimental_design': self.experimental_design,
            'writing_clarity': self.writing_clarity,
            'technical_depth': self.technical_depth,
            'novelty_significance': self.novelty_significance,
            'reproducibility': self.reproducibility,
            'critical_issues': self.critical_issues,
            'major_issues': self.major_issues,
            'minor_issues': self.minor_issues,
            'raw_score': self.raw_score,
            'penalty_adjusted_score': self.penalty_adjusted_score,
            'final_score': self.final_score
        }


class FairPaperScorer:
    """
    Fair and realistic paper quality scoring system.
    
    This system is designed to be strict and realistic, providing scores that
    better reflect actual paper quality and reviewer expectations.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """Initialize scorer with configuration."""
        self.config = config or self._default_config()
        
    def _default_config(self) -> Dict:
        """Default configuration with realistic thresholds."""
        return {
            # Scoring weights (must sum to 1.0)
            "weights": {
                "empirical_validation": 0.25,    # Most important
                "methodology_rigor": 0.20,       # Second most important  
                "experimental_design": 0.15,     # Third most important
                "figure_quality": 0.10,          # Important for presentation
                "writing_clarity": 0.10,         # Important for understanding
                "technical_depth": 0.10,         # Important for contribution
                "novelty_significance": 0.05,    # Nice to have but often overstated
                "reproducibility": 0.05          # Important but often lacking
            },
            
            # Penalty weights (subtracted from final score)
            "penalties": {
                "critical_issue": 0.10,     # Each critical issue -0.10 (was 0.15)
                "major_issue": 0.05,        # Each major issue -0.05 (was 0.08)  
                "minor_issue": 0.02         # Each minor issue -0.02 (was 0.03)
            },
            
            # Minimum thresholds for decent scores
            "thresholds": {
                "min_figures": 3,           # Minimum figures required
                "min_references": 15,       # Minimum references required
                "min_experiments": 2,       # Minimum experimental evaluations
                "min_baselines": 3,         # Minimum baseline comparisons
                "min_datasets": 2,          # Minimum datasets evaluated
                "min_word_count": 4000,     # Minimum paper length
                "max_word_count": 15000     # Maximum reasonable length
            },
            
            # Grade boundaries (realistic academic standards)
            "grade_boundaries": {
                "excellent": 0.85,      # Publishable at top venue (very rare)
                "good": 0.70,           # Publishable at good venue  
                "acceptable": 0.55,     # Publishable with minor revisions
                "weak": 0.40,           # Major revisions needed
                "poor": 0.25,           # Likely rejection
                "terrible": 0.0         # Definite rejection
            }
        }
    
    def score_paper(self, paper_content: str, simulation_output: str = "", 
                   metadata: Optional[Dict] = None) -> Tuple[QualityMetrics, List[str]]:
        """
        Comprehensive paper scoring with realistic assessment.
        
        Returns:
            Tuple of (quality_metrics, issues_list)
        """
        metrics = QualityMetrics()
        issues = []
        
        # Score individual components
        metrics.empirical_validation, emp_issues = self._score_empirical_validation(
            paper_content, simulation_output)
        issues.extend(emp_issues)
        
        metrics.figure_quality, fig_issues = self._score_figure_quality(paper_content)
        issues.extend(fig_issues)
        
        metrics.methodology_rigor, method_issues = self._score_methodology_rigor(paper_content)
        issues.extend(method_issues)
        
        metrics.experimental_design, exp_issues = self._score_experimental_design(
            paper_content, simulation_output)
        issues.extend(exp_issues)
        
        metrics.writing_clarity, writing_issues = self._score_writing_clarity(paper_content)
        issues.extend(writing_issues)
        
        metrics.technical_depth, tech_issues = self._score_technical_depth(paper_content)
        issues.extend(tech_issues)
        
        metrics.novelty_significance, novelty_issues = self._score_novelty_significance(paper_content)
        issues.extend(novelty_issues)
        
        metrics.reproducibility, repro_issues = self._score_reproducibility(paper_content)
        issues.extend(repro_issues)
        
        # Count issue severity
        metrics.critical_issues = sum(1 for issue in issues if issue.startswith("CRITICAL"))
        metrics.major_issues = sum(1 for issue in issues if issue.startswith("MAJOR"))
        metrics.minor_issues = sum(1 for issue in issues if issue.startswith("MINOR"))
        
        # Calculate final scores
        metrics.raw_score = self._calculate_weighted_score(metrics)
        metrics.penalty_adjusted_score = self._apply_penalties(metrics)
        metrics.final_score = self._apply_reality_check(metrics)
        
        # Print detailed scoring breakdown
        self._print_detailed_breakdown(metrics, issues)
        
        return metrics, issues
    
    def _score_empirical_validation(self, content: str, simulation: str) -> Tuple[float, List[str]]:
        """Score empirical validation quality (most important factor)."""
        score = 0.0
        issues = []
        
        # Check for real experiments vs analytical/theoretical only
        experimental_indicators = [
            "experiment", "evaluation", "benchmark", "dataset", "test", 
            "accuracy", "precision", "recall", "f1", "results"
        ]
        
        analytical_only_indicators = [
            "analytical", "theoretical", "simulation only", "toy example",
            "synthetic", "artificial", "hypothetical"
        ]
        
        has_experiments = any(indicator in content.lower() for indicator in experimental_indicators)
        is_analytical_only = any(indicator in content.lower() for indicator in analytical_only_indicators)
        
        if not has_experiments:
            issues.append("CRITICAL: No experimental validation found")
            return 0.0, issues
        
        if is_analytical_only and not has_experiments:
            issues.append("CRITICAL: Appears to be analytical/theoretical only without empirical validation")
            score = 0.1
        
        # Check for standard benchmarks
        standard_benchmarks = [
            "gsm8k", "math", "humaneval", "mbpp", "hellaswag", "mmlu", 
            "superglue", "glue", "squad", "natural questions", "ms marco",
            "imagenet", "cifar", "coco", "pascal voc", "ade20k"
        ]
        
        benchmark_count = sum(1 for benchmark in standard_benchmarks 
                             if benchmark.lower() in content.lower())
        
        if benchmark_count == 0:
            issues.append("MAJOR: No standard benchmarks mentioned")
            score = max(score, 0.2)
        elif benchmark_count == 1:
            issues.append("MINOR: Only one benchmark used - need more for robustness")
            score = max(score, 0.4)
        elif benchmark_count >= 2:
            score = max(score, 0.6)
        
        # Check for measured vs simulated results
        if "measured" in content.lower() or "empirical" in content.lower():
            score += 0.1
        
        if simulation and len(simulation) > 100:
            score += 0.1
            
        # Check for statistical significance
        if "p-value" in content.lower() or "significance" in content.lower():
            score += 0.1
        else:
            issues.append("MAJOR: No statistical significance testing mentioned")
            
        # Check for confidence intervals  
        if "confidence interval" in content.lower() or "error bar" in content.lower():
            score += 0.1
        else:
            issues.append("MINOR: No confidence intervals or error bars mentioned")
            
        return min(score, 1.0), issues
    
    def _score_figure_quality(self, content: str) -> Tuple[float, List[str]]:
        """Score figure quality and quantity."""
        score = 0.0
        issues = []
        
        # Count figures
        figure_count = len(re.findall(r'\\begin{figure}|\\includegraphics', content))
        table_count = len(re.findall(r'\\begin{table}', content))
        total_visuals = figure_count + table_count
        
        if total_visuals == 0:
            issues.append("CRITICAL: No figures or tables found - papers with 0 visuals are often rejected")
            return 0.0, issues
        elif total_visuals < 3:
            issues.append(f"MAJOR: Only {total_visuals} visual elements - need at least 3-4")
            score = 0.3
        elif total_visuals < 5:
            issues.append(f"MINOR: Only {total_visuals} visual elements - could benefit from more")
            score = 0.6
        else:
            score = 0.8
        
        # Check for figure quality indicators
        quality_indicators = [
            "tikz", "pgfplot", "algorithm", "diagram", "architecture", 
            "comparison", "ablation", "performance", "chart"
        ]
        
        quality_count = sum(1 for indicator in quality_indicators 
                           if indicator in content.lower())
        
        if quality_count >= 3:
            score += 0.2
        elif quality_count >= 1:
            score += 0.1
        else:
            issues.append("MINOR: Figures appear to be basic - could improve with better visualization")
            
        return min(score, 1.0), issues
    
    def _score_methodology_rigor(self, content: str) -> Tuple[float, List[str]]:
        """Score methodology rigor and clarity."""
        score = 0.0
        issues = []
        
        # Check for algorithm/method description
        if "algorithm" in content.lower() or "method" in content.lower():
            score += 0.2
        else:
            issues.append("MAJOR: No clear algorithm or methodology section")
            
        # Check for pseudocode
        if "algorithm{" in content or "algorithmic" in content:
            score += 0.2
        else:
            issues.append("MINOR: No pseudocode provided - reduces clarity")
            
        # Check for mathematical formulation
        math_indicators = len(re.findall(r'\$.*\$|\\begin{equation}|\\begin{align}', content))
        if math_indicators >= 10:
            score += 0.3
        elif math_indicators >= 5:
            score += 0.2
        elif math_indicators >= 1:
            score += 0.1
        else:
            issues.append("MAJOR: Very little mathematical formulation")
            
        # Check for complexity analysis
        if "complexity" in content.lower() or "time complexity" in content.lower():
            score += 0.1
        else:
            issues.append("MINOR: No computational complexity analysis")
            
        # Check for theoretical grounding
        if "theorem" in content.lower() or "lemma" in content.lower() or "proof" in content.lower():
            score += 0.2
        
        return min(score, 1.0), issues
    
    def _score_experimental_design(self, content: str, simulation: str) -> Tuple[float, List[str]]:
        """Score experimental design quality."""
        score = 0.0
        issues = []
        
        # Check for baseline comparisons
        baseline_indicators = ["baseline", "comparison", "vs", "versus", "compared to"]
        baseline_count = sum(1 for indicator in baseline_indicators 
                            if indicator in content.lower())
        
        if baseline_count == 0:
            issues.append("CRITICAL: No baseline comparisons found")
            return 0.1, issues
        elif baseline_count < 3:
            issues.append("MAJOR: Limited baseline comparisons")
            score = 0.3
        else:
            score = 0.5
            
        # Check for ablation studies
        if "ablation" in content.lower():
            score += 0.2
        else:
            issues.append("MAJOR: No ablation study performed")
            
        # Check for multiple datasets/settings
        dataset_indicators = ["dataset", "benchmark", "corpus", "collection"]
        dataset_count = sum(1 for indicator in dataset_indicators 
                           if indicator in content.lower())
        
        if dataset_count >= 3:
            score += 0.2
        elif dataset_count >= 2:
            score += 0.1
        else:
            issues.append("MINOR: Limited dataset diversity")
            
        # Check for hyperparameter analysis  
        if "hyperparameter" in content.lower() or "parameter" in content.lower():
            score += 0.1
        else:
            issues.append("MINOR: No hyperparameter analysis mentioned")
            
        return min(score, 1.0), issues
    
    def _score_writing_clarity(self, content: str) -> Tuple[float, List[str]]:
        """Score writing quality and clarity."""
        score = 0.6  # Start with reasonable baseline
        issues = []
        
        word_count = len(content.split())
        
        # Check paper length (more lenient)
        if word_count < 1000:  # Very short
            issues.append(f"MAJOR: Paper very short ({word_count} words)")
            score -= 0.2
        elif word_count < self.config["thresholds"]["min_word_count"]:
            issues.append(f"MINOR: Paper could be longer ({word_count} words)")
            score -= 0.1
        elif word_count > self.config["thresholds"]["max_word_count"]:
            issues.append(f"MINOR: Paper quite long ({word_count} words)")
            score -= 0.05
            
        # Check for proper sections
        expected_sections = [
            "introduction", "related work", "methodology", "experiment", 
            "result", "conclusion", "reference"
        ]
        
        section_count = sum(1 for section in expected_sections 
                           if section in content.lower())
        
        if section_count < 3:  # Very few sections
            issues.append("MAJOR: Missing many standard paper sections")
            score -= 0.2
        elif section_count < 5:
            issues.append("MINOR: Missing some standard paper sections")
            score -= 0.1
        else:
            score += 0.1  # Bonus for good structure
            
        # Check for formatting issues (less harsh)
        if re.search(r'\n\s*-\s+', content):  # Plain text bullets
            issues.append("MINOR: Poor LaTeX formatting (using - instead of itemize)")
            score -= 0.05
            
        # Check reference count (more reasonable)
        ref_count = len(re.findall(r'\\cite{|\\bibitem{|\\bibliography', content))
        if ref_count < 5:  # Very few references
            issues.append(f"MAJOR: Very few references ({ref_count})")
            score -= 0.15
        elif ref_count < self.config["thresholds"]["min_references"]:
            issues.append(f"MINOR: Could use more references ({ref_count})")
            score -= 0.05
        elif ref_count >= 25:
            score += 0.1
            
        return max(0.2, min(score, 1.0)), issues  # Floor at 0.2
    
    def _score_technical_depth(self, content: str) -> Tuple[float, List[str]]:
        """Score technical depth and sophistication."""
        score = 0.3  # Conservative baseline
        issues = []
        
        # Check for technical concepts
        technical_indicators = [
            "neural network", "deep learning", "machine learning", "algorithm",
            "optimization", "gradient", "loss function", "architecture",
            "transformer", "attention", "convolution", "lstm", "gru"
        ]
        
        tech_count = sum(1 for indicator in technical_indicators 
                        if indicator in content.lower())
        
        if tech_count >= 10:
            score += 0.4
        elif tech_count >= 5:
            score += 0.3
        elif tech_count >= 2:
            score += 0.2
        else:
            issues.append("MAJOR: Limited technical depth")
            
        # Check for advanced concepts
        advanced_indicators = [
            "attention mechanism", "self-attention", "cross-attention", "multi-head",
            "batch normalization", "dropout", "regularization", "activation function",
            "backpropagation", "gradient descent", "stochastic", "optimization"
        ]
        
        advanced_count = sum(1 for indicator in advanced_indicators 
                            if indicator in content.lower())
        
        if advanced_count >= 5:
            score += 0.3
        elif advanced_count >= 2:
            score += 0.2
        elif advanced_count >= 1:
            score += 0.1
        
        return min(score, 1.0), issues
    
    def _score_novelty_significance(self, content: str) -> Tuple[float, List[str]]:
        """Score novelty and significance (often overestimated)."""
        score = 0.2  # Start low - novelty is often overstated
        issues = []
        
        # Check for novelty claims
        novelty_claims = ["novel", "new", "first", "unprecedented", "breakthrough"]
        claim_count = sum(1 for claim in novelty_claims 
                         if claim in content.lower())
        
        if claim_count > 5:
            issues.append("MINOR: Excessive novelty claims - may be overstated")
            score = 0.3  # Cap score if too many claims
        elif claim_count >= 2:
            score += 0.2
        elif claim_count >= 1:
            score += 0.1
        else:
            issues.append("MINOR: No explicit novelty claims")
            
        # Check for contribution clarity
        if "contribution" in content.lower():
            score += 0.2
        else:
            issues.append("MAJOR: No clear contribution statement")
            
        # Check for significance indicators
        significance_indicators = ["significant", "improvement", "outperform", "state-of-the-art"]
        sig_count = sum(1 for indicator in significance_indicators 
                       if indicator in content.lower())
        
        if sig_count >= 3:
            score += 0.2
        elif sig_count >= 1:
            score += 0.1
            
        return min(score, 1.0), issues
    
    def _score_reproducibility(self, content: str) -> Tuple[float, List[str]]:
        """Score reproducibility potential."""
        score = 0.2  # Start low - reproducibility often lacking
        issues = []
        
        # Check for code availability
        code_indicators = ["code", "implementation", "github", "repository", "available"]
        if any(indicator in content.lower() for indicator in code_indicators):
            score += 0.3
        else:
            issues.append("MAJOR: No mention of code availability")
            
        # Check for hyperparameter details
        if "hyperparameter" in content.lower():
            score += 0.2
        else:
            issues.append("MINOR: No hyperparameter details provided")
            
        # Check for experimental details
        detail_indicators = ["seed", "random", "initialization", "training", "epochs"]
        detail_count = sum(1 for indicator in detail_indicators 
                          if indicator in content.lower())
        
        if detail_count >= 3:
            score += 0.3
        elif detail_count >= 1:
            score += 0.2
        else:
            issues.append("MAJOR: Insufficient experimental details for reproduction")
            
        # Check for data availability
        if "data" in content.lower() and "available" in content.lower():
            score += 0.2
        else:
            issues.append("MINOR: No mention of data availability")
            
        return min(score, 1.0), issues
    
    def _calculate_weighted_score(self, metrics: QualityMetrics) -> float:
        """Calculate weighted average score."""
        weights = self.config["weights"]
        
        weighted_score = (
            metrics.empirical_validation * weights["empirical_validation"] +
            metrics.figure_quality * weights["figure_quality"] +
            metrics.methodology_rigor * weights["methodology_rigor"] +
            metrics.experimental_design * weights["experimental_design"] +
            metrics.writing_clarity * weights["writing_clarity"] +
            metrics.technical_depth * weights["technical_depth"] +
            metrics.novelty_significance * weights["novelty_significance"] +
            metrics.reproducibility * weights["reproducibility"]
        )
        
        return weighted_score
    
    def _apply_penalties(self, metrics: QualityMetrics) -> float:
        """Apply penalties for issues."""
        penalties = self.config["penalties"]
        
        total_penalty = (
            metrics.critical_issues * penalties["critical_issue"] +
            metrics.major_issues * penalties["major_issue"] +
            metrics.minor_issues * penalties["minor_issue"]
        )
        
        return max(0.0, metrics.raw_score - total_penalty)
    
    def _apply_reality_check(self, metrics: QualityMetrics) -> float:
        """Apply final reality check to prevent inflated scores."""
        score = metrics.penalty_adjusted_score
        
        # Reality adjustments based on common issues (reduced severity)
        if metrics.empirical_validation < 0.3:
            score *= 0.6  # Severely penalize very weak empirical validation
        elif metrics.empirical_validation < 0.5:
            score *= 0.8  # Moderately penalize weak empirical validation
            
        if metrics.figure_quality < 0.2:
            score *= 0.7  # Penalize missing figures
        elif metrics.figure_quality < 0.4:
            score *= 0.9  # Lightly penalize few figures
            
        if metrics.critical_issues > 2:
            score *= (0.7 ** (metrics.critical_issues - 2))  # Only heavy penalty for many critical issues
        elif metrics.critical_issues > 0:
            score *= 0.9  # Light penalty for some critical issues
            
        # Apply more reasonable grade curve
        if score > 0.9:
            score = 0.9 + (score - 0.9) * 0.3  # Compress very high scores
        elif score > 0.75:
            score = 0.75 + (score - 0.75) * 0.6  # Moderate compression
        elif score > 0.6:
            score = 0.6 + (score - 0.6) * 0.8  # Light compression
            
        return max(0.0, min(score, 1.0))
    
    def _print_detailed_breakdown(self, metrics: QualityMetrics, issues: List[str]) -> None:
        """Print detailed scoring breakdown for transparency."""
        print("\n" + "="*70)
        print("📊 DETAILED FAIR SCORING BREAKDOWN")
        print("="*70)
        
        # Component scores with weights
        weights = self.config["weights"]
        print(f"🔍 COMPONENT SCORES (0.0-1.0):")
        print(f"  • Empirical Validation:  {metrics.empirical_validation:.3f} (weight: {weights['empirical_validation']:.1f})")
        print(f"  • Figure Quality:        {metrics.figure_quality:.3f} (weight: {weights['figure_quality']:.1f})")
        print(f"  • Methodology Rigor:     {metrics.methodology_rigor:.3f} (weight: {weights['methodology_rigor']:.1f})")
        print(f"  • Experimental Design:   {metrics.experimental_design:.3f} (weight: {weights['experimental_design']:.1f})")
        print(f"  • Writing Clarity:       {metrics.writing_clarity:.3f} (weight: {weights['writing_clarity']:.1f})")
        print(f"  • Technical Depth:       {metrics.technical_depth:.3f} (weight: {weights['technical_depth']:.1f})")
        print(f"  • Novelty/Significance:  {metrics.novelty_significance:.3f} (weight: {weights['novelty_significance']:.1f})")
        print(f"  • Reproducibility:       {metrics.reproducibility:.3f} (weight: {weights['reproducibility']:.1f})")
        
        # Weighted score calculation
        weighted_sum = (
            metrics.empirical_validation * weights['empirical_validation'] +
            metrics.figure_quality * weights['figure_quality'] +
            metrics.methodology_rigor * weights['methodology_rigor'] +
            metrics.experimental_design * weights['experimental_design'] +
            metrics.writing_clarity * weights['writing_clarity'] +
            metrics.technical_depth * weights['technical_depth'] +
            metrics.novelty_significance * weights['novelty_significance'] +
            metrics.reproducibility * weights['reproducibility']
        )
        
        print(f"\n📈 SCORING CALCULATION:")
        print(f"  • Raw Weighted Score:    {metrics.raw_score:.3f} (Σ components × weights)")
        
        # Issue penalties
        penalties = self.config["penalties"]
        penalty_amount = (
            metrics.critical_issues * penalties["critical_issue"] +
            metrics.major_issues * penalties["major_issue"] +
            metrics.minor_issues * penalties["minor_issue"]
        )
        
        print(f"\n⚠️  ISSUE PENALTIES:")
        print(f"  • Critical Issues: {metrics.critical_issues} × {penalties['critical_issue']:.3f} = -{metrics.critical_issues * penalties['critical_issue']:.3f}")
        print(f"  • Major Issues:    {metrics.major_issues} × {penalties['major_issue']:.3f} = -{metrics.major_issues * penalties['major_issue']:.3f}")
        print(f"  • Minor Issues:    {metrics.minor_issues} × {penalties['minor_issue']:.3f} = -{metrics.minor_issues * penalties['minor_issue']:.3f}")
        print(f"  • Total Penalty:   -{penalty_amount:.3f}")
        print(f"  • After Penalties: {metrics.penalty_adjusted_score:.3f}")
        
        # Reality check adjustment
        reality_adjustment = metrics.final_score - metrics.penalty_adjusted_score
        print(f"\n🎯 REALITY CHECK:")
        print(f"  • Pre-Reality Score:  {metrics.penalty_adjusted_score:.3f}")
        print(f"  • Reality Adjustment: {reality_adjustment:+.3f}")
        print(f"  • Final Score:        {metrics.final_score:.3f}")
        
        # Grade
        grade = self.get_grade_description(metrics.final_score)
        print(f"\n🏆 FINAL ASSESSMENT:")
        print(f"  • Score: {metrics.final_score:.3f}")
        print(f"  • Grade: {grade}")
        
        # Top issues summary
        critical_issues = [issue for issue in issues if issue.startswith("CRITICAL")]
        major_issues = [issue for issue in issues if issue.startswith("MAJOR")]
        
        if critical_issues or major_issues:
            print(f"\n🚨 KEY ISSUES TO ADDRESS:")
            for issue in critical_issues[:3]:  # Show top 3 critical
                print(f"  ⚠️  {issue}")
            for issue in major_issues[:3]:  # Show top 3 major
                print(f"  📋 {issue}")
            
            if len(critical_issues) + len(major_issues) > 6:
                remaining = len(critical_issues) + len(major_issues) - 6
                print(f"  ... and {remaining} more issues")
        else:
            print(f"\n✅ NO CRITICAL OR MAJOR ISSUES FOUND!")
            
        print("="*70)
    
    def get_grade_description(self, score: float) -> str:
        """Get human-readable grade description."""
        boundaries = self.config["grade_boundaries"]
        
        if score >= boundaries["excellent"]:
            return "EXCELLENT (Publishable at top venues)"
        elif score >= boundaries["good"]:
            return "GOOD (Publishable at solid venues)"
        elif score >= boundaries["acceptable"]:
            return "ACCEPTABLE (Minor revisions needed)"
        elif score >= boundaries["weak"]:
            return "WEAK (Major revisions required)"
        elif score >= boundaries["poor"]:
            return "POOR (Likely rejection)"
        else:
            return "TERRIBLE (Definite rejection)"
    
    def generate_detailed_report(self, metrics: QualityMetrics, issues: List[str]) -> str:
        """Generate comprehensive scoring report."""
        report = []
        report.append("=" * 80)
        report.append("FAIR PAPER QUALITY ASSESSMENT REPORT")
        report.append("=" * 80)
        report.append("")
        
        # Overall score
        report.append(f"FINAL SCORE: {metrics.final_score:.3f} / 1.000")
        report.append(f"GRADE: {self.get_grade_description(metrics.final_score)}")
        report.append("")
        
        # Score breakdown
        report.append("DETAILED SCORE BREAKDOWN:")
        report.append("-" * 40)
        report.append(f"Empirical Validation:   {metrics.empirical_validation:.3f} (weight: 25%)")
        report.append(f"Methodology Rigor:      {metrics.methodology_rigor:.3f} (weight: 20%)")
        report.append(f"Experimental Design:    {metrics.experimental_design:.3f} (weight: 15%)")
        report.append(f"Figure Quality:         {metrics.figure_quality:.3f} (weight: 10%)")
        report.append(f"Writing Clarity:        {metrics.writing_clarity:.3f} (weight: 10%)")
        report.append(f"Technical Depth:        {metrics.technical_depth:.3f} (weight: 10%)")
        report.append(f"Novelty Significance:   {metrics.novelty_significance:.3f} (weight: 5%)")
        report.append(f"Reproducibility:        {metrics.reproducibility:.3f} (weight: 5%)")
        report.append("")
        
        # Score calculation
        report.append("SCORE CALCULATION:")
        report.append("-" * 40)
        report.append(f"Raw Weighted Score:     {metrics.raw_score:.3f}")
        report.append(f"After Penalties:        {metrics.penalty_adjusted_score:.3f}")
        report.append(f"After Reality Check:    {metrics.final_score:.3f}")
        report.append("")
        
        # Issues summary
        report.append(f"ISSUES SUMMARY:")
        report.append("-" * 40)
        report.append(f"Critical Issues: {metrics.critical_issues} (-{metrics.critical_issues * self.config['penalties']['critical_issue']:.3f})")
        report.append(f"Major Issues:    {metrics.major_issues} (-{metrics.major_issues * self.config['penalties']['major_issue']:.3f})")
        report.append(f"Minor Issues:    {metrics.minor_issues} (-{metrics.minor_issues * self.config['penalties']['minor_issue']:.3f})")
        report.append("")
        
        # Detailed issues
        if issues:
            report.append("DETAILED ISSUES:")
            report.append("-" * 40)
            for issue in issues:
                report.append(f"• {issue}")
            report.append("")
        
        # Recommendations
        report.append("RECOMMENDATIONS FOR IMPROVEMENT:")
        report.append("-" * 40)
        if metrics.empirical_validation < 0.5:
            report.append("• URGENT: Strengthen empirical validation with real experiments")
        if metrics.figure_quality < 0.3:
            report.append("• URGENT: Add more figures and improve visual presentation")
        if metrics.experimental_design < 0.5:
            report.append("• IMPORTANT: Improve experimental design with better baselines")
        if metrics.methodology_rigor < 0.5:
            report.append("• IMPORTANT: Provide more rigorous methodology description")
        if metrics.reproducibility < 0.4:
            report.append("• RECOMMENDED: Improve reproducibility with code/data availability")
        
        report.append("=" * 80)
        
        return "\n".join(report)


def score_paper_file(paper_path: Union[str, Path], simulation_path: Optional[Union[str, Path]] = None,
                    config: Optional[Dict] = None) -> Tuple[QualityMetrics, List[str], str]:
    """
    Convenience function to score a paper from file.
    
    Returns:
        Tuple of (quality_metrics, issues_list, detailed_report)
    """
    scorer = FairPaperScorer(config)
    
    # Read paper content
    paper_content = Path(paper_path).read_text(encoding='utf-8')
    
    # Read simulation output if provided
    simulation_output = ""
    if simulation_path and Path(simulation_path).exists():
        simulation_output = Path(simulation_path).read_text(encoding='utf-8')
    
    # Score the paper
    metrics, issues = scorer.score_paper(paper_content, simulation_output)
    
    # Generate detailed report
    report = scorer.generate_detailed_report(metrics, issues)
    
    return metrics, issues, report


if __name__ == "__main__":
    # Example usage
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python fair_scoring_system.py <paper.tex> [simulation.py]")
        sys.exit(1)
    
    paper_file = sys.argv[1]
    simulation_file = sys.argv[2] if len(sys.argv) > 2 else None
    
    try:
        metrics, issues, report = score_paper_file(paper_file, simulation_file)
        print(report)
        
        # Save detailed results
        results = {
            "metrics": metrics.to_dict(),
            "issues": issues,
            "summary": {
                "final_score": metrics.final_score,
                "grade": FairPaperScorer().get_grade_description(metrics.final_score),
                "critical_issues": metrics.critical_issues,
                "total_issues": len(issues)
            }
        }
        
        output_file = Path(paper_file).with_suffix('.quality_report.json')
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\nDetailed results saved to: {output_file}")
        
    except Exception as e:
        print(f"Error scoring paper: {e}")
        sys.exit(1)