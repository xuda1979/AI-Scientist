"""
Quality assessment and validation for research papers.
"""
from __future__ import annotations
import re
from typing import List, Dict, Any, Optional
from pathlib import Path


class QualityAssessment:
    """Comprehensive quality assessment for research papers."""
    
    def __init__(self, config):
        self.config = config
        self.quality_history = []
        self.stagnation_count = 0
        self.best_quality_score = 0.0
    
    def assess_paper_quality(
        self, 
        tex_content: str, 
        sim_summary: str,
        latex_errors: str = ""
    ) -> Dict[str, Any]:
        """Comprehensive quality assessment of a paper."""
        issues = []
        
        # Basic structural checks
        structural_issues = self._check_structure(tex_content)
        issues.extend(structural_issues)
        
        # Content quality checks
        content_issues = self._check_content_quality(tex_content)
        issues.extend(content_issues)
        
        # Reference validation
        if self.config.reference_validation:
            ref_issues = self._validate_references(tex_content)
            issues.extend(ref_issues)
        
        # LaTeX compilation issues
        if latex_errors:
            issues.append("LaTeX compilation errors present")
        
        # Calculate metrics
        metrics = self._extract_quality_metrics(tex_content, sim_summary)
        quality_score = self._calculate_quality_score(metrics, issues)
        
        # Update tracking
        self.quality_history.append(quality_score)
        if quality_score > self.best_quality_score:
            self.best_quality_score = quality_score
            self.stagnation_count = 0
        else:
            self.stagnation_count += 1
        
        return {
            "quality_score": quality_score,
            "issues": issues,
            "metrics": metrics,
            "stagnation_count": self.stagnation_count,
            "best_score": self.best_quality_score,
            "meets_threshold": quality_score >= self.config.quality_threshold
        }
    
    def _check_structure(self, tex_content: str) -> List[str]:
        """Check basic LaTeX document structure."""
        issues = []
        
        if len(tex_content) < 1000:
            issues.append("Paper too short")
        elif len(tex_content) > 100000:
            issues.append("Paper too long")
        
        if "\\begin{abstract}" not in tex_content:
            issues.append("Missing abstract")
        
        if "\\section" not in tex_content:
            issues.append("Missing sections")
        
        required_sections = ["introduction", "related work", "methodology", "results", "conclusion"]
        for section in required_sections:
            if section.lower() not in tex_content.lower():
                issues.append(f"Missing {section} section")
        
        return issues
    
    def _check_content_quality(self, tex_content: str) -> List[str]:
        """Check content quality indicators."""
        issues = []
        
        # Check for figures and tables
        if "\\begin{figure}" not in tex_content and "\\includegraphics" not in tex_content:
            issues.append("No figures found")
        
        if "\\begin{table}" not in tex_content:
            issues.append("No tables found")
        
        # Check for mathematical content
        math_patterns = [r"\\begin{equation}", r"\\begin{align}", r"\$.*\$", r"\\[.*\\]"]
        has_math = any(re.search(pattern, tex_content) for pattern in math_patterns)
        if not has_math:
            issues.append("Limited mathematical content")
        
        # Check for code/algorithms
        code_patterns = [r"\\begin{lstlisting}", r"\\begin{algorithm}", r"\\begin{verbatim}"]
        has_code = any(re.search(pattern, tex_content) for pattern in code_patterns)
        if not has_code:
            issues.append("No code or algorithms found")
        
        return issues
    
    def _validate_references(self, tex_content: str) -> List[str]:
        """Validate references and citations."""
        issues = []
        
        # Check for citations
        if "\\cite{" not in tex_content and "\\bibitem{" not in tex_content:
            issues.append("No citations found")
        
        # Check for bibliography
        if "\\begin{thebibliography}" not in tex_content and "\\bibliography{" not in tex_content:
            issues.append("No bibliography found")
        
        # Count citations vs bibliography entries
        citations = len(re.findall(r"\\cite\{[^}]+\}", tex_content))
        bibitems = len(re.findall(r"\\bibitem\{[^}]+\}", tex_content))
        
        if citations > 0 and bibitems == 0:
            issues.append("Citations without bibliography")
        elif bibitems > 0 and citations == 0:
            issues.append("Bibliography without citations")
        
        return issues
    
    def _extract_quality_metrics(self, tex_content: str, sim_summary: str) -> Dict[str, Any]:
        """Extract quantitative quality metrics."""
        return {
            "word_count": len(tex_content.split()),
            "char_count": len(tex_content),
            "section_count": len(re.findall(r"\\section\{", tex_content)),
            "subsection_count": len(re.findall(r"\\subsection\{", tex_content)),
            "citation_count": len(re.findall(r"\\cite\{[^}]+\}", tex_content)),
            "figure_count": len(re.findall(r"\\begin\{figure\}", tex_content)),
            "table_count": len(re.findall(r"\\begin\{table\}", tex_content)),
            "equation_count": len(re.findall(r"\\begin\{equation\}", tex_content)),
            "has_simulation": bool(sim_summary and len(sim_summary) > 100)
        }
    
    def _calculate_quality_score(self, metrics: Dict[str, Any], issues: List[str]) -> float:
        """Calculate overall quality score (0.0 - 1.0)."""
        base_score = 0.5  # Start at 50%
        
        # Positive factors
        if metrics["word_count"] > 3000:
            base_score += 0.1
        if metrics["section_count"] >= 5:
            base_score += 0.1
        if metrics["citation_count"] > 10:
            base_score += 0.1
        if metrics["figure_count"] > 0:
            base_score += 0.05
        if metrics["table_count"] > 0:
            base_score += 0.05
        if metrics["has_simulation"]:
            base_score += 0.1
        
        # Negative factors
        penalty = len(issues) * 0.05
        final_score = max(0.0, min(1.0, base_score - penalty))
        
        return final_score
    
    def get_progress_summary(self) -> Dict[str, Any]:
        """Get quality progress summary."""
        return {
            "quality_history": self.quality_history,
            "best_score": self.best_quality_score,
            "current_score": self.quality_history[-1] if self.quality_history else 0.0,
            "stagnation_count": self.stagnation_count,
            "improvement_trend": self._calculate_trend()
        }
    
    def _calculate_trend(self) -> str:
        """Calculate quality improvement trend."""
        if len(self.quality_history) < 2:
            return "insufficient_data"
        
        recent = self.quality_history[-3:] if len(self.quality_history) >= 3 else self.quality_history
        if all(recent[i] >= recent[i-1] for i in range(1, len(recent))):
            return "improving"
        elif all(recent[i] <= recent[i-1] for i in range(1, len(recent))):
            return "declining"
        else:
            return "stable"
