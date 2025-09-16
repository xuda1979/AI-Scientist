"""
Enhanced Quality Metrics Module
==============================

Advanced quality assessment with semantic coherence analysis,
methodology validation, novelty detection, and reproducibility scoring.
"""

import re
import math
import statistics
from typing import Dict, Any, List, Tuple, Optional
from pathlib import Path
import json
from dataclasses import dataclass
from datetime import datetime

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False

try:
    from sentence_transformers import SentenceTransformer
    HAS_SENTENCE_TRANSFORMERS = True
except ImportError:
    HAS_SENTENCE_TRANSFORMERS = False


@dataclass
class QualityMetrics:
    """Comprehensive quality metrics container."""
    # Basic structural metrics
    word_count: int = 0
    section_count: int = 0
    figure_count: int = 0
    table_count: int = 0
    citation_count: int = 0
    equation_count: int = 0
    
    # Content quality metrics
    has_abstract: bool = False
    has_related_work: bool = False
    has_methodology: bool = False
    has_results: bool = False
    has_discussion: bool = False
    has_conclusion: bool = False
    
    # Enhanced quality metrics
    semantic_coherence_score: float = 0.0
    methodology_soundness_score: float = 0.0
    novelty_score: float = 0.0
    reproducibility_score: float = 0.0
    technical_depth_score: float = 0.0
    clarity_score: float = 0.0
    
    # Composite scores
    overall_quality_score: float = 0.0
    readiness_score: float = 0.0


class EnhancedQualityAnalyzer:
    """Advanced quality analyzer with semantic and methodological validation."""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.sentence_model = None
        self._init_sentence_transformer()
        
        # Quality assessment thresholds
        self.thresholds = {
            'min_word_count': 5000,
            'min_sections': 6,
            'min_citations': 15,
            'min_coherence': 0.7,
            'min_methodology': 0.6,
            'min_novelty': 0.5,
            'min_reproducibility': 0.8
        }
    
    def _init_sentence_transformer(self):
        """Initialize sentence transformer model if available."""
        if HAS_SENTENCE_TRANSFORMERS:
            try:
                self.sentence_model = SentenceTransformer(self.model_name)
                print(f"✓ Loaded sentence transformer: {self.model_name}")
            except Exception as e:
                print(f"⚠ Failed to load sentence transformer: {e}")
                self.sentence_model = None
        else:
            print("⚠ sentence-transformers not available - semantic analysis disabled")
    
    def analyze_paper_quality(self, paper_content: str, sim_summary: str = "", 
                            project_dir: Optional[Path] = None) -> QualityMetrics:
        """Comprehensive quality analysis of research paper."""
        metrics = QualityMetrics()
        
        # Basic structural analysis
        self._analyze_structure(paper_content, metrics)
        
        # Content quality analysis
        self._analyze_content_quality(paper_content, metrics)
        
        # Enhanced quality metrics
        metrics.semantic_coherence_score = self._analyze_semantic_coherence(paper_content)
        metrics.methodology_soundness_score = self._analyze_methodology_soundness(paper_content)
        metrics.novelty_score = self._detect_novelty_indicators(paper_content)
        metrics.reproducibility_score = self._assess_reproducibility(paper_content, sim_summary)
        metrics.technical_depth_score = self._analyze_technical_depth(paper_content)
        metrics.clarity_score = self._analyze_clarity(paper_content)
        
        # Calculate composite scores
        metrics.overall_quality_score = self._calculate_overall_quality(metrics)
        metrics.readiness_score = self._calculate_readiness_score(metrics)
        
        return metrics
    
    def _analyze_structure(self, paper_content: str, metrics: QualityMetrics):
        """Analyze basic paper structure."""
        metrics.word_count = len(paper_content.split())
        metrics.section_count = len(re.findall(r'\\section\{', paper_content))
        metrics.figure_count = len(re.findall(r'\\begin\{figure\}|\\includegraphics', paper_content))
        metrics.table_count = len(re.findall(r'\\begin\{table\}', paper_content))
        metrics.citation_count = len(re.findall(r'\\cite\{', paper_content))
        metrics.equation_count = len(re.findall(r'\\begin\{equation\}|\\begin\{align\}', paper_content))
    
    def _analyze_content_quality(self, paper_content: str, metrics: QualityMetrics):
        """Analyze content quality indicators."""
        metrics.has_abstract = bool(re.search(r'\\begin\{abstract\}', paper_content))
        metrics.has_related_work = bool(re.search(r'related.work|literature.review', paper_content, re.IGNORECASE))
        metrics.has_methodology = bool(re.search(r'methodology|method|approach', paper_content, re.IGNORECASE))
        metrics.has_results = bool(re.search(r'results|findings|outcomes', paper_content, re.IGNORECASE))
        metrics.has_discussion = bool(re.search(r'discussion|analysis', paper_content, re.IGNORECASE))
        metrics.has_conclusion = bool(re.search(r'conclusion|summary', paper_content, re.IGNORECASE))
    
    def _analyze_semantic_coherence(self, paper_content: str) -> float:
        """Analyze semantic coherence using sentence embeddings."""
        if not self.sentence_model:
            # Fallback: simple coherence heuristics
            return self._heuristic_coherence_analysis(paper_content)
        
        try:
            # Extract sentences from each section
            sections = self._extract_sections(paper_content)
            if len(sections) < 2:
                return 0.5  # Insufficient content for analysis
            
            coherence_scores = []
            
            for section_name, section_content in sections.items():
                sentences = self._extract_sentences(section_content)
                if len(sentences) < 2:
                    continue
                
                # Calculate sentence embeddings
                embeddings = self.sentence_model.encode(sentences)
                
                # Calculate pairwise similarities
                similarities = []
                for i in range(len(embeddings) - 1):
                    similarity = np.dot(embeddings[i], embeddings[i + 1]) / (
                        np.linalg.norm(embeddings[i]) * np.linalg.norm(embeddings[i + 1])
                    )
                    similarities.append(similarity)
                
                if similarities:
                    section_coherence = np.mean(similarities)
                    coherence_scores.append(section_coherence)
            
            if coherence_scores:
                return float(np.mean(coherence_scores))
            else:
                return 0.5
                
        except Exception as e:
            print(f"⚠ Semantic coherence analysis failed: {e}")
            return self._heuristic_coherence_analysis(paper_content)
    
    def _heuristic_coherence_analysis(self, paper_content: str) -> float:
        """Fallback coherence analysis using heuristics."""
        score = 0.5  # Base score
        
        # Check for transition words/phrases
        transitions = [
            'however', 'therefore', 'furthermore', 'moreover', 'consequently',
            'in addition', 'on the other hand', 'as a result', 'thus', 'hence'
        ]
        transition_count = sum(paper_content.lower().count(t) for t in transitions)
        score += min(0.2, transition_count / 100)  # Up to 0.2 bonus
        
        # Check for cross-references
        cross_refs = len(re.findall(r'\\ref\{|as shown in|see section|according to', paper_content, re.IGNORECASE))
        score += min(0.2, cross_refs / 20)  # Up to 0.2 bonus
        
        # Check for logical flow indicators
        flow_indicators = ['first', 'second', 'finally', 'next', 'then', 'subsequently']
        flow_count = sum(paper_content.lower().count(ind) for ind in flow_indicators)
        score += min(0.1, flow_count / 20)  # Up to 0.1 bonus
        
        return min(1.0, score)
    
    def _analyze_methodology_soundness(self, paper_content: str) -> float:
        """Analyze methodology soundness."""
        score = 0.0
        content_lower = paper_content.lower()
        
        # Experimental design indicators
        exp_design_terms = [
            'control group', 'randomized', 'statistical significance', 'p-value',
            'confidence interval', 'sample size', 'power analysis', 'effect size'
        ]
        exp_score = sum(1 for term in exp_design_terms if term in content_lower)
        score += min(0.3, exp_score / 8)
        
        # Validation indicators
        validation_terms = [
            'cross-validation', 'validation set', 'baseline comparison', 'ablation study',
            'robustness', 'sensitivity analysis', 'statistical test'
        ]
        val_score = sum(1 for term in validation_terms if term in content_lower)
        score += min(0.3, val_score / 7)
        
        # Reproducibility indicators
        repro_terms = [
            'reproducible', 'replicable', 'open source', 'code available',
            'implementation details', 'hyperparameters', 'random seed'
        ]
        repro_score = sum(1 for term in repro_terms if term in content_lower)
        score += min(0.2, repro_score / 7)
        
        # Limitation acknowledgment
        if re.search(r'limitation|constraint|assumption|bias', content_lower):
            score += 0.1
        
        # Ethics consideration
        if re.search(r'ethics|ethical|consent|privacy|fairness', content_lower):
            score += 0.1
        
        return min(1.0, score)
    
    def _detect_novelty_indicators(self, paper_content: str) -> float:
        """Detect novelty indicators in the paper."""
        score = 0.0
        content_lower = paper_content.lower()
        
        # Novelty claim indicators
        novelty_terms = [
            'novel', 'new', 'first', 'innovative', 'pioneering', 'breakthrough',
            'unprecedented', 'original', 'unique', 'state-of-the-art'
        ]
        novelty_count = sum(content_lower.count(term) for term in novelty_terms)
        score += min(0.3, novelty_count / 20)
        
        # Technical contribution indicators
        contribution_terms = [
            'contribution', 'advance', 'improvement', 'enhancement', 'extension',
            'generalization', 'framework', 'methodology', 'algorithm'
        ]
        contrib_count = sum(content_lower.count(term) for term in contribution_terms)
        score += min(0.3, contrib_count / 15)
        
        # Comparison with existing work
        if re.search(r'compared to|versus|outperform|superior|better than', content_lower):
            score += 0.2
        
        # Problem formulation novelty
        if re.search(r'formulate|define|introduce.*problem', content_lower):
            score += 0.1
        
        # New dataset or benchmark
        if re.search(r'new dataset|novel benchmark|introduce.*benchmark', content_lower):
            score += 0.1
        
        return min(1.0, score)
    
    def _assess_reproducibility(self, paper_content: str, sim_summary: str) -> float:
        """Assess reproducibility of the research."""
        score = 0.0
        
        # Check for simulation code
        if 'simulation.py' in sim_summary or 'SIMULATION CODE:' in sim_summary:
            score += 0.3
        
        # Check for implementation details
        impl_indicators = [
            'implementation', 'algorithm', 'pseudocode', 'parameters',
            'hyperparameters', 'configuration', 'setup'
        ]
        impl_count = sum(paper_content.lower().count(term) for term in impl_indicators)
        score += min(0.3, impl_count / 20)
        
        # Check for reproducibility statements
        repro_statements = [
            'reproducible', 'replicable', 'code available', 'open source',
            'github', 'repository', 'supplementary material'
        ]
        repro_count = sum(paper_content.lower().count(stmt) for stmt in repro_statements)
        score += min(0.2, repro_count / 7)
        
        # Check for statistical reporting
        if re.search(r'standard deviation|confidence interval|error bars', paper_content, re.IGNORECASE):
            score += 0.1
        
        # Check for detailed experimental setup
        if re.search(r'experimental setup|experimental configuration', paper_content, re.IGNORECASE):
            score += 0.1
        
        return min(1.0, score)
    
    def _analyze_technical_depth(self, paper_content: str) -> float:
        """Analyze technical depth of the paper."""
        score = 0.0
        
        # Mathematical content
        math_indicators = [
            '\\begin{equation}', '\\begin{align}', '\\theorem', '\\lemma',
            '\\proof', '\\definition', 'complexity', 'O('
        ]
        math_count = sum(paper_content.count(indicator) for indicator in math_indicators)
        score += min(0.3, math_count / 10)
        
        # Technical terminology
        tech_terms = [
            'algorithm', 'optimization', 'convergence', 'asymptotic',
            'polynomial', 'exponential', 'heuristic', 'approximation'
        ]
        tech_count = sum(paper_content.lower().count(term) for term in tech_terms)
        score += min(0.3, tech_count / 15)
        
        # Theoretical analysis
        theory_terms = [
            'theoretical', 'analysis', 'proof', 'theorem', 'lemma',
            'proposition', 'corollary', 'formal'
        ]
        theory_count = sum(paper_content.lower().count(term) for term in theory_terms)
        score += min(0.2, theory_count / 10)
        
        # Experimental rigor
        if len(re.findall(r'\\begin\{table\}|\\begin\{figure\}', paper_content)) >= 3:
            score += 0.1
        
        # Citation density
        citation_density = self._calculate_citation_density(paper_content)
        score += min(0.1, citation_density / 0.02)  # Up to 0.1 for 2% citation density
        
        return min(1.0, score)
    
    def _analyze_clarity(self, paper_content: str) -> float:
        """Analyze writing clarity."""
        score = 0.7  # Base clarity score
        
        # Calculate average sentence length (readability indicator)
        sentences = re.split(r'[.!?]+', paper_content)
        if sentences:
            avg_sentence_length = sum(len(s.split()) for s in sentences) / len(sentences)
            # Optimal range: 15-25 words per sentence
            if 15 <= avg_sentence_length <= 25:
                score += 0.1
            elif avg_sentence_length > 30:
                score -= 0.1  # Penalty for overly long sentences
        
        # Check for clear structure indicators
        structure_indicators = ['\\section{', '\\subsection{', '\\paragraph{']
        structure_count = sum(paper_content.count(ind) for ind in structure_indicators)
        if structure_count >= 6:
            score += 0.1
        
        # Check for explanation quality
        explanation_terms = [
            'in other words', 'that is', 'specifically', 'for example',
            'for instance', 'namely', 'i.e.', 'e.g.'
        ]
        explanation_count = sum(paper_content.lower().count(term) for term in explanation_terms)
        score += min(0.1, explanation_count / 10)
        
        return min(1.0, max(0.0, score))
    
    def _calculate_overall_quality(self, metrics: QualityMetrics) -> float:
        """Calculate overall quality score."""
        weights = {
            'structure': 0.15,
            'content': 0.15,
            'semantic_coherence': 0.15,
            'methodology': 0.20,
            'novelty': 0.10,
            'reproducibility': 0.15,
            'technical_depth': 0.10,
            'clarity': 0.10
        }
        
        # Structure score
        structure_score = min(1.0, (
            (metrics.word_count / self.thresholds['min_word_count']) * 0.3 +
            (metrics.section_count / self.thresholds['min_sections']) * 0.3 +
            (metrics.citation_count / self.thresholds['min_citations']) * 0.4
        ))
        
        # Content completeness score
        content_indicators = [
            metrics.has_abstract, metrics.has_related_work, metrics.has_methodology,
            metrics.has_results, metrics.has_discussion, metrics.has_conclusion
        ]
        content_score = sum(content_indicators) / len(content_indicators)
        
        overall_score = (
            structure_score * weights['structure'] +
            content_score * weights['content'] +
            metrics.semantic_coherence_score * weights['semantic_coherence'] +
            metrics.methodology_soundness_score * weights['methodology'] +
            metrics.novelty_score * weights['novelty'] +
            metrics.reproducibility_score * weights['reproducibility'] +
            metrics.technical_depth_score * weights['technical_depth'] +
            metrics.clarity_score * weights['clarity']
        )
        
        return min(1.0, overall_score)
    
    def _calculate_readiness_score(self, metrics: QualityMetrics) -> float:
        """Calculate readiness for publication."""
        # Must meet minimum thresholds
        if (metrics.word_count < self.thresholds['min_word_count'] or
            metrics.section_count < self.thresholds['min_sections'] or
            metrics.citation_count < self.thresholds['min_citations']):
            return 0.0
        
        # Quality gates
        quality_gates = [
            metrics.semantic_coherence_score >= self.thresholds['min_coherence'],
            metrics.methodology_soundness_score >= self.thresholds['min_methodology'],
            metrics.novelty_score >= self.thresholds['min_novelty'],
            metrics.reproducibility_score >= self.thresholds['min_reproducibility'],
            metrics.has_abstract and metrics.has_methodology and metrics.has_results
        ]
        
        gates_passed = sum(quality_gates)
        readiness = (gates_passed / len(quality_gates)) * metrics.overall_quality_score
        
        return min(1.0, readiness)
    
    def _extract_sections(self, paper_content: str) -> Dict[str, str]:
        """Extract sections from paper content."""
        sections = {}
        
        # Find section boundaries
        section_pattern = r'\\section\{([^}]+)\}(.*?)(?=\\section\{|\\bibliography|\\end\{document\}|$)'
        matches = re.findall(section_pattern, paper_content, re.DOTALL)
        
        for section_title, section_content in matches:
            sections[section_title] = section_content.strip()
        
        return sections
    
    def _extract_sentences(self, text: str) -> List[str]:
        """Extract sentences from text."""
        # Remove LaTeX commands for better sentence extraction
        clean_text = re.sub(r'\\[a-zA-Z]+\{[^}]*\}', '', text)
        clean_text = re.sub(r'\\[a-zA-Z]+', '', clean_text)
        
        # Split into sentences
        sentences = re.split(r'[.!?]+', clean_text)
        sentences = [s.strip() for s in sentences if s.strip() and len(s.split()) > 3]
        
        return sentences
    
    def _calculate_citation_density(self, paper_content: str) -> float:
        """Calculate citation density (citations per word)."""
        word_count = len(paper_content.split())
        citation_count = len(re.findall(r'\\cite\{', paper_content))
        
        if word_count == 0:
            return 0.0
        
        return citation_count / word_count
    
    def generate_quality_report(self, metrics: QualityMetrics) -> str:
        """Generate a comprehensive quality report."""
        report = []
        report.append("=" * 60)
        report.append("ENHANCED QUALITY ANALYSIS REPORT")
        report.append("=" * 60)
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")
        
        # Overall scores
        report.append("OVERALL ASSESSMENT:")
        report.append(f"  Overall Quality Score: {metrics.overall_quality_score:.3f}")
        report.append(f"  Publication Readiness: {metrics.readiness_score:.3f}")
        report.append("")
        
        # Structural metrics
        report.append("STRUCTURAL METRICS:")
        report.append(f"  Word Count: {metrics.word_count:,}")
        report.append(f"  Sections: {metrics.section_count}")
        report.append(f"  Figures: {metrics.figure_count}")
        report.append(f"  Tables: {metrics.table_count}")
        report.append(f"  Citations: {metrics.citation_count}")
        report.append(f"  Equations: {metrics.equation_count}")
        report.append("")
        
        # Content completeness
        report.append("CONTENT COMPLETENESS:")
        content_items = [
            ("Abstract", metrics.has_abstract),
            ("Related Work", metrics.has_related_work),
            ("Methodology", metrics.has_methodology),
            ("Results", metrics.has_results),
            ("Discussion", metrics.has_discussion),
            ("Conclusion", metrics.has_conclusion)
        ]
        for item, present in content_items:
            status = "✓" if present else "✗"
            report.append(f"  {status} {item}")
        report.append("")
        
        # Enhanced metrics
        report.append("ADVANCED QUALITY METRICS:")
        report.append(f"  Semantic Coherence: {metrics.semantic_coherence_score:.3f}")
        report.append(f"  Methodology Soundness: {metrics.methodology_soundness_score:.3f}")
        report.append(f"  Novelty Indicators: {metrics.novelty_score:.3f}")
        report.append(f"  Reproducibility: {metrics.reproducibility_score:.3f}")
        report.append(f"  Technical Depth: {metrics.technical_depth_score:.3f}")
        report.append(f"  Writing Clarity: {metrics.clarity_score:.3f}")
        report.append("")
        
        # Recommendations
        report.append("IMPROVEMENT RECOMMENDATIONS:")
        recommendations = self._generate_recommendations(metrics)
        for rec in recommendations:
            report.append(f"  • {rec}")
        
        report.append("=" * 60)
        
        return "\n".join(report)
    
    def _generate_recommendations(self, metrics: QualityMetrics) -> List[str]:
        """Generate improvement recommendations."""
        recommendations = []
        
        if metrics.word_count < self.thresholds['min_word_count']:
            recommendations.append(f"Expand content (current: {metrics.word_count}, target: {self.thresholds['min_word_count']})")
        
        if metrics.citation_count < self.thresholds['min_citations']:
            recommendations.append(f"Add more references (current: {metrics.citation_count}, target: {self.thresholds['min_citations']})")
        
        if metrics.semantic_coherence_score < self.thresholds['min_coherence']:
            recommendations.append("Improve semantic coherence between sections and paragraphs")
        
        if metrics.methodology_soundness_score < self.thresholds['min_methodology']:
            recommendations.append("Strengthen experimental methodology and validation")
        
        if metrics.novelty_score < self.thresholds['min_novelty']:
            recommendations.append("Better highlight novel contributions and differences from existing work")
        
        if metrics.reproducibility_score < self.thresholds['min_reproducibility']:
            recommendations.append("Improve reproducibility with better documentation and code availability")
        
        if not metrics.has_abstract:
            recommendations.append("Add comprehensive abstract")
        
        if not metrics.has_related_work:
            recommendations.append("Add related work section with proper literature review")
        
        if not metrics.has_methodology:
            recommendations.append("Add detailed methodology section")
        
        if metrics.figure_count == 0:
            recommendations.append("Add figures to illustrate key concepts and results")
        
        if metrics.table_count == 0:
            recommendations.append("Add tables to present numerical results")
        
        return recommendations if recommendations else ["Paper meets all quality thresholds"]
