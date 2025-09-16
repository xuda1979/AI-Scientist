"""
Intelligent Content Enhancer
=============================

Adaptive content generation with domain-specific knowledge injection,
writing style optimization, and coherence improvement.
"""

import re
import json
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
from dataclasses import dataclass
from datetime import datetime
import hashlib

try:
    from sentence_transformers import SentenceTransformer, util
    HAS_SENTENCE_TRANSFORMERS = True
except ImportError:
    HAS_SENTENCE_TRANSFORMERS = False

try:
    import nltk
    from nltk.tokenize import sent_tokenize, word_tokenize
    from nltk.corpus import stopwords
    HAS_NLTK = True
except ImportError:
    HAS_NLTK = False


@dataclass
class WritingStyleAnalysis:
    """Analysis of writing style characteristics."""
    avg_sentence_length: float
    vocabulary_richness: float
    passive_voice_ratio: float
    transition_word_usage: float
    technical_term_density: float
    readability_score: float
    consistency_score: float


@dataclass
class ContentGap:
    """Represents a content gap that needs enhancement."""
    gap_type: str  # logical, methodological, background, conclusion
    location: str
    description: str
    severity: str  # critical, important, minor
    enhancement_suggestions: List[str]


@dataclass
class DomainKnowledge:
    """Domain-specific knowledge to inject."""
    field: str
    concepts: List[str]
    terminology: Dict[str, str]
    best_practices: List[str]
    common_pitfalls: List[str]


@dataclass
class ContentEnhancement:
    """Represents a content enhancement."""
    enhancement_type: str  # clarity, coherence, technical, style
    location: str
    original_text: str
    enhanced_text: str
    rationale: str
    confidence_score: float


@dataclass
class ContentEnhancementResult:
    """Complete content enhancement result."""
    writing_style_analysis: WritingStyleAnalysis
    identified_gaps: List[ContentGap]
    domain_knowledge: DomainKnowledge
    content_enhancements: List[ContentEnhancement]
    overall_improvement_score: float
    readability_improvement: float
    coherence_improvement: float
    recommendations: List[str]
    enhancement_summary: str


class IntelligentContentEnhancer:
    """Advanced content enhancement system."""
    
    def __init__(self, universal_chat_fn):
        self.universal_chat = universal_chat_fn
        
        # Initialize NLP models if available
        self.semantic_model = None
        if HAS_SENTENCE_TRANSFORMERS:
            try:
                self.semantic_model = SentenceTransformer('all-MiniLM-L6-v2')
            except Exception:
                pass
        
        # Writing style standards
        self.target_metrics = {
            'avg_sentence_length': (15, 25),  # words per sentence
            'passive_voice_ratio': (0.0, 0.2),  # max 20% passive voice
            'transition_word_usage': (0.05, 0.15),  # 5-15% of sentences
            'technical_term_density': (0.1, 0.3),  # 10-30% technical terms
            'readability_score': (40, 60)  # Flesch Reading Ease
        }
        
        # Domain knowledge bases
        self.domain_knowledge_base = {
            'machine_learning': {
                'concepts': [
                    'neural networks', 'deep learning', 'gradient descent',
                    'backpropagation', 'overfitting', 'regularization',
                    'cross-validation', 'feature engineering', 'ensemble methods'
                ],
                'terminology': {
                    'ML': 'machine learning',
                    'NN': 'neural network',
                    'CNN': 'convolutional neural network',
                    'RNN': 'recurrent neural network',
                    'LSTM': 'long short-term memory'
                },
                'best_practices': [
                    'Always report training and validation accuracy',
                    'Use appropriate evaluation metrics',
                    'Discuss potential biases in data',
                    'Compare with relevant baselines'
                ]
            },
            'statistics': {
                'concepts': [
                    'hypothesis testing', 'p-values', 'confidence intervals',
                    'effect size', 'statistical power', 'multiple comparisons'
                ],
                'terminology': {
                    'CI': 'confidence interval',
                    'SE': 'standard error',
                    'SD': 'standard deviation'
                },
                'best_practices': [
                    'Report effect sizes along with p-values',
                    'Discuss statistical assumptions',
                    'Address multiple comparison corrections'
                ]
            }
        }
        
        # Enhancement patterns
        self.enhancement_patterns = {
            'clarity': [
                (r'it is shown that', 'we demonstrate that'),
                (r'it can be seen that', 'as shown in'),
                (r'it is important to note', 'notably'),
                (r'in order to', 'to')
            ],
            'precision': [
                (r'very significant', 'highly significant (p < 0.001)'),
                (r'good results', 'improved performance'),
                (r'bad performance', 'reduced performance'),
                (r'a lot of', 'numerous')
            ],
            'formality': [
                (r'we can see', 'we observe'),
                (r'pretty good', 'satisfactory'),
                (r'a bunch of', 'several'),
                (r'gets better', 'improves')
            ]
        }
    
    def enhance_content(self, paper_content: str, field: str,
                       model: str, request_timeout: int = 1800) -> ContentEnhancementResult:
        """Comprehensive content enhancement."""
        print("✨ Starting Intelligent Content Enhancement...")
        
        # Analyze current writing style
        style_analysis = self._analyze_writing_style(paper_content)
        print(f"  Writing style analyzed - readability: {style_analysis.readability_score:.1f}")
        
        # Identify content gaps
        identified_gaps = self._identify_content_gaps(paper_content, model, request_timeout)
        print(f"  Identified {len(identified_gaps)} content gaps")
        
        # Get domain-specific knowledge
        domain_knowledge = self._get_domain_knowledge(field, paper_content)
        print(f"  Loaded {len(domain_knowledge.concepts)} domain concepts")
        
        # Generate content enhancements
        content_enhancements = self._generate_enhancements(
            paper_content, style_analysis, identified_gaps, 
            domain_knowledge, model, request_timeout
        )
        print(f"  Generated {len(content_enhancements)} content enhancements")
        
        # Calculate improvement scores
        overall_improvement_score = self._calculate_improvement_score(content_enhancements)
        readability_improvement = self._calculate_readability_improvement(
            style_analysis, content_enhancements
        )
        coherence_improvement = self._calculate_coherence_improvement(content_enhancements)
        
        # Generate recommendations
        recommendations = self._generate_enhancement_recommendations(
            style_analysis, identified_gaps, content_enhancements
        )
        
        # Generate enhancement summary
        enhancement_summary = self._generate_enhancement_summary(
            len(content_enhancements), overall_improvement_score,
            readability_improvement, coherence_improvement
        )
        
        return ContentEnhancementResult(
            writing_style_analysis=style_analysis,
            identified_gaps=identified_gaps,
            domain_knowledge=domain_knowledge,
            content_enhancements=content_enhancements,
            overall_improvement_score=overall_improvement_score,
            readability_improvement=readability_improvement,
            coherence_improvement=coherence_improvement,
            recommendations=recommendations,
            enhancement_summary=enhancement_summary
        )
    
    def _analyze_writing_style(self, paper_content: str) -> WritingStyleAnalysis:
        """Analyze writing style characteristics."""
        # Extract main text (exclude LaTeX commands, references, etc.)
        clean_text = self._clean_text_for_analysis(paper_content)
        
        # Sentence-level analysis
        sentences = self._split_sentences(clean_text)
        if not sentences:
            return self._default_style_analysis()
        
        # Calculate metrics
        avg_sentence_length = self._calculate_avg_sentence_length(sentences)
        vocabulary_richness = self._calculate_vocabulary_richness(clean_text)
        passive_voice_ratio = self._calculate_passive_voice_ratio(sentences)
        transition_word_usage = self._calculate_transition_word_usage(sentences)
        technical_term_density = self._calculate_technical_term_density(clean_text)
        readability_score = self._calculate_readability_score(sentences)
        consistency_score = self._calculate_consistency_score(sentences)
        
        return WritingStyleAnalysis(
            avg_sentence_length=avg_sentence_length,
            vocabulary_richness=vocabulary_richness,
            passive_voice_ratio=passive_voice_ratio,
            transition_word_usage=transition_word_usage,
            technical_term_density=technical_term_density,
            readability_score=readability_score,
            consistency_score=consistency_score
        )
    
    def _clean_text_for_analysis(self, paper_content: str) -> str:
        """Clean LaTeX content for text analysis."""
        # Remove LaTeX commands
        text = re.sub(r'\\[a-zA-Z]+\{[^}]*\}', '', paper_content)
        text = re.sub(r'\\[a-zA-Z]+', '', text)
        
        # Remove special characters and normalize whitespace
        text = re.sub(r'[{}$%&#^_~]', '', text)
        text = re.sub(r'\s+', ' ', text)
        
        # Remove common LaTeX environments
        text = re.sub(r'\\begin\{[^}]+\}.*?\\end\{[^}]+\}', '', text, flags=re.DOTALL)
        
        return text.strip()
    
    def _split_sentences(self, text: str) -> List[str]:
        """Split text into sentences."""
        if HAS_NLTK:
            try:
                return sent_tokenize(text)
            except LookupError:
                # Download required NLTK data if missing
                try:
                    nltk.download('punkt', quiet=True)
                    return sent_tokenize(text)
                except:
                    pass
        
        # Fallback: simple sentence splitting
        sentences = re.split(r'[.!?]+', text)
        return [s.strip() for s in sentences if s.strip()]
    
    def _calculate_avg_sentence_length(self, sentences: List[str]) -> float:
        """Calculate average sentence length in words."""
        if not sentences:
            return 0.0
        
        word_counts = []
        for sentence in sentences:
            words = len(sentence.split())
            if words > 0:  # Skip empty sentences
                word_counts.append(words)
        
        return sum(word_counts) / len(word_counts) if word_counts else 0.0
    
    def _calculate_vocabulary_richness(self, text: str) -> float:
        """Calculate vocabulary richness (unique words / total words)."""
        words = text.lower().split()
        if not words:
            return 0.0
        
        unique_words = set(words)
        return len(unique_words) / len(words)
    
    def _calculate_passive_voice_ratio(self, sentences: List[str]) -> float:
        """Calculate ratio of passive voice sentences."""
        if not sentences:
            return 0.0
        
        passive_patterns = [
            r'\bis\s+\w+ed\b', r'\bare\s+\w+ed\b', r'\bwas\s+\w+ed\b',
            r'\bwere\s+\w+ed\b', r'\bbeen\s+\w+ed\b'
        ]
        
        passive_count = 0
        for sentence in sentences:
            sentence_lower = sentence.lower()
            if any(re.search(pattern, sentence_lower) for pattern in passive_patterns):
                passive_count += 1
        
        return passive_count / len(sentences)
    
    def _calculate_transition_word_usage(self, sentences: List[str]) -> float:
        """Calculate usage of transition words."""
        if not sentences:
            return 0.0
        
        transition_words = {
            'however', 'therefore', 'furthermore', 'moreover', 'additionally',
            'consequently', 'nevertheless', 'specifically', 'particularly',
            'similarly', 'conversely', 'thus', 'hence', 'meanwhile'
        }
        
        transition_count = 0
        for sentence in sentences:
            sentence_lower = sentence.lower()
            if any(word in sentence_lower for word in transition_words):
                transition_count += 1
        
        return transition_count / len(sentences)
    
    def _calculate_technical_term_density(self, text: str) -> float:
        """Calculate density of technical terms."""
        words = text.lower().split()
        if not words:
            return 0.0
        
        # Common technical indicators
        technical_indicators = [
            'algorithm', 'method', 'approach', 'technique', 'framework',
            'model', 'analysis', 'evaluation', 'experiment', 'result',
            'performance', 'accuracy', 'precision', 'recall', 'optimization',
            'parameter', 'variable', 'function', 'matrix', 'vector'
        ]
        
        technical_count = sum(1 for word in words if any(term in word for term in technical_indicators))
        return technical_count / len(words)
    
    def _calculate_readability_score(self, sentences: List[str]) -> float:
        """Calculate Flesch Reading Ease score (simplified)."""
        if not sentences:
            return 0.0
        
        total_words = 0
        total_syllables = 0
        
        for sentence in sentences:
            words = sentence.split()
            total_words += len(words)
            
            for word in words:
                # Simplified syllable counting
                syllables = max(1, len(re.findall(r'[aeiouAEIOU]', word)))
                total_syllables += syllables
        
        if total_words == 0:
            return 0.0
        
        avg_sentence_length = total_words / len(sentences)
        avg_syllables_per_word = total_syllables / total_words
        
        # Simplified Flesch Reading Ease formula
        score = 206.835 - (1.015 * avg_sentence_length) - (84.6 * avg_syllables_per_word)
        return max(0, min(100, score))
    
    def _calculate_consistency_score(self, sentences: List[str]) -> float:
        """Calculate writing consistency score."""
        if len(sentences) < 2:
            return 1.0
        
        # Measure consistency in sentence length
        lengths = [len(s.split()) for s in sentences]
        if not lengths:
            return 1.0
        
        mean_length = sum(lengths) / len(lengths)
        variance = sum((l - mean_length) ** 2 for l in lengths) / len(lengths)
        coefficient_of_variation = (variance ** 0.5) / mean_length if mean_length > 0 else 0
        
        # Convert to 0-1 score (lower variation = higher consistency)
        consistency = max(0, 1 - coefficient_of_variation / 2)
        
        return consistency
    
    def _default_style_analysis(self) -> WritingStyleAnalysis:
        """Return default style analysis when text analysis fails."""
        return WritingStyleAnalysis(
            avg_sentence_length=20.0,
            vocabulary_richness=0.4,
            passive_voice_ratio=0.2,
            transition_word_usage=0.1,
            technical_term_density=0.2,
            readability_score=50.0,
            consistency_score=0.7
        )
    
    def _identify_content_gaps(self, paper_content: str, model: str, 
                             request_timeout: int) -> List[ContentGap]:
        """Identify gaps in content that need enhancement."""
        prompt = [
            {
                "role": "system",
                "content": """You are an expert academic editor identifying content gaps.
                Analyze the paper for missing elements, logical gaps, insufficient explanations,
                and areas that need enhancement.
                
                Respond with JSON format:
                {
                    "gaps": [
                        {
                            "gap_type": "<logical/methodological/background/conclusion>",
                            "location": "<section>",
                            "description": "<description>",
                            "severity": "<critical/important/minor>",
                            "enhancement_suggestions": ["<suggestion1>", "<suggestion2>"]
                        }
                    ]
                }"""
            },
            {
                "role": "user",
                "content": f"""Analyze this research paper for content gaps:

                PAPER CONTENT (first 3000 characters):
                {paper_content[:3000]}
                
                Identify gaps in:
                1. Logical flow and argumentation
                2. Methodological explanations
                3. Background and related work
                4. Results interpretation
                5. Conclusions and implications
                
                Focus on the most important gaps that would improve paper quality."""
            }
        ]
        
        try:
            response = self.universal_chat(
                prompt, model=model, request_timeout=request_timeout,
                prompt_type="content_gap_analysis"
            )
            
            data = json.loads(response.strip())
            gaps = []
            
            for gap_data in data.get('gaps', []):
                gap = ContentGap(
                    gap_type=gap_data.get('gap_type', 'unknown'),
                    location=gap_data.get('location', 'unknown'),
                    description=gap_data.get('description', ''),
                    severity=gap_data.get('severity', 'minor'),
                    enhancement_suggestions=gap_data.get('enhancement_suggestions', [])
                )
                gaps.append(gap)
            
            return gaps
            
        except Exception as e:
            print(f"⚠ Content gap analysis failed: {e}")
            return []
    
    def _get_domain_knowledge(self, field: str, paper_content: str) -> DomainKnowledge:
        """Get domain-specific knowledge for enhancement."""
        # Normalize field name
        field_lower = field.lower()
        
        # Map to known domains
        domain_mapping = {
            'ml': 'machine_learning',
            'ai': 'machine_learning',
            'deep learning': 'machine_learning',
            'neural networks': 'machine_learning',
            'statistics': 'statistics',
            'statistical': 'statistics'
        }
        
        mapped_field = None
        for key, value in domain_mapping.items():
            if key in field_lower:
                mapped_field = value
                break
        
        # Get knowledge base
        if mapped_field and mapped_field in self.domain_knowledge_base:
            kb = self.domain_knowledge_base[mapped_field]
            return DomainKnowledge(
                field=mapped_field,
                concepts=kb['concepts'],
                terminology=kb['terminology'],
                best_practices=kb['best_practices'],
                common_pitfalls=kb.get('common_pitfalls', [])
            )
        
        # Generic knowledge for unknown domains
        return DomainKnowledge(
            field=field,
            concepts=[],
            terminology={},
            best_practices=[
                'Provide clear methodology descriptions',
                'Include appropriate statistical analysis',
                'Discuss limitations and future work'
            ],
            common_pitfalls=[]
        )
    
    def _generate_enhancements(self, paper_content: str, style_analysis: WritingStyleAnalysis,
                             gaps: List[ContentGap], domain_knowledge: DomainKnowledge,
                             model: str, request_timeout: int) -> List[ContentEnhancement]:
        """Generate content enhancements."""
        enhancements = []
        
        # Style-based enhancements
        style_enhancements = self._generate_style_enhancements(
            paper_content, style_analysis
        )
        enhancements.extend(style_enhancements)
        
        # Gap-based enhancements
        gap_enhancements = self._generate_gap_enhancements(
            gaps, model, request_timeout
        )
        enhancements.extend(gap_enhancements)
        
        # Domain knowledge enhancements
        domain_enhancements = self._generate_domain_enhancements(
            paper_content, domain_knowledge
        )
        enhancements.extend(domain_enhancements)
        
        # Coherence enhancements
        coherence_enhancements = self._generate_coherence_enhancements(
            paper_content, model, request_timeout
        )
        enhancements.extend(coherence_enhancements)
        
        return enhancements
    
    def _generate_style_enhancements(self, paper_content: str, 
                                   style_analysis: WritingStyleAnalysis) -> List[ContentEnhancement]:
        """Generate style-based enhancements."""
        enhancements = []
        
        # Apply pattern-based enhancements
        for enhancement_type, patterns in self.enhancement_patterns.items():
            for old_pattern, new_pattern in patterns:
                matches = list(re.finditer(old_pattern, paper_content, re.IGNORECASE))
                
                for match in matches:
                    # Get surrounding context
                    start = max(0, match.start() - 50)
                    end = min(len(paper_content), match.end() + 50)
                    context = paper_content[start:end]
                    
                    enhancement = ContentEnhancement(
                        enhancement_type=enhancement_type,
                        location=f"Position {match.start()}",
                        original_text=match.group(0),
                        enhanced_text=re.sub(old_pattern, new_pattern, match.group(0), flags=re.IGNORECASE),
                        rationale=f"Improve {enhancement_type}",
                        confidence_score=0.7
                    )
                    enhancements.append(enhancement)
        
        # Passive voice reduction
        if style_analysis.passive_voice_ratio > 0.3:
            passive_enhancements = self._suggest_passive_voice_improvements(paper_content)
            enhancements.extend(passive_enhancements)
        
        return enhancements
    
    def _suggest_passive_voice_improvements(self, paper_content: str) -> List[ContentEnhancement]:
        """Suggest improvements for passive voice usage."""
        enhancements = []
        
        # Simple passive voice patterns
        passive_patterns = [
            (r'is conducted', 'we conduct'),
            (r'are performed', 'we perform'),
            (r'was implemented', 'we implemented'),
            (r'were evaluated', 'we evaluated')
        ]
        
        for old_pattern, new_pattern in passive_patterns:
            matches = list(re.finditer(old_pattern, paper_content, re.IGNORECASE))
            
            for match in matches[:5]:  # Limit suggestions
                enhancement = ContentEnhancement(
                    enhancement_type='active_voice',
                    location=f"Position {match.start()}",
                    original_text=match.group(0),
                    enhanced_text=new_pattern,
                    rationale="Convert passive voice to active voice for clarity",
                    confidence_score=0.8
                )
                enhancements.append(enhancement)
        
        return enhancements
    
    def _generate_gap_enhancements(self, gaps: List[ContentGap], model: str, 
                                 request_timeout: int) -> List[ContentEnhancement]:
        """Generate enhancements for identified content gaps."""
        enhancements = []
        
        for gap in gaps[:5]:  # Limit to top 5 gaps
            if gap.enhancement_suggestions:
                # Convert suggestions to enhancements
                for suggestion in gap.enhancement_suggestions:
                    enhancement = ContentEnhancement(
                        enhancement_type=gap.gap_type,
                        location=gap.location,
                        original_text="[Missing content]",
                        enhanced_text=f"Addition: {suggestion}",
                        rationale=gap.description,
                        confidence_score=0.6 if gap.severity == 'critical' else 0.4
                    )
                    enhancements.append(enhancement)
        
        return enhancements
    
    def _generate_domain_enhancements(self, paper_content: str, 
                                    domain_knowledge: DomainKnowledge) -> List[ContentEnhancement]:
        """Generate domain-specific enhancements."""
        enhancements = []
        
        # Terminology improvements
        for abbreviation, full_form in domain_knowledge.terminology.items():
            # Look for instances where abbreviation is used without definition
            pattern = r'\b' + re.escape(abbreviation) + r'\b'
            matches = list(re.finditer(pattern, paper_content))
            
            if matches:
                # Check if first instance is defined
                first_match = matches[0]
                context_before = paper_content[max(0, first_match.start()-100):first_match.start()]
                
                if full_form.lower() not in context_before.lower():
                    enhancement = ContentEnhancement(
                        enhancement_type='terminology',
                        location=f"First occurrence at position {first_match.start()}",
                        original_text=abbreviation,
                        enhanced_text=f"{full_form} ({abbreviation})",
                        rationale=f"Define {abbreviation} on first use",
                        confidence_score=0.8
                    )
                    enhancements.append(enhancement)
        
        return enhancements
    
    def _generate_coherence_enhancements(self, paper_content: str, model: str, 
                                       request_timeout: int) -> List[ContentEnhancement]:
        """Generate enhancements for logical coherence."""
        # This would analyze paragraph transitions and logical flow
        # For now, return simple transition improvements
        enhancements = []
        
        # Look for paragraphs without transitions
        paragraphs = paper_content.split('\n\n')
        
        for i, paragraph in enumerate(paragraphs[1:], 1):  # Skip first paragraph
            if len(paragraph.strip()) > 50:  # Skip short paragraphs
                first_sentence = paragraph.strip().split('.')[0]
                
                # Check if paragraph starts with transition
                transition_words = [
                    'however', 'furthermore', 'moreover', 'additionally',
                    'consequently', 'therefore', 'thus', 'meanwhile'
                ]
                
                if not any(word in first_sentence.lower() for word in transition_words):
                    enhancement = ContentEnhancement(
                        enhancement_type='coherence',
                        location=f"Paragraph {i}",
                        original_text=first_sentence[:50] + "...",
                        enhanced_text="[Add transition word] " + first_sentence[:50] + "...",
                        rationale="Improve paragraph transition",
                        confidence_score=0.5
                    )
                    enhancements.append(enhancement)
        
        return enhancements[:3]  # Limit to 3 suggestions
    
    def _calculate_improvement_score(self, enhancements: List[ContentEnhancement]) -> float:
        """Calculate overall improvement score."""
        if not enhancements:
            return 0.0
        
        # Weight by enhancement type and confidence
        type_weights = {
            'clarity': 1.0,
            'precision': 0.9,
            'formality': 0.7,
            'active_voice': 0.8,
            'terminology': 0.8,
            'coherence': 0.9,
            'logical': 1.0,
            'methodological': 0.9
        }
        
        total_score = 0.0
        total_weight = 0.0
        
        for enhancement in enhancements:
            weight = type_weights.get(enhancement.enhancement_type, 0.6)
            score = enhancement.confidence_score * weight
            total_score += score
            total_weight += weight
        
        return total_score / total_weight if total_weight > 0 else 0.0
    
    def _calculate_readability_improvement(self, style_analysis: WritingStyleAnalysis,
                                         enhancements: List[ContentEnhancement]) -> float:
        """Calculate readability improvement score."""
        base_readability = style_analysis.readability_score / 100.0
        
        # Count readability-improving enhancements
        readability_enhancements = [
            e for e in enhancements 
            if e.enhancement_type in ['clarity', 'active_voice', 'precision']
        ]
        
        improvement_factor = min(0.3, len(readability_enhancements) * 0.05)
        return min(1.0, base_readability + improvement_factor)
    
    def _calculate_coherence_improvement(self, enhancements: List[ContentEnhancement]) -> float:
        """Calculate coherence improvement score."""
        coherence_enhancements = [
            e for e in enhancements 
            if e.enhancement_type in ['coherence', 'logical', 'methodological']
        ]
        
        if not coherence_enhancements:
            return 0.0
        
        avg_confidence = sum(e.confidence_score for e in coherence_enhancements) / len(coherence_enhancements)
        return min(1.0, avg_confidence + len(coherence_enhancements) * 0.1)
    
    def _generate_enhancement_recommendations(self, style_analysis: WritingStyleAnalysis,
                                            gaps: List[ContentGap],
                                            enhancements: List[ContentEnhancement]) -> List[str]:
        """Generate recommendations for content enhancement."""
        recommendations = []
        
        # Style-based recommendations
        if style_analysis.avg_sentence_length > 30:
            recommendations.append("Reduce average sentence length for better readability")
        
        if style_analysis.passive_voice_ratio > 0.3:
            recommendations.append("Reduce passive voice usage - aim for more active constructions")
        
        if style_analysis.readability_score < 30:
            recommendations.append("Improve readability by simplifying complex sentences")
        
        if style_analysis.transition_word_usage < 0.05:
            recommendations.append("Add more transition words to improve flow between ideas")
        
        # Gap-based recommendations
        critical_gaps = [g for g in gaps if g.severity == 'critical']
        if critical_gaps:
            recommendations.append(f"Address {len(critical_gaps)} critical content gaps")
        
        # Enhancement-based recommendations
        high_confidence_enhancements = [e for e in enhancements if e.confidence_score >= 0.7]
        if high_confidence_enhancements:
            recommendations.append(f"Implement {len(high_confidence_enhancements)} high-confidence improvements")
        
        # General recommendations
        if not recommendations:
            recommendations.append("Content quality is good - consider minor stylistic improvements")
        
        return recommendations
    
    def _generate_enhancement_summary(self, enhancement_count: int, improvement_score: float,
                                    readability_improvement: float, coherence_improvement: float) -> str:
        """Generate content enhancement summary."""
        summary = []
        
        summary.append(f"Content Enhancement Summary:")
        summary.append(f"- Total enhancements suggested: {enhancement_count}")
        summary.append(f"- Overall improvement score: {improvement_score:.2f}")
        summary.append(f"- Readability improvement: {readability_improvement:.2f}")
        summary.append(f"- Coherence improvement: {coherence_improvement:.2f}")
        
        # Overall assessment
        if improvement_score >= 0.8 and readability_improvement >= 0.7:
            summary.append("- Enhancement potential: EXCELLENT")
        elif improvement_score >= 0.6 and readability_improvement >= 0.5:
            summary.append("- Enhancement potential: GOOD")
        elif improvement_score >= 0.4:
            summary.append("- Enhancement potential: MODERATE")
        else:
            summary.append("- Enhancement potential: LIMITED")
        
        return "\n".join(summary)
    
    def apply_enhancements(self, paper_content: str, 
                         enhancements: List[ContentEnhancement]) -> str:
        """Apply selected enhancements to paper content."""
        enhanced_content = paper_content
        
        # Sort enhancements by position (reverse order to avoid index shifts)
        position_enhancements = []
        for enhancement in enhancements:
            if enhancement.location.startswith("Position"):
                try:
                    pos = int(enhancement.location.split()[-1])
                    position_enhancements.append((pos, enhancement))
                except ValueError:
                    continue
        
        position_enhancements.sort(key=lambda x: x[0], reverse=True)
        
        # Apply position-based enhancements
        for pos, enhancement in position_enhancements:
            if enhancement.enhancement_type in ['clarity', 'precision', 'formality', 'active_voice']:
                # Simple text replacement
                original = enhancement.original_text
                enhanced = enhancement.enhanced_text
                
                # Find and replace the text
                start_pos = enhanced_content.find(original, pos - 50)
                if start_pos != -1:
                    enhanced_content = (enhanced_content[:start_pos] + 
                                      enhanced + 
                                      enhanced_content[start_pos + len(original):])
        
        return enhanced_content
    
    def generate_enhancement_report(self, result: ContentEnhancementResult) -> str:
        """Generate comprehensive content enhancement report."""
        report = []
        report.append("=" * 80)
        report.append("INTELLIGENT CONTENT ENHANCEMENT REPORT")
        report.append("=" * 80)
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")
        
        # Summary
        report.append("ENHANCEMENT SUMMARY:")
        for line in result.enhancement_summary.split('\n'):
            report.append(f"  {line}")
        report.append("")
        
        # Writing style analysis
        style = result.writing_style_analysis
        report.append("WRITING STYLE ANALYSIS:")
        report.append(f"  Average sentence length: {style.avg_sentence_length:.1f} words")
        report.append(f"  Vocabulary richness: {style.vocabulary_richness:.2f}")
        report.append(f"  Passive voice ratio: {style.passive_voice_ratio:.2f}")
        report.append(f"  Transition word usage: {style.transition_word_usage:.2f}")
        report.append(f"  Technical term density: {style.technical_term_density:.2f}")
        report.append(f"  Readability score: {style.readability_score:.1f}")
        report.append(f"  Consistency score: {style.consistency_score:.2f}")
        report.append("")
        
        # Content gaps
        if result.identified_gaps:
            report.append("IDENTIFIED CONTENT GAPS:")
            critical_gaps = [g for g in result.identified_gaps if g.severity == 'critical']
            important_gaps = [g for g in result.identified_gaps if g.severity == 'important']
            
            if critical_gaps:
                report.append("  CRITICAL:")
                for gap in critical_gaps:
                    report.append(f"    • {gap.description} ({gap.location})")
                    for suggestion in gap.enhancement_suggestions[:2]:
                        report.append(f"      → {suggestion}")
            
            if important_gaps:
                report.append("  IMPORTANT:")
                for gap in important_gaps:
                    report.append(f"    • {gap.description} ({gap.location})")
            
            report.append("")
        
        # Top enhancements
        if result.content_enhancements:
            high_confidence = [e for e in result.content_enhancements if e.confidence_score >= 0.7]
            if high_confidence:
                report.append("HIGH-CONFIDENCE ENHANCEMENTS:")
                for enhancement in high_confidence[:10]:  # Show top 10
                    report.append(f"  • {enhancement.enhancement_type.title()}: {enhancement.rationale}")
                    report.append(f"    Original: \"{enhancement.original_text[:50]}...\"")
                    report.append(f"    Enhanced: \"{enhancement.enhanced_text[:50]}...\"")
                    report.append(f"    Confidence: {enhancement.confidence_score:.2f}")
                report.append("")
        
        # Recommendations
        if result.recommendations:
            report.append("RECOMMENDATIONS:")
            for rec in result.recommendations:
                report.append(f"  • {rec}")
            report.append("")
        
        report.append("=" * 80)
        
        return "\n".join(report)
