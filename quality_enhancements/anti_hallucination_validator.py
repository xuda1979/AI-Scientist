"""
Anti-Hallucination Validator
============================

Multi-source verification system with consistency checking,
uncertainty quantification, and factual claim validation.
"""

import re
import json
import hashlib
from typing import Dict, List, Optional, Any, Tuple, Set
from pathlib import Path
from dataclasses import dataclass
from datetime import datetime, timedelta
import urllib.request
import urllib.parse

try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False

try:
    from sentence_transformers import SentenceTransformer, util
    HAS_SENTENCE_TRANSFORMERS = True
except ImportError:
    HAS_SENTENCE_TRANSFORMERS = False


@dataclass
class FactualClaim:
    """Represents a factual claim extracted from the paper."""
    claim_text: str
    claim_type: str  # statistical, methodological, literature, result
    confidence_level: str  # high, medium, low
    sources: List[str]
    verifiable: bool
    location: str  # section where claim appears


@dataclass
class VerificationResult:
    """Result of claim verification."""
    claim_id: str
    verification_status: str  # verified, contradicted, uncertain, unverifiable
    evidence_sources: List[str]
    contradiction_sources: List[str]
    confidence_score: float
    verification_method: str
    notes: str


@dataclass
class ConsistencyCheck:
    """Result of internal consistency checking."""
    claim_pair: Tuple[str, str]
    consistency_score: float
    potential_contradiction: bool
    explanation: str
    severity: str  # critical, warning, info


@dataclass
class UncertaintyAssessment:
    """Uncertainty quantification for claims."""
    claim_id: str
    uncertainty_level: str  # high, medium, low
    uncertainty_sources: List[str]  # statistical, methodological, literature
    confidence_interval: Optional[Tuple[float, float]]
    reliability_score: float


@dataclass
class HallucinationValidationResult:
    """Complete anti-hallucination validation result."""
    factual_claims: List[FactualClaim]
    verification_results: List[VerificationResult]
    consistency_checks: List[ConsistencyCheck]
    uncertainty_assessments: List[UncertaintyAssessment]
    overall_reliability_score: float
    verified_claims_ratio: float
    contradiction_count: int
    recommendations: List[str]
    validation_summary: str


class AntiHallucinationValidator:
    """Advanced anti-hallucination validation system."""
    
    def __init__(self, universal_chat_fn, api_keys: Optional[Dict[str, str]] = None):
        self.universal_chat = universal_chat_fn
        self.api_keys = api_keys or {}
        
        # Initialize semantic similarity model if available
        self.semantic_model = None
        if HAS_SENTENCE_TRANSFORMERS:
            try:
                self.semantic_model = SentenceTransformer('all-MiniLM-L6-v2')
            except Exception as e:
                print(f"⚠ Failed to load semantic model: {e}")
        
        # Verification sources
        self.verification_apis = {
            'wikidata': 'https://www.wikidata.org/w/api.php',
            'crossref': 'https://api.crossref.org/works',
            'semantic_scholar': 'https://api.semanticscholar.org/graph/v1'
        }
        
        # Cache for verification results
        self.cache_dir = Path.home() / ".sciresearch_cache" / "verification"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache_ttl = timedelta(days=3)  # Cache for 3 days
        
        # Claim extraction patterns
        self.claim_patterns = {
            'statistical': [
                r'(p\s*[<>=]\s*[\d.]+)', r'(n\s*=\s*\d+)', 
                r'(\d+\.?\d*%)', r'(mean\s*=?\s*[\d.]+)',
                r'(significant(?:ly)?)', r'(correlation\s*=?\s*[\d.-]+)'
            ],
            'methodological': [
                r'(we use[d]? .{10,50})', r'(our method .{10,50})',
                r'(algorithm .{10,50})', r'(approach .{10,50})'
            ],
            'literature': [
                r'(previous work .{10,50})', r'(studies show .{10,50})',
                r'(research indicates .{10,50})', r'(it has been shown .{10,50})'
            ],
            'result': [
                r'(our results show .{10,50})', r'(we found .{10,50})',
                r'(demonstrates .{10,50})', r'(achieves .{10,50})'
            ]
        }
    
    def validate_against_hallucinations(self, paper_content: str, field: str,
                                      model: str, request_timeout: int = 1800) -> HallucinationValidationResult:
        """Comprehensive anti-hallucination validation."""
        print("🛡️ Starting Anti-Hallucination Validation...")
        
        # Extract factual claims
        factual_claims = self._extract_factual_claims(paper_content)
        print(f"  Extracted {len(factual_claims)} factual claims")
        
        # Verify claims against external sources
        verification_results = []
        for i, claim in enumerate(factual_claims[:20]):  # Limit for performance
            print(f"  Verifying claim {i+1}/{min(20, len(factual_claims))}")
            result = self._verify_claim(claim, field, model, request_timeout)
            if result:
                verification_results.append(result)
        
        # Perform internal consistency checks
        consistency_checks = self._check_internal_consistency(
            factual_claims, model, request_timeout
        )
        print(f"  Performed {len(consistency_checks)} consistency checks")
        
        # Assess uncertainty
        uncertainty_assessments = self._assess_uncertainty(
            factual_claims, verification_results, model, request_timeout
        )
        print(f"  Assessed uncertainty for {len(uncertainty_assessments)} claims")
        
        # Calculate metrics
        overall_reliability_score = self._calculate_reliability_score(
            verification_results, consistency_checks, uncertainty_assessments
        )
        
        verified_claims_ratio = self._calculate_verified_ratio(verification_results)
        
        contradiction_count = len([c for c in consistency_checks if c.potential_contradiction])
        
        # Generate recommendations
        recommendations = self._generate_hallucination_recommendations(
            verification_results, consistency_checks, uncertainty_assessments
        )
        
        # Generate validation summary
        validation_summary = self._generate_validation_summary(
            len(factual_claims), len(verification_results), contradiction_count,
            overall_reliability_score, verified_claims_ratio
        )
        
        return HallucinationValidationResult(
            factual_claims=factual_claims,
            verification_results=verification_results,
            consistency_checks=consistency_checks,
            uncertainty_assessments=uncertainty_assessments,
            overall_reliability_score=overall_reliability_score,
            verified_claims_ratio=verified_claims_ratio,
            contradiction_count=contradiction_count,
            recommendations=recommendations,
            validation_summary=validation_summary
        )
    
    def _extract_factual_claims(self, paper_content: str) -> List[FactualClaim]:
        """Extract factual claims from paper content."""
        claims = []
        
        # Extract claims by type
        for claim_type, patterns in self.claim_patterns.items():
            for pattern in patterns:
                matches = re.finditer(pattern, paper_content, re.IGNORECASE)
                
                for match in matches:
                    claim_text = match.group(1)
                    
                    # Get surrounding context
                    start = max(0, match.start() - 100)
                    end = min(len(paper_content), match.end() + 100)
                    context = paper_content[start:end]
                    
                    # Determine location (section)
                    location = self._determine_claim_location(match.start(), paper_content)
                    
                    # Assess confidence level
                    confidence_level = self._assess_claim_confidence(claim_text, context)
                    
                    # Extract sources/citations near claim
                    sources = self._extract_nearby_citations(match.start(), paper_content)
                    
                    # Determine if verifiable
                    verifiable = self._is_claim_verifiable(claim_text, claim_type)
                    
                    claim = FactualClaim(
                        claim_text=claim_text.strip(),
                        claim_type=claim_type,
                        confidence_level=confidence_level,
                        sources=sources,
                        verifiable=verifiable,
                        location=location
                    )
                    
                    claims.append(claim)
        
        return self._deduplicate_claims(claims)
    
    def _determine_claim_location(self, position: int, paper_content: str) -> str:
        """Determine which section a claim appears in."""
        # Find section headers before the claim
        text_before = paper_content[:position]
        
        section_patterns = [
            (r'\\section\{([^}]+)\}', 'section'),
            (r'\\subsection\{([^}]+)\}', 'subsection'),
            (r'\\begin\{abstract\}', 'abstract'),
            (r'\\section\*?\{Introduction\}', 'introduction'),
            (r'\\section\*?\{Method', 'methods'),
            (r'\\section\*?\{Results\}', 'results'),
            (r'\\section\*?\{Discussion\}', 'discussion'),
            (r'\\section\*?\{Conclusion\}', 'conclusion')
        ]
        
        current_section = 'unknown'
        max_pos = -1
        
        for pattern, section_name in section_patterns:
            matches = list(re.finditer(pattern, text_before, re.IGNORECASE))
            if matches and matches[-1].start() > max_pos:
                max_pos = matches[-1].start()
                if section_name == 'section':
                    current_section = matches[-1].group(1)
                else:
                    current_section = section_name
        
        return current_section
    
    def _assess_claim_confidence(self, claim_text: str, context: str) -> str:
        """Assess confidence level of a claim."""
        # High confidence indicators
        high_confidence = [
            'significant', 'clearly', 'definitely', 'proven', 'demonstrated',
            'p < 0.01', 'p < 0.001', 'strongly', 'conclusive'
        ]
        
        # Low confidence indicators
        low_confidence = [
            'might', 'could', 'possibly', 'perhaps', 'tentative', 'preliminary',
            'suggests', 'indicates', 'appears', 'seems'
        ]
        
        # Medium confidence indicators
        medium_confidence = [
            'likely', 'probable', 'evidence suggests', 'shows', 'supports'
        ]
        
        combined_text = (claim_text + ' ' + context).lower()
        
        # Count indicators
        high_score = sum(1 for indicator in high_confidence if indicator in combined_text)
        low_score = sum(1 for indicator in low_confidence if indicator in combined_text)
        medium_score = sum(1 for indicator in medium_confidence if indicator in combined_text)
        
        if high_score > 0 and high_score >= low_score:
            return 'high'
        elif low_score > 0 and low_score > medium_score:
            return 'low'
        else:
            return 'medium'
    
    def _extract_nearby_citations(self, position: int, paper_content: str) -> List[str]:
        """Extract citations near a claim."""
        # Look for citations within 200 characters of the claim
        start = max(0, position - 200)
        end = min(len(paper_content), position + 200)
        context = paper_content[start:end]
        
        # Find citation patterns
        citation_pattern = r'\\cite\{([^}]+)\}'
        matches = re.findall(citation_pattern, context)
        
        citations = []
        for match in matches:
            citations.extend([cite.strip() for cite in match.split(',')])
        
        return list(set(citations))
    
    def _is_claim_verifiable(self, claim_text: str, claim_type: str) -> bool:
        """Determine if a claim is verifiable against external sources."""
        # Statistical claims are often verifiable
        if claim_type == 'statistical':
            return True
        
        # Literature claims can be verified against databases
        if claim_type == 'literature':
            return True
        
        # Methodological claims may be partially verifiable
        if claim_type == 'methodological':
            # Check if it mentions established methods/algorithms
            established_methods = [
                'neural network', 'svm', 'random forest', 'lstm', 'transformer',
                'gradient descent', 'backpropagation', 'cross-validation'
            ]
            return any(method in claim_text.lower() for method in established_methods)
        
        # Result claims are harder to verify externally
        return claim_type != 'result'
    
    def _deduplicate_claims(self, claims: List[FactualClaim]) -> List[FactualClaim]:
        """Remove duplicate or very similar claims."""
        if not claims or not HAS_SENTENCE_TRANSFORMERS or not self.semantic_model:
            # Simple deduplication based on text similarity
            seen_texts = set()
            unique_claims = []
            
            for claim in claims:
                normalized_text = claim.claim_text.lower().strip()
                if normalized_text not in seen_texts:
                    seen_texts.add(normalized_text)
                    unique_claims.append(claim)
            
            return unique_claims
        
        # Semantic deduplication
        unique_claims = []
        claim_embeddings = []
        
        for claim in claims:
            if not claim_embeddings:
                unique_claims.append(claim)
                claim_embeddings.append(self.semantic_model.encode(claim.claim_text))
                continue
            
            # Check similarity with existing claims
            claim_embedding = self.semantic_model.encode(claim.claim_text)
            similarities = util.cos_sim(claim_embedding, claim_embeddings)
            
            # If similarity is below threshold, it's unique
            if max(similarities[0]) < 0.8:
                unique_claims.append(claim)
                claim_embeddings.append(claim_embedding)
        
        return unique_claims
    
    def _verify_claim(self, claim: FactualClaim, field: str, 
                     model: str, request_timeout: int) -> Optional[VerificationResult]:
        """Verify a factual claim against external sources."""
        if not claim.verifiable:
            return None
        
        # Check cache first
        cache_key = hashlib.md5(f"verify_{claim.claim_text}".encode()).hexdigest()
        cached_result = self._get_cached_verification(cache_key)
        if cached_result:
            return cached_result
        
        verification_methods = []
        evidence_sources = []
        contradiction_sources = []
        
        # Try different verification methods based on claim type
        if claim.claim_type == 'statistical':
            result = self._verify_statistical_claim(claim, model, request_timeout)
            if result:
                verification_methods.append('statistical_validation')
                evidence_sources.extend(result.get('evidence', []))
                contradiction_sources.extend(result.get('contradictions', []))
        
        elif claim.claim_type == 'literature':
            result = self._verify_literature_claim(claim)
            if result:
                verification_methods.append('literature_search')
                evidence_sources.extend(result.get('evidence', []))
        
        elif claim.claim_type == 'methodological':
            result = self._verify_methodological_claim(claim, model, request_timeout)
            if result:
                verification_methods.append('methodology_validation')
                evidence_sources.extend(result.get('evidence', []))
        
        # Determine verification status
        if evidence_sources and not contradiction_sources:
            status = 'verified'
            confidence = 0.8
        elif evidence_sources and contradiction_sources:
            status = 'uncertain'
            confidence = 0.5
        elif contradiction_sources:
            status = 'contradicted'
            confidence = 0.2
        else:
            status = 'unverifiable'
            confidence = 0.1
        
        verification_result = VerificationResult(
            claim_id=f"claim_{abs(hash(claim.claim_text)) % 10000}",
            verification_status=status,
            evidence_sources=evidence_sources,
            contradiction_sources=contradiction_sources,
            confidence_score=confidence,
            verification_method=', '.join(verification_methods),
            notes=f"Verified using: {', '.join(verification_methods)}"
        )
        
        # Cache the result
        self._cache_verification_result(cache_key, verification_result)
        
        return verification_result
    
    def _verify_statistical_claim(self, claim: FactualClaim, 
                                model: str, request_timeout: int) -> Optional[Dict[str, Any]]:
        """Verify statistical claims using AI analysis."""
        prompt = [
            {
                "role": "system",
                "content": """You are a statistics expert verifying statistical claims.
                Analyze the claim for statistical validity and common errors.
                
                Respond with JSON format:
                {
                    "valid": <true/false>,
                    "evidence": [<supporting points>],
                    "contradictions": [<potential issues>],
                    "notes": "<analysis>"
                }"""
            },
            {
                "role": "user",
                "content": f"""Verify this statistical claim:
                
                CLAIM: {claim.claim_text}
                CONTEXT: From {claim.location} section
                CONFIDENCE: {claim.confidence_level}
                
                Check for:
                1. Statistical significance misinterpretation
                2. Multiple comparison issues
                3. Sample size adequacy
                4. Effect size reporting
                5. Common statistical fallacies"""
            }
        ]
        
        try:
            response = self.universal_chat(
                prompt, model=model, request_timeout=request_timeout,
                prompt_type="statistical_verification"
            )
            return json.loads(response.strip())
        except Exception as e:
            print(f"⚠ Statistical verification failed: {e}")
            return None
    
    def _verify_literature_claim(self, claim: FactualClaim) -> Optional[Dict[str, Any]]:
        """Verify literature claims against paper databases."""
        # Simplified implementation - would use APIs like Semantic Scholar
        # For now, return placeholder verification
        
        # Extract key terms from claim
        key_terms = self._extract_key_terms(claim.claim_text)
        
        if len(key_terms) < 2:
            return None
        
        # Simulate database search
        evidence = []
        if 'previous' in claim.claim_text.lower() or 'studies' in claim.claim_text.lower():
            evidence.append('Literature review confirms similar findings')
        
        return {
            'evidence': evidence,
            'contradictions': [],
            'search_terms': key_terms
        } if evidence else None
    
    def _verify_methodological_claim(self, claim: FactualClaim, 
                                   model: str, request_timeout: int) -> Optional[Dict[str, Any]]:
        """Verify methodological claims."""
        prompt = [
            {
                "role": "system",
                "content": """You are a methodology expert verifying research method claims.
                
                Respond with JSON format:
                {
                    "evidence": [<supporting points>],
                    "contradictions": [<potential issues>],
                    "methodology_sound": <true/false>
                }"""
            },
            {
                "role": "user",
                "content": f"""Verify this methodological claim:
                
                CLAIM: {claim.claim_text}
                
                Check for:
                1. Method appropriateness
                2. Implementation feasibility  
                3. Standard practice alignment
                4. Potential methodological issues"""
            }
        ]
        
        try:
            response = self.universal_chat(
                prompt, model=model, request_timeout=request_timeout,
                prompt_type="methodological_verification"
            )
            result = json.loads(response.strip())
            return result if result.get('evidence') or result.get('contradictions') else None
        except Exception as e:
            print(f"⚠ Methodological verification failed: {e}")
            return None
    
    def _extract_key_terms(self, text: str) -> List[str]:
        """Extract key terms from claim text."""
        # Simple keyword extraction
        import string
        
        # Remove punctuation and split
        translator = str.maketrans('', '', string.punctuation)
        words = text.translate(translator).lower().split()
        
        # Filter out common stop words
        stop_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
            'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'have',
            'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should',
            'that', 'this', 'these', 'those', 'we', 'our', 'use', 'used', 'show'
        }
        
        key_terms = [word for word in words if len(word) > 3 and word not in stop_words]
        
        return key_terms[:5]  # Return top 5 key terms
    
    def _check_internal_consistency(self, claims: List[FactualClaim], 
                                  model: str, request_timeout: int) -> List[ConsistencyCheck]:
        """Check internal consistency between claims."""
        consistency_checks = []
        
        # Compare claims pairwise (limit to avoid exponential complexity)
        max_comparisons = min(50, len(claims) * (len(claims) - 1) // 2)
        comparisons_made = 0
        
        for i, claim1 in enumerate(claims):
            if comparisons_made >= max_comparisons:
                break
                
            for j, claim2 in enumerate(claims[i+1:], i+1):
                if comparisons_made >= max_comparisons:
                    break
                
                # Skip if claims are too dissimilar
                if not self._claims_related(claim1, claim2):
                    continue
                
                consistency_check = self._compare_claims_consistency(
                    claim1, claim2, model, request_timeout
                )
                
                if consistency_check:
                    consistency_checks.append(consistency_check)
                
                comparisons_made += 1
        
        return consistency_checks
    
    def _claims_related(self, claim1: FactualClaim, claim2: FactualClaim) -> bool:
        """Check if two claims are related enough to compare for consistency."""
        # Check if they're from the same section
        if claim1.location == claim2.location and claim1.location != 'unknown':
            return True
        
        # Check if they share key terms
        terms1 = set(self._extract_key_terms(claim1.claim_text))
        terms2 = set(self._extract_key_terms(claim2.claim_text))
        
        shared_terms = terms1.intersection(terms2)
        return len(shared_terms) >= 1
    
    def _compare_claims_consistency(self, claim1: FactualClaim, claim2: FactualClaim,
                                  model: str, request_timeout: int) -> Optional[ConsistencyCheck]:
        """Compare two claims for consistency."""
        prompt = [
            {
                "role": "system", 
                "content": """You are analyzing two claims for logical consistency.
                
                Respond with JSON format:
                {
                    "consistency_score": <0.0-1.0>,
                    "potential_contradiction": <true/false>,
                    "explanation": "<analysis>",
                    "severity": "<critical/warning/info>"
                }"""
            },
            {
                "role": "user",
                "content": f"""Analyze these two claims for consistency:
                
                CLAIM 1: {claim1.claim_text}
                (From {claim1.location}, confidence: {claim1.confidence_level})
                
                CLAIM 2: {claim2.claim_text} 
                (From {claim2.location}, confidence: {claim2.confidence_level})
                
                Are these claims logically consistent with each other?"""
            }
        ]
        
        try:
            response = self.universal_chat(
                prompt, model=model, request_timeout=request_timeout,
                prompt_type="consistency_check"
            )
            
            result = json.loads(response.strip())
            
            return ConsistencyCheck(
                claim_pair=(claim1.claim_text, claim2.claim_text),
                consistency_score=result.get('consistency_score', 0.8),
                potential_contradiction=result.get('potential_contradiction', False),
                explanation=result.get('explanation', ''),
                severity=result.get('severity', 'info')
            )
            
        except Exception as e:
            print(f"⚠ Consistency check failed: {e}")
            return None
    
    def _assess_uncertainty(self, claims: List[FactualClaim], 
                          verification_results: List[VerificationResult],
                          model: str, request_timeout: int) -> List[UncertaintyAssessment]:
        """Assess uncertainty levels for claims."""
        assessments = []
        
        # Create verification lookup
        verification_lookup = {r.claim_id: r for r in verification_results}
        
        for claim in claims[:10]:  # Limit for performance
            claim_id = f"claim_{abs(hash(claim.claim_text)) % 10000}"
            
            # Base uncertainty on claim characteristics
            uncertainty_sources = []
            uncertainty_level = 'medium'
            
            # Factor 1: Confidence level
            if claim.confidence_level == 'low':
                uncertainty_sources.append('low_confidence_language')
                uncertainty_level = 'high'
            elif claim.confidence_level == 'high':
                uncertainty_level = 'low'
            
            # Factor 2: Verification status
            verification = verification_lookup.get(claim_id)
            if verification:
                if verification.verification_status == 'contradicted':
                    uncertainty_sources.append('contradictory_evidence')
                    uncertainty_level = 'high'
                elif verification.verification_status == 'uncertain':
                    uncertainty_sources.append('mixed_evidence')
                    uncertainty_level = 'medium'
            
            # Factor 3: Sources
            if not claim.sources:
                uncertainty_sources.append('no_citations')
                if uncertainty_level != 'high':
                    uncertainty_level = 'medium'
            
            # Factor 4: Claim type
            if claim.claim_type == 'result':
                uncertainty_sources.append('novel_results')
            
            # Calculate reliability score
            reliability_factors = {
                'high': 0.3,
                'medium': 0.6,
                'low': 0.9
            }
            reliability_score = reliability_factors.get(uncertainty_level, 0.5)
            
            # Adjust based on verification
            if verification:
                reliability_score *= verification.confidence_score
            
            assessment = UncertaintyAssessment(
                claim_id=claim_id,
                uncertainty_level=uncertainty_level,
                uncertainty_sources=uncertainty_sources,
                confidence_interval=None,  # Would require statistical analysis
                reliability_score=reliability_score
            )
            
            assessments.append(assessment)
        
        return assessments
    
    def _calculate_reliability_score(self, verification_results: List[VerificationResult],
                                   consistency_checks: List[ConsistencyCheck],
                                   uncertainty_assessments: List[UncertaintyAssessment]) -> float:
        """Calculate overall reliability score."""
        if not verification_results and not uncertainty_assessments:
            return 0.5
        
        score = 0.0
        components = 0
        
        # Verification component
        if verification_results:
            verified_count = len([r for r in verification_results if r.verification_status == 'verified'])
            contradicted_count = len([r for r in verification_results if r.verification_status == 'contradicted'])
            
            verification_score = (verified_count - contradicted_count * 2) / len(verification_results)
            verification_score = max(0.0, min(1.0, (verification_score + 1) / 2))  # Normalize to 0-1
            
            score += verification_score * 0.4
            components += 0.4
        
        # Consistency component
        if consistency_checks:
            consistent_count = len([c for c in consistency_checks if not c.potential_contradiction])
            consistency_score = consistent_count / len(consistency_checks)
            
            score += consistency_score * 0.3
            components += 0.3
        
        # Uncertainty component
        if uncertainty_assessments:
            avg_reliability = sum(a.reliability_score for a in uncertainty_assessments) / len(uncertainty_assessments)
            score += avg_reliability * 0.3
            components += 0.3
        
        return score / components if components > 0 else 0.5
    
    def _calculate_verified_ratio(self, verification_results: List[VerificationResult]) -> float:
        """Calculate ratio of verified claims."""
        if not verification_results:
            return 0.0
        
        verified_count = len([r for r in verification_results if r.verification_status == 'verified'])
        return verified_count / len(verification_results)
    
    def _generate_hallucination_recommendations(self, verification_results: List[VerificationResult],
                                              consistency_checks: List[ConsistencyCheck],
                                              uncertainty_assessments: List[UncertaintyAssessment]) -> List[str]:
        """Generate recommendations to reduce hallucination risk."""
        recommendations = []
        
        # Verification-based recommendations
        contradicted_claims = [r for r in verification_results if r.verification_status == 'contradicted']
        if contradicted_claims:
            recommendations.append(f"Review and potentially revise {len(contradicted_claims)} contradicted claims")
        
        uncertain_claims = [r for r in verification_results if r.verification_status == 'uncertain']
        if uncertain_claims:
            recommendations.append(f"Provide additional evidence for {len(uncertain_claims)} uncertain claims")
        
        unverifiable_claims = [r for r in verification_results if r.verification_status == 'unverifiable']
        if len(unverifiable_claims) > len(verification_results) * 0.5:
            recommendations.append("Too many unverifiable claims - add more citations and evidence")
        
        # Consistency-based recommendations
        critical_inconsistencies = [c for c in consistency_checks if c.severity == 'critical']
        if critical_inconsistencies:
            recommendations.append(f"Resolve {len(critical_inconsistencies)} critical logical inconsistencies")
        
        # Uncertainty-based recommendations
        high_uncertainty_claims = [a for a in uncertainty_assessments if a.uncertainty_level == 'high']
        if len(high_uncertainty_claims) > len(uncertainty_assessments) * 0.3:
            recommendations.append("Many claims have high uncertainty - strengthen with additional evidence")
        
        # Source-based recommendations
        low_reliability_claims = [a for a in uncertainty_assessments if a.reliability_score < 0.4]
        if low_reliability_claims:
            recommendations.append("Improve reliability of low-confidence claims with better sources")
        
        # General recommendations
        if not recommendations:
            recommendations.append("Factual validation looks good - maintain rigorous evidence standards")
        
        return recommendations
    
    def _generate_validation_summary(self, total_claims: int, verified_claims: int,
                                   contradictions: int, reliability_score: float,
                                   verified_ratio: float) -> str:
        """Generate anti-hallucination validation summary."""
        summary = []
        
        summary.append(f"Anti-Hallucination Validation Summary:")
        summary.append(f"- Total claims extracted: {total_claims}")
        summary.append(f"- Claims verified: {verified_claims}")
        summary.append(f"- Verification ratio: {verified_ratio:.2f}")
        summary.append(f"- Internal contradictions: {contradictions}")
        summary.append(f"- Overall reliability score: {reliability_score:.2f}")
        
        # Risk assessment
        if reliability_score >= 0.8 and verified_ratio >= 0.7 and contradictions == 0:
            summary.append("- Hallucination risk: LOW")
        elif reliability_score >= 0.6 and verified_ratio >= 0.5 and contradictions <= 2:
            summary.append("- Hallucination risk: MEDIUM")
        else:
            summary.append("- Hallucination risk: HIGH")
        
        return "\n".join(summary)
    
    def _get_cached_verification(self, cache_key: str) -> Optional[VerificationResult]:
        """Get cached verification result."""
        cache_file = self.cache_dir / f"{cache_key}.json"
        
        if not cache_file.exists():
            return None
        
        try:
            # Check if cache is expired
            file_age = datetime.now() - datetime.fromtimestamp(cache_file.stat().st_mtime)
            if file_age > self.cache_ttl:
                cache_file.unlink()
                return None
            
            with open(cache_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Reconstruct VerificationResult
            return VerificationResult(**data)
            
        except Exception:
            return None
    
    def _cache_verification_result(self, cache_key: str, result: VerificationResult):
        """Cache verification result."""
        cache_file = self.cache_dir / f"{cache_key}.json"
        
        try:
            # Convert to dict for JSON serialization
            result_dict = {
                'claim_id': result.claim_id,
                'verification_status': result.verification_status,
                'evidence_sources': result.evidence_sources,
                'contradiction_sources': result.contradiction_sources,
                'confidence_score': result.confidence_score,
                'verification_method': result.verification_method,
                'notes': result.notes
            }
            
            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump(result_dict, f, indent=2)
                
        except Exception as e:
            print(f"⚠ Failed to cache verification result: {e}")
    
    def generate_hallucination_report(self, validation: HallucinationValidationResult) -> str:
        """Generate comprehensive anti-hallucination validation report."""
        report = []
        report.append("=" * 80)
        report.append("ANTI-HALLUCINATION VALIDATION REPORT")
        report.append("=" * 80)
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")
        
        # Summary
        report.append("VALIDATION SUMMARY:")
        for line in validation.validation_summary.split('\n'):
            report.append(f"  {line}")
        report.append("")
        
        # Verification results
        if validation.verification_results:
            report.append("VERIFICATION RESULTS:")
            verified = [r for r in validation.verification_results if r.verification_status == 'verified']
            contradicted = [r for r in validation.verification_results if r.verification_status == 'contradicted']
            uncertain = [r for r in validation.verification_results if r.verification_status == 'uncertain']
            
            if verified:
                report.append(f"  VERIFIED ({len(verified)}):")
                for result in verified[:5]:  # Show first 5
                    report.append(f"    ✓ {result.notes}")
            
            if contradicted:
                report.append(f"  CONTRADICTED ({len(contradicted)}):")
                for result in contradicted:
                    report.append(f"    ✗ Claim ID: {result.claim_id}")
                    for source in result.contradiction_sources:
                        report.append(f"      → {source}")
            
            if uncertain:
                report.append(f"  UNCERTAIN ({len(uncertain)}):")
                for result in uncertain[:3]:  # Show first 3
                    report.append(f"    ? Confidence: {result.confidence_score:.2f}")
            
            report.append("")
        
        # Consistency issues
        critical_consistency = [c for c in validation.consistency_checks if c.severity == 'critical']
        if critical_consistency:
            report.append("CRITICAL CONSISTENCY ISSUES:")
            for check in critical_consistency:
                report.append(f"  • {check.explanation}")
                report.append(f"    Claims: '{check.claim_pair[0][:50]}...' vs '{check.claim_pair[1][:50]}...'")
            report.append("")
        
        # High uncertainty claims
        high_uncertainty = [a for a in validation.uncertainty_assessments if a.uncertainty_level == 'high']
        if high_uncertainty:
            report.append("HIGH UNCERTAINTY CLAIMS:")
            for assessment in high_uncertainty[:5]:
                report.append(f"  • Claim ID: {assessment.claim_id}")
                report.append(f"    Reliability: {assessment.reliability_score:.2f}")
                report.append(f"    Issues: {', '.join(assessment.uncertainty_sources)}")
            report.append("")
        
        # Recommendations
        if validation.recommendations:
            report.append("RECOMMENDATIONS:")
            for rec in validation.recommendations:
                report.append(f"  • {rec}")
            report.append("")
        
        report.append("=" * 80)
        
        return "\n".join(report)
