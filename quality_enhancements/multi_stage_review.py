"""
Multi-Stage Review System
========================

Advanced peer review simulation with multiple specialized reviewers
including academic, domain-specific, statistical, and ethical perspectives.
"""

import re
import json
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
from dataclasses import dataclass
from datetime import datetime


@dataclass
class ReviewResult:
    """Result from a single reviewer, including rubric-based scoring and evidence."""
    reviewer_type: str
    overall_recommendation: str  # ACCEPT, MINOR_REVISION, MAJOR_REVISION, REJECT
    confidence_score: float  # 0.0 to 1.0
    strengths: List[str]
    weaknesses: List[str]
    specific_comments: List[str]
    questions_for_authors: List[str]
    detailed_review: str
    review_duration_estimate: str  # e.g., "15 minutes"
    criterion_scores: Dict[str, float] = None  # e.g., {"novelty": 0.8, "rigor": 0.7}
    evidence_justification: Dict[str, str] = None  # e.g., {"novelty": "New approach to..."}


@dataclass
class ConsolidatedReview:
    """Consolidated review from multiple reviewers."""
    overall_decision: str
    consensus_level: float  # 0.0 to 1.0
    individual_reviews: List[ReviewResult]
    consensus_strengths: List[str]
    consensus_weaknesses: List[str]
    priority_issues: List[str]
    revision_roadmap: List[str]
    meta_analysis: str


class MultiStageReviewer:
    """Multi-stage review system with specialized reviewers."""
    
    def __init__(self, universal_chat_fn, fallback_models: Optional[List[str]] = None):
        self.universal_chat = universal_chat_fn
        self.fallback_models = fallback_models or ["gpt-4o", "gpt-4"]
        
        # Review criteria weights for different paper types
        self.paper_type_weights = {
            'theoretical': {
                'mathematical_rigor': 0.25,
                'proof_quality': 0.20,
                'novelty': 0.20,
                'significance': 0.15,
                'clarity': 0.15,
                'related_work': 0.05
            },
            'experimental': {
                'methodology': 0.25,
                'experimental_design': 0.20,
                'statistical_analysis': 0.20,
                'reproducibility': 0.15,
                'novelty': 0.10,
                'significance': 0.10
            },
            'survey': {
                'comprehensiveness': 0.30,
                'organization': 0.20,
                'critical_analysis': 0.20,
                'currency': 0.15,
                'synthesis': 0.15
            },
            'systems': {
                'technical_contribution': 0.25,
                'evaluation': 0.25,
                'implementation_quality': 0.20,
                'performance': 0.15,
                'scalability': 0.15
            }
        }
    
    def conduct_multi_stage_review(self, paper_content: str, field: str, paper_type: str,
                                 model: str, request_timeout: int = 1800,
                                 project_dir: Optional[Path] = None,
                                 sim_summary: str = "") -> ConsolidatedReview:
        """Conduct comprehensive multi-stage review."""
        print("🔍 Starting Multi-Stage Review Process...")
        
        reviews = []
        
        # Stage 1: Academic Review
        print("  Stage 1: Academic Review")
        academic_review = self._conduct_academic_review(
            paper_content, field, paper_type, model, request_timeout
        )
        reviews.append(academic_review)
        
        # Stage 2: Domain Expert Review
        print("  Stage 2: Domain Expert Review")
        domain_review = self._conduct_domain_expert_review(
            paper_content, field, model, request_timeout
        )
        reviews.append(domain_review)
        
        # Stage 3: Statistical Review (if experimental)
        if self._is_experimental_paper(paper_content):
            print("  Stage 3: Statistical Review")
            statistical_review = self._conduct_statistical_review(
                paper_content, model, request_timeout, sim_summary
            )
            reviews.append(statistical_review)
        
        # Stage 4: Ethics Review (if applicable)
        if self._requires_ethics_review(paper_content):
            print("  Stage 4: Ethics Review")
            ethics_review = self._conduct_ethics_review(
                paper_content, model, request_timeout
            )
            reviews.append(ethics_review)
        
        # Stage 5: Technical Review (for systems/algorithm papers)
        if paper_type in ['systems', 'algorithm', 'theoretical']:
            print("  Stage 5: Technical Review")
            technical_review = self._conduct_technical_review(
                paper_content, paper_type, model, request_timeout
            )
            reviews.append(technical_review)
        
        # Consolidate reviews
        print("  Consolidating reviews...")
        consolidated = self._consolidate_reviews(reviews, paper_type)
        
        print(f"✓ Multi-stage review completed with {len(reviews)} reviewers")
        return consolidated
    
    def _conduct_academic_review(self, paper_content: str, field: str, paper_type: str,
                               model: str, request_timeout: int) -> ReviewResult:
        """Conduct general academic review."""
        prompt = self._generate_academic_review_prompt(paper_content, field, paper_type)
        
        try:
            response = self.universal_chat(
                prompt, model=model, request_timeout=request_timeout,
                prompt_type="academic_review", fallback_models=self.fallback_models
            )
            return self._parse_review_response(response, "academic")
        except Exception as e:
            print(f"⚠ Academic review failed: {e}")
            return self._generate_fallback_review("academic", str(e))
    
    def _conduct_domain_expert_review(self, paper_content: str, field: str,
                                    model: str, request_timeout: int) -> ReviewResult:
        """Conduct domain-specific expert review."""
        prompt = self._generate_domain_expert_prompt(paper_content, field)
        
        try:
            response = self.universal_chat(
                prompt, model=model, request_timeout=request_timeout,
                prompt_type="domain_expert_review", fallback_models=self.fallback_models
            )
            return self._parse_review_response(response, "domain_expert")
        except Exception as e:
            print(f"⚠ Domain expert review failed: {e}")
            return self._generate_fallback_review("domain_expert", str(e))
    
    def _conduct_statistical_review(self, paper_content: str, model: str, request_timeout: int,
                                  sim_summary: str = "") -> ReviewResult:
        """Conduct statistical methodology review."""
        prompt = self._generate_statistical_review_prompt(paper_content, sim_summary)
        
        try:
            response = self.universal_chat(
                prompt, model=model, request_timeout=request_timeout,
                prompt_type="statistical_review", fallback_models=self.fallback_models
            )
            return self._parse_review_response(response, "statistical")
        except Exception as e:
            print(f"⚠ Statistical review failed: {e}")
            return self._generate_fallback_review("statistical", str(e))
    
    def _conduct_ethics_review(self, paper_content: str, model: str, request_timeout: int) -> ReviewResult:
        """Conduct ethics and responsible research review."""
        prompt = self._generate_ethics_review_prompt(paper_content)
        
        try:
            response = self.universal_chat(
                prompt, model=model, request_timeout=request_timeout,
                prompt_type="ethics_review", fallback_models=self.fallback_models
            )
            return self._parse_review_response(response, "ethics")
        except Exception as e:
            print(f"⚠ Ethics review failed: {e}")
            return self._generate_fallback_review("ethics", str(e))
    
    def _conduct_technical_review(self, paper_content: str, paper_type: str,
                                model: str, request_timeout: int) -> ReviewResult:
        """Conduct technical implementation review."""
        prompt = self._generate_technical_review_prompt(paper_content, paper_type)
        
        try:
            response = self.universal_chat(
                prompt, model=model, request_timeout=request_timeout,
                prompt_type="technical_review", fallback_models=self.fallback_models
            )
            return self._parse_review_response(response, "technical")
        except Exception as e:
            print(f"⚠ Technical review failed: {e}")
            return self._generate_fallback_review("technical", str(e))
    
    def _generate_academic_review_prompt(self, paper_content: str, field: str, paper_type: str) -> List[Dict[str, str]]:
        """Generate academic review prompt with rubric-based scoring and evidence."""
        system_prompt = f"""You are a senior academic reviewer for a top-tier {field} journal. 
        You are reviewing a {paper_type} paper. Conduct a thorough academic review focusing on:

    EVALUATION CRITERIA:
    1. Scientific rigor and methodology soundness
    2. Novelty and significance of contributions (identify precise novel claims; compare to 2–3 closest prior works; judge incremental vs. substantive novelty)
        3. Quality of writing and presentation
        4. Adequacy of literature review
        5. Appropriateness of conclusions
        6. Overall impact potential

        REVIEW STRUCTURE - Respond with exactly this format:
        OVERALL_RECOMMENDATION: [ACCEPT/MINOR_REVISION/MAJOR_REVISION/REJECT]
        CONFIDENCE_SCORE: [0.0-1.0]

        CRITERION_SCORES: [criterion1=score, criterion2=score, ...]  # e.g., novelty=0.8, rigor=0.7 (0.0-1.0 scale)
        EVIDENCE_JUSTIFICATION: [criterion1: justification; criterion2: justification; ...]

        STRENGTHS:
        - [List 3-5 key strengths]

        WEAKNESSES:
        - [List 3-5 key weaknesses]

        SPECIFIC_COMMENTS:
        - [List 5-7 specific technical comments]

        QUESTIONS_FOR_AUTHORS:
        - [List 2-4 questions that need clarification]

        DETAILED_REVIEW:
        [Comprehensive review paragraph of 200-300 words]

        Be rigorous but constructive. For each high or low criterion score, provide clear evidence or justification in the EVIDENCE_JUSTIFICATION section. Focus on both technical quality and presentation."""

        user_content = f"""Please review this {paper_type} paper in {field}:

        {paper_content[:8000]}{'...[CONTENT TRUNCATED]' if len(paper_content) > 8000 else ''}

        Provide a comprehensive academic review following the specified format, including rubric-based scores and evidence."""

        return [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_content}]
    
    def _generate_domain_expert_prompt(self, paper_content: str, field: str) -> List[Dict[str, str]]:
        """Generate domain expert review prompt with rubric-based scoring and evidence."""
        domain_expertise = self._get_domain_expertise_areas(field)

        system_prompt = f"""You are a leading expert in {field} with deep knowledge in {', '.join(domain_expertise)}.
        Review this paper from your specialized perspective, focusing on:

    DOMAIN-SPECIFIC CRITERIA:
        1. Technical accuracy within the field
        2. Appropriate use of domain terminology
        3. Understanding of field-specific challenges
        4. Relevance to current research directions
    5. Positioning relative to state-of-the-art (explicit novelty vs. 2–3 closest works)
        6. Potential impact on the research community

        Use the same response format as the academic reviewer, including:
        CRITERION_SCORES: [criterion1=score, ...]
        EVIDENCE_JUSTIFICATION: [criterion1: justification; ...]
        OVERALL_RECOMMENDATION, CONFIDENCE_SCORE, STRENGTHS, WEAKNESSES, 
        SPECIFIC_COMMENTS, QUESTIONS_FOR_AUTHORS, DETAILED_REVIEW

        For each high or low criterion score, provide clear evidence or justification in the EVIDENCE_JUSTIFICATION section. Focus particularly on domain-specific technical details, current trends, and how this work fits into the broader research landscape."""

        user_content = f"""As a {field} expert, please review this paper with particular attention to 
        domain-specific technical quality and positioning:

        {paper_content[:8000]}{'...[CONTENT TRUNCATED]' if len(paper_content) > 8000 else ''}

        Provide a comprehensive domain expert review following the specified format, including rubric-based scores and evidence."""

        return [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_content}]
    
    def _generate_statistical_review_prompt(self, paper_content: str, sim_summary: str) -> List[Dict[str, str]]:
        """Generate statistical methodology review prompt with rubric-based scoring and evidence."""
        system_prompt = """You are a statistical methodology expert reviewing the experimental design 
        and analysis of this research paper. Focus on:

        STATISTICAL CRITERIA:
        1. Appropriateness of experimental design
        2. Statistical power and sample size adequacy
        3. Correct application of statistical tests
        4. Proper handling of multiple comparisons
        5. Confidence intervals and effect size reporting
        6. Assumptions validation for chosen methods
        7. Reproducibility of statistical analysis

        Use the standard review format, including:
        CRITERION_SCORES: [criterion1=score, ...]
        EVIDENCE_JUSTIFICATION: [criterion1: justification; ...]
        Pay special attention to:
        - Statistical significance claims
        - P-hacking or data dredging concerns
        - Appropriate controls and baselines
        - Error analysis and uncertainty quantification
        - Generalizability of findings
        For each high or low criterion score, provide clear evidence or justification in the EVIDENCE_JUSTIFICATION section."""

        content = paper_content
        if sim_summary:
            content += f"\n\nSIMULATION SUMMARY:\n{sim_summary[:2000]}"

        user_content = f"""Please review the statistical methodology and analysis in this paper:

        {content[:10000]}{'...[CONTENT TRUNCATED]' if len(content) > 10000 else ''}

        Provide a comprehensive statistical review following the specified format, including rubric-based scores and evidence."""

        return [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_content}]
    
    def _generate_ethics_review_prompt(self, paper_content: str) -> List[Dict[str, str]]:
        """Generate ethics review prompt with rubric-based scoring and evidence."""
        system_prompt = """You are an ethics committee member reviewing this research for ethical considerations. 
        Evaluate:

        ETHICAL CRITERIA:
        1. Human subjects protection (if applicable)
        2. Data privacy and confidentiality
        3. Potential for misuse or harm
        4. Bias and fairness considerations
        5. Environmental impact (if applicable)
        6. Transparency and disclosure
        7. Responsible AI practices (if applicable)

        Use the standard review format, including:
        CRITERION_SCORES: [criterion1=score, ...]
        EVIDENCE_JUSTIFICATION: [criterion1: justification; ...]
        but focus on ethical implications. For each high or low criterion score, provide clear evidence or justification in the EVIDENCE_JUSTIFICATION section."""

        user_content = f"""Please review this paper for ethical considerations and responsible research practices:

        {paper_content[:8000]}{'...[CONTENT TRUNCATED]' if len(paper_content) > 8000 else ''}

        Provide a comprehensive ethics review following the specified format, including rubric-based scores and evidence."""

        return [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_content}]
    
    def _generate_technical_review_prompt(self, paper_content: str, paper_type: str) -> List[Dict[str, str]]:
        """Generate technical implementation review prompt with rubric-based scoring and evidence."""
        technical_focus = {
            'systems': 'system architecture, scalability, performance, and implementation quality',
            'algorithm': 'algorithmic correctness, complexity analysis, and optimization',
            'theoretical': 'mathematical rigor, proof techniques, and theoretical contributions'
        }

        focus_area = technical_focus.get(paper_type, 'technical implementation and validation')

        system_prompt = f"""You are a technical expert reviewing the {focus_area} aspects of this paper. 
        Focus on:

    TECHNICAL CRITERIA:
        1. Technical soundness and correctness
        2. Implementation quality and best practices
        3. Performance analysis and optimization
        4. Scalability and efficiency considerations
        5. Code quality and documentation (if applicable)
    6. Technical novelty and innovation (what is technically new vs. engineering of known ideas?)
        7. Practical applicability

        Use the standard review format, including:
        CRITERION_SCORES: [criterion1=score, ...]
        EVIDENCE_JUSTIFICATION: [criterion1: justification; ...]
        with emphasis on technical depth and implementation quality. For each high or low criterion score, provide clear evidence or justification in the EVIDENCE_JUSTIFICATION section."""

        user_content = f"""Please review the technical aspects of this {paper_type} paper:

        {paper_content[:8000]}{'...[CONTENT TRUNCATED]' if len(paper_content) > 8000 else ''}

        Provide a comprehensive technical review following the specified format, including rubric-based scores and evidence."""

        return [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_content}]
    
    def _parse_review_response(self, response: str, reviewer_type: str) -> ReviewResult:
        """Parse review response into structured format, including rubric-based scores and evidence."""
        try:
            # Extract structured sections
            overall_rec = self._extract_section(response, "OVERALL_RECOMMENDATION", "ACCEPT")
            confidence = float(self._extract_section(response, "CONFIDENCE_SCORE", "0.7"))

            strengths = self._extract_list_section(response, "STRENGTHS")
            weaknesses = self._extract_list_section(response, "WEAKNESSES")
            comments = self._extract_list_section(response, "SPECIFIC_COMMENTS")
            questions = self._extract_list_section(response, "QUESTIONS_FOR_AUTHORS")

            detailed_review = self._extract_section(response, "DETAILED_REVIEW", "Detailed review section not found.")

            # Rubric-based criterion scores and evidence
            criterion_scores = self._extract_criterion_scores(response)
            evidence_justification = self._extract_evidence_justification(response)

            return ReviewResult(
                reviewer_type=reviewer_type,
                overall_recommendation=overall_rec.strip(),
                confidence_score=max(0.0, min(1.0, confidence)),
                strengths=strengths,
                weaknesses=weaknesses,
                specific_comments=comments,
                questions_for_authors=questions,
                detailed_review=detailed_review,
                review_duration_estimate=self._estimate_review_duration(response),
                criterion_scores=criterion_scores,
                evidence_justification=evidence_justification
            )
        except Exception as e:
            print(f"⚠ Failed to parse {reviewer_type} review: {e}")
            return self._generate_fallback_review(reviewer_type, str(e))

    def _extract_criterion_scores(self, text: str) -> dict:
        """Extract rubric-based criterion scores from review text."""
        # Look for a section like: CRITERION_SCORES: novelty=0.8, rigor=0.7, ...
        pattern = r'CRITERION_SCORES:\s*(.*?)(?=\n[A-Z_]+:|$)'
        match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
        scores = {}
        if match:
            line = match.group(1).strip()
            # Accept both comma and newline separated
            for part in re.split(r'[\n,]', line):
                if '=' in part:
                    k, v = part.split('=', 1)
                    try:
                        scores[k.strip().lower()] = float(v.strip())
                    except Exception:
                        continue
        return scores if scores else None

    def _extract_evidence_justification(self, text: str) -> dict:
        """Extract evidence justifications for each criterion from review text."""
        # Look for a section like: EVIDENCE_JUSTIFICATION: novelty: ...; rigor: ...
        pattern = r'EVIDENCE_JUSTIFICATION:\s*(.*?)(?=\n[A-Z_]+:|$)'
        match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
        evidence = {}
        if match:
            block = match.group(1).strip()
            # Accept both semicolon and newline separated
            for part in re.split(r'[\n;]', block):
                if ':' in part:
                    k, v = part.split(':', 1)
                    evidence[k.strip().lower()] = v.strip()
        return evidence if evidence else None
    
    def _extract_section(self, text: str, section_name: str, default: str = "") -> str:
        """Extract a specific section from review text."""
        pattern = rf'{section_name}:\s*(.*?)(?=\n[A-Z_]+:|$)'
        match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
        return match.group(1).strip() if match else default
    
    def _extract_list_section(self, text: str, section_name: str) -> List[str]:
        """Extract a list section from review text."""
        pattern = rf'{section_name}:\s*(.*?)(?=\n[A-Z_]+:|$)'
        match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
        
        if not match:
            return []
        
        section_content = match.group(1).strip()
        # Extract list items (lines starting with -, •, or numbers)
        items = re.findall(r'^[-•*]\s*(.+)$', section_content, re.MULTILINE)
        if not items:
            items = re.findall(r'^\d+\.\s*(.+)$', section_content, re.MULTILINE)
        
        return [item.strip() for item in items if item.strip()]
    
    def _consolidate_reviews(self, reviews: List[ReviewResult], paper_type: str) -> ConsolidatedReview:
        """Consolidate multiple reviews into a unified decision, with penalty for repeated issues."""
        if not reviews:
            return ConsolidatedReview(
                overall_decision="INSUFFICIENT_REVIEWS",
                consensus_level=0.0,
                individual_reviews=[],
                consensus_strengths=[],
                consensus_weaknesses=[],
                priority_issues=[],
                revision_roadmap=[],
                meta_analysis="No reviews available."
            )

        # Calculate consensus
        recommendations = [r.overall_recommendation for r in reviews]
        recommendation_weights = {
            'ACCEPT': 4,
            'MINOR_REVISION': 3,
            'MAJOR_REVISION': 2,
            'REJECT': 1
        }


        # Weighted decision based on confidence scores, with fairness adjustment
        weighted_scores = []
        for review in reviews:
            weight = recommendation_weights.get(review.overall_recommendation, 2)
            # Outlier/optimism/evidence adjustment
            evidence_count = len(review.evidence_justification) if review.evidence_justification else 0
            criterion_count = len(review.criterion_scores) if review.criterion_scores else 0
            # If high scores but little evidence, reduce weight
            if criterion_count > 0 and evidence_count < criterion_count // 2:
                adj_conf = review.confidence_score * 0.7
            else:
                adj_conf = review.confidence_score
            # Outlier: if this review's score is much higher/lower than others, reduce its impact
            # (simple: if >1.5 std dev from mean, halve its weight)
            weighted_scores.append(weight * adj_conf)

        # Outlier adjustment (after initial pass)
        import statistics
        if len(weighted_scores) > 1:
            mean_score = statistics.mean(weighted_scores)
            stdev_score = statistics.stdev(weighted_scores)
            for i, ws in enumerate(weighted_scores):
                if abs(ws - mean_score) > 1.5 * stdev_score:
                    weighted_scores[i] = ws * 0.5

        avg_weighted_score = sum(weighted_scores) / len(weighted_scores)

        # Penalty for repeated issues (if a weakness appears in 2+ reviews, reduce score)
        all_weaknesses = []
        for review in reviews:
            all_weaknesses.extend(review.weaknesses)
        repeated_issues = self._find_consensus_items(all_weaknesses, min_mentions=2)
        penalty = 0.0
        if repeated_issues:
            penalty = 0.3 * len(repeated_issues) / len(reviews)  # scale penalty by number of repeated issues
            avg_weighted_score = max(0.0, avg_weighted_score - penalty)

        # Map back to decision
        if avg_weighted_score >= 3.5:
            overall_decision = "ACCEPT"
        elif avg_weighted_score >= 2.5:
            overall_decision = "MINOR_REVISION"
        elif avg_weighted_score >= 1.5:
            overall_decision = "MAJOR_REVISION"
        else:
            overall_decision = "REJECT"

        # Calculate consensus level
        most_common_rec = max(set(recommendations), key=recommendations.count)
        consensus_level = recommendations.count(most_common_rec) / len(reviews)

        # Consolidate strengths and weaknesses
        all_strengths = []
        for review in reviews:
            all_strengths.extend(review.strengths)

        # Find consensus items (mentioned by multiple reviewers)
        consensus_strengths = self._find_consensus_items(all_strengths)
        consensus_weaknesses = repeated_issues

        # Generate priority issues and revision roadmap
        priority_issues = self._identify_priority_issues(reviews)
        revision_roadmap = self._generate_revision_roadmap(reviews, overall_decision)

        # Meta-analysis
        meta_analysis = self._generate_meta_analysis(reviews, consensus_level, overall_decision)

        return ConsolidatedReview(
            overall_decision=overall_decision,
            consensus_level=consensus_level,
            individual_reviews=reviews,
            consensus_strengths=consensus_strengths,
            consensus_weaknesses=consensus_weaknesses,
            priority_issues=priority_issues,
            revision_roadmap=revision_roadmap,
            meta_analysis=meta_analysis
        )
    
    def _find_consensus_items(self, items: List[str], min_mentions: int = 2) -> List[str]:
        """Find items mentioned by multiple reviewers."""
        # Simple similarity-based grouping
        consensus_items = []
        used_indices = set()
        
        for i, item in enumerate(items):
            if i in used_indices:
                continue
            
            similar_items = [item]
            for j, other_item in enumerate(items[i+1:], i+1):
                if j in used_indices:
                    continue
                
                # Simple similarity check (can be enhanced with embeddings)
                if self._items_similar(item, other_item):
                    similar_items.append(other_item)
                    used_indices.add(j)
            
            if len(similar_items) >= min_mentions:
                consensus_items.append(self._merge_similar_items(similar_items))
            
            used_indices.add(i)
        
        return consensus_items
    
    def _items_similar(self, item1: str, item2: str, threshold: float = 0.3) -> bool:
        """Check if two review items are similar."""
        # Simple word overlap similarity
        words1 = set(item1.lower().split())
        words2 = set(item2.lower().split())
        
        if not words1 or not words2:
            return False
        
        overlap = len(words1 & words2)
        union = len(words1 | words2)
        
        return overlap / union >= threshold
    
    def _merge_similar_items(self, items: List[str]) -> str:
        """Merge similar items into a single consensus item."""
        if len(items) == 1:
            return items[0]
        
        # Use the longest item as base, or combine key points
        return max(items, key=len)
    
    def _identify_priority_issues(self, reviews: List[ReviewResult]) -> List[str]:
        """Identify priority issues that need immediate attention."""
        priority_issues = []
        
        # Issues mentioned by multiple reviewers
        all_issues = []
        for review in reviews:
            all_issues.extend(review.weaknesses)
            all_issues.extend([c for c in review.specific_comments if "issue" in c.lower() or "problem" in c.lower()])
        
        consensus_issues = self._find_consensus_items(all_issues, min_mentions=2)
        priority_issues.extend(consensus_issues)
        
        # Issues from high-confidence reviewers
        for review in reviews:
            if review.confidence_score >= 0.8:
                priority_issues.extend(review.weaknesses[:2])  # Top 2 issues from high-confidence reviewers
        
        return list(set(priority_issues))  # Remove duplicates
    
    def _generate_revision_roadmap(self, reviews: List[ReviewResult], overall_decision: str) -> List[str]:
        """Generate a revision roadmap based on reviews."""
        roadmap = []
        
        if overall_decision == "REJECT":
            roadmap.append("Consider fundamental restructuring of the research approach")
            roadmap.append("Address core methodological concerns before resubmission")
        
        elif overall_decision == "MAJOR_REVISION":
            roadmap.append("Address all major methodological and technical issues")
            roadmap.append("Strengthen experimental validation and analysis")
            roadmap.append("Improve writing clarity and organization")
        
        elif overall_decision == "MINOR_REVISION":
            roadmap.append("Address specific reviewer comments and suggestions")
            roadmap.append("Clarify technical details and improve presentation")
        
        # Add specific items based on reviewer feedback
        all_questions = []
        for review in reviews:
            all_questions.extend(review.questions_for_authors)
        
        if all_questions:
            roadmap.append("Answer all reviewer questions in revision letter")
        
        return roadmap
    
    def _generate_meta_analysis(self, reviews: List[ReviewResult], consensus_level: float, 
                               overall_decision: str) -> str:
        """Generate meta-analysis of the review process, including venue fit and confidence."""
        analysis = []

        analysis.append(f"Review Process Summary ({len(reviews)} reviewers):")
        analysis.append(f"- Overall Decision: {overall_decision}")
        analysis.append(f"- Consensus Level: {consensus_level:.2f}")

        # Reviewer confidence analysis
        confidences = [r.confidence_score for r in reviews]
        avg_confidence = sum(confidences) / len(confidences)
        analysis.append(f"- Average Reviewer Confidence: {avg_confidence:.2f}")

        # Venue fit (if available in evidence_justification)
        venue_fit_scores = []
        for r in reviews:
            if r.criterion_scores and 'venue_fit' in r.criterion_scores:
                venue_fit_scores.append(r.criterion_scores['venue_fit'])
        if venue_fit_scores:
            avg_venue_fit = sum(venue_fit_scores) / len(venue_fit_scores)
            analysis.append(f"- Average Venue Fit Score: {avg_venue_fit:.2f}")

        # Review type coverage
        reviewer_types = [r.reviewer_type for r in reviews]
        analysis.append(f"- Review Types: {', '.join(reviewer_types)}")

        # Decision distribution
        decisions = [r.overall_recommendation for r in reviews]
        decision_counts = {dec: decisions.count(dec) for dec in set(decisions)}
        analysis.append(f"- Decision Distribution: {decision_counts}")

        # Key insights
        if consensus_level >= 0.8:
            analysis.append("- Strong consensus among reviewers")
        elif consensus_level >= 0.6:
            analysis.append("- Moderate consensus with some disagreement")
        else:
            analysis.append("- Significant disagreement among reviewers")

        if avg_confidence >= 0.8:
            analysis.append("- High reviewer confidence in assessments")
        elif avg_confidence >= 0.6:
            analysis.append("- Moderate reviewer confidence")
        else:
            analysis.append("- Low reviewer confidence - may need additional expert input")

        if venue_fit_scores:
            if avg_venue_fit >= 0.8:
                analysis.append("- Paper is a strong fit for the intended venue")
            elif avg_venue_fit >= 0.6:
                analysis.append("- Paper is a reasonable fit for the venue")
            else:
                analysis.append("- Paper may not be a good fit for the venue")

        return "\n".join(analysis)
    
    def _get_domain_expertise_areas(self, field: str) -> List[str]:
        """Get domain expertise areas for different fields."""
        expertise_map = {
            'Computer Science': ['algorithms', 'systems', 'machine learning', 'security', 'HCI'],
            'Physics': ['quantum mechanics', 'condensed matter', 'particle physics', 'astrophysics'],
            'Biology': ['molecular biology', 'genetics', 'ecology', 'bioinformatics'],
            'Chemistry': ['organic chemistry', 'physical chemistry', 'biochemistry', 'materials'],
            'Mathematics': ['analysis', 'algebra', 'topology', 'probability', 'optimization'],
            'Engineering': ['electrical', 'mechanical', 'software', 'systems', 'control theory']
        }
        
        return expertise_map.get(field, ['technical analysis', 'research methodology'])
    
    def _is_experimental_paper(self, paper_content: str) -> bool:
        """Determine if paper requires statistical review."""
        experimental_indicators = [
            'experiment', 'statistical', 'p-value', 'confidence interval',
            'sample size', 'hypothesis test', 'anova', 'regression',
            'correlation', 'significance', 'control group'
        ]
        
        content_lower = paper_content.lower()
        return sum(content_lower.count(indicator) for indicator in experimental_indicators) >= 3
    
    def _requires_ethics_review(self, paper_content: str) -> bool:
        """Determine if paper requires ethics review."""
        ethics_indicators = [
            'human subjects', 'participant', 'survey', 'interview',
            'personal data', 'privacy', 'sensitive', 'vulnerable',
            'consent', 'irb', 'ethics', 'bias', 'fairness', 'discrimination'
        ]
        
        content_lower = paper_content.lower()
        return any(indicator in content_lower for indicator in ethics_indicators)
    
    def _generate_fallback_review(self, reviewer_type: str, error_msg: str) -> ReviewResult:
        """Generate fallback review when AI review fails."""
        return ReviewResult(
            reviewer_type=reviewer_type,
            overall_recommendation="MANUAL_REVIEW_REQUIRED",
            confidence_score=0.0,
            strengths=["Automated review system unavailable"],
            weaknesses=[f"Review system error: {error_msg}"],
            specific_comments=["Manual review required due to system failure"],
            questions_for_authors=["Please ensure all technical details are clearly presented"],
            detailed_review=f"Automated {reviewer_type} review failed. Manual review recommended.",
            review_duration_estimate="Manual review required"
        )
    
    def _estimate_review_duration(self, review_text: str) -> str:
        """Estimate how long the review process took."""
        word_count = len(review_text.split())
        
        if word_count < 200:
            return "10-15 minutes"
        elif word_count < 500:
            return "15-25 minutes"
        elif word_count < 1000:
            return "25-40 minutes"
        else:
            return "40+ minutes"
    
    def generate_consolidated_report(self, consolidated_review: ConsolidatedReview) -> str:
        """Generate comprehensive review report."""
        report = []
        report.append("=" * 80)
        report.append("MULTI-STAGE PEER REVIEW REPORT")
        report.append("=" * 80)
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"Reviewers: {len(consolidated_review.individual_reviews)}")
        report.append("")
        
        # Overall decision
        report.append("EDITORIAL DECISION:")
        report.append(f"  Recommendation: {consolidated_review.overall_decision}")
        report.append(f"  Consensus Level: {consolidated_review.consensus_level:.2%}")
        report.append("")
        
        # Individual reviewer summaries
        report.append("REVIEWER SUMMARIES:")
        for i, review in enumerate(consolidated_review.individual_reviews, 1):
            report.append(f"  Reviewer {i} ({review.reviewer_type}):")
            report.append(f"    Recommendation: {review.overall_recommendation}")
            report.append(f"    Confidence: {review.confidence_score:.2f}")
            report.append(f"    Duration: {review.review_duration_estimate}")
        report.append("")
        
        # Consensus strengths and weaknesses
        if consolidated_review.consensus_strengths:
            report.append("CONSENSUS STRENGTHS:")
            for strength in consolidated_review.consensus_strengths:
                report.append(f"  • {strength}")
            report.append("")
        
        if consolidated_review.consensus_weaknesses:
            report.append("CONSENSUS WEAKNESSES:")
            for weakness in consolidated_review.consensus_weaknesses:
                report.append(f"  • {weakness}")
            report.append("")
        
        # Priority issues
        if consolidated_review.priority_issues:
            report.append("PRIORITY ISSUES TO ADDRESS:")
            for issue in consolidated_review.priority_issues:
                report.append(f"  1. {issue}")
            report.append("")
        
        # Revision roadmap
        if consolidated_review.revision_roadmap:
            report.append("REVISION ROADMAP:")
            for step in consolidated_review.revision_roadmap:
                report.append(f"  → {step}")
            report.append("")
        
        # Meta-analysis
        report.append("META-ANALYSIS:")
        for line in consolidated_review.meta_analysis.split('\n'):
            report.append(f"  {line}")
        report.append("")
        
        # Detailed reviews
        report.append("DETAILED REVIEWS:")
        report.append("-" * 40)
        
        for i, review in enumerate(consolidated_review.individual_reviews, 1):
            report.append(f"REVIEWER {i} ({review.reviewer_type.upper()}):")
            report.append(f"Recommendation: {review.overall_recommendation}")
            report.append(f"Confidence: {review.confidence_score:.2f}")
            report.append("")
            
            if review.strengths:
                report.append("Strengths:")
                for strength in review.strengths:
                    report.append(f"  + {strength}")
                report.append("")
            
            if review.weaknesses:
                report.append("Weaknesses:")
                for weakness in review.weaknesses:
                    report.append(f"  - {weakness}")
                report.append("")
            
            if review.specific_comments:
                report.append("Specific Comments:")
                for comment in review.specific_comments:
                    report.append(f"  • {comment}")
                report.append("")
            
            if review.questions_for_authors:
                report.append("Questions for Authors:")
                for question in review.questions_for_authors:
                    report.append(f"  ? {question}")
                report.append("")
            
            report.append("Detailed Review:")
            for line in review.detailed_review.split('\n'):
                if line.strip():
                    report.append(f"  {line}")
            
            report.append("")
            report.append("-" * 40)
        
        report.append("=" * 80)
        
        return "\n".join(report)
