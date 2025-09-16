"""
Advanced Literature Integration Module
=====================================

Comprehensive literature analysis including gap detection,
citation relevance scoring, and competing work identification.
"""

import re
import json
import urllib.request
import urllib.parse
from typing import Dict, List, Optional, Any, Tuple, Set
from pathlib import Path
from dataclasses import dataclass
from datetime import datetime, timedelta
import time
import hashlib

try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False


@dataclass
class LiteraturePaper:
    """Represents a paper from literature search."""
    title: str
    authors: List[str]
    year: int
    venue: str
    abstract: str = ""
    citations: int = 0
    doi: str = ""
    url: str = ""
    relevance_score: float = 0.0
    paper_type: str = ""  # competing, supporting, foundational


@dataclass
class LiteratureGap:
    """Represents an identified research gap."""
    description: str
    supporting_evidence: List[str]
    potential_impact: str
    related_papers: List[LiteraturePaper]
    confidence_score: float


@dataclass
class CitationAnalysis:
    """Analysis of citations in the paper."""
    total_citations: int
    relevant_citations: int
    irrelevant_citations: List[str]
    missing_key_citations: List[LiteraturePaper]
    citation_density: float
    recency_score: float
    authority_score: float


@dataclass
class LiteratureAnalysisResult:
    """Complete literature analysis result."""
    citation_analysis: CitationAnalysis
    identified_gaps: List[LiteratureGap]
    competing_work: List[LiteraturePaper]
    supporting_work: List[LiteraturePaper]
    foundational_work: List[LiteraturePaper]
    recommendations: List[str]
    analysis_summary: str


class LiteratureIntegrator:
    """Advanced literature integration and analysis system."""
    
    def __init__(self, universal_chat_fn, api_keys: Optional[Dict[str, str]] = None):
        self.universal_chat = universal_chat_fn
        self.api_keys = api_keys or {}
        
        # API endpoints
        self.semantic_scholar_base = "https://api.semanticscholar.org/graph/v1"
        self.crossref_base = "https://api.crossref.org/works"
        
        # Rate limiting
        self.last_api_call = {}
        self.api_delays = {
            'semantic_scholar': 1.0,  # 1 second between calls
            'crossref': 0.1,         # 0.1 second between calls
        }
        
        # Caching
        self.cache_dir = Path.home() / ".sciresearch_cache" / "literature"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache_ttl = timedelta(days=7)  # Cache for 7 days
    
    def analyze_literature(self, paper_content: str, field: str, 
                         model: str, request_timeout: int = 1800) -> LiteratureAnalysisResult:
        """Comprehensive literature analysis."""
        print("📚 Starting Advanced Literature Analysis...")
        
        # Extract research claims and contributions
        research_claims = self._extract_research_claims(paper_content)
        print(f"  Extracted {len(research_claims)} research claims")
        
        # Analyze existing citations
        citation_analysis = self._analyze_citations(paper_content, model, request_timeout)
        print(f"  Analyzed {citation_analysis.total_citations} existing citations")
        
        # Search for relevant literature
        supporting_work = []
        competing_work = []
        foundational_work = []
        
        for claim in research_claims:
            print(f"  Searching literature for: {claim[:50]}...")
            
            # Search supporting literature
            supporting = self._search_supporting_literature(claim, field)
            supporting_work.extend(supporting)
            
            # Search competing work
            competing = self._search_competing_literature(claim, field)
            competing_work.extend(competing)
        
        # Find foundational work
        foundational_work = self._identify_foundational_work(paper_content, field)
        
        # Identify literature gaps
        identified_gaps = self._identify_literature_gaps(
            research_claims, supporting_work, competing_work, model, request_timeout
        )
        print(f"  Identified {len(identified_gaps)} potential research gaps")
        
        # Generate recommendations
        recommendations = self._generate_literature_recommendations(
            citation_analysis, supporting_work, competing_work, identified_gaps
        )
        
        # Generate analysis summary
        analysis_summary = self._generate_analysis_summary(
            citation_analysis, identified_gaps, len(supporting_work), 
            len(competing_work), len(foundational_work)
        )
        
        return LiteratureAnalysisResult(
            citation_analysis=citation_analysis,
            identified_gaps=identified_gaps,
            competing_work=self._deduplicate_papers(competing_work),
            supporting_work=self._deduplicate_papers(supporting_work),
            foundational_work=self._deduplicate_papers(foundational_work),
            recommendations=recommendations,
            analysis_summary=analysis_summary
        )
    
    def _extract_research_claims(self, paper_content: str) -> List[str]:
        """Extract key research claims and contributions."""
        claims = []
        
        # Look for contribution statements
        contribution_patterns = [
            r'we propose ([^.]+)',
            r'we introduce ([^.]+)',
            r'we present ([^.]+)',
            r'we develop ([^.]+)',
            r'we demonstrate ([^.]+)',
            r'our contribution[s]? ([^.]+)',
            r'our approach ([^.]+)',
            r'our method ([^.]+)',
        ]
        
        for pattern in contribution_patterns:
            matches = re.findall(pattern, paper_content, re.IGNORECASE)
            claims.extend([match.strip() for match in matches])
        
        # Look for novelty claims
        novelty_patterns = [
            r'for the first time ([^.]+)',
            r'novel ([^.]+)',
            r'new ([^.]+)',
            r'unprecedented ([^.]+)',
        ]
        
        for pattern in novelty_patterns:
            matches = re.findall(pattern, paper_content, re.IGNORECASE)
            claims.extend([match.strip() for match in matches])
        
        # Extract from abstract and conclusion
        abstract_match = re.search(r'\\begin\{abstract\}(.*?)\\end\{abstract\}', 
                                 paper_content, re.DOTALL)
        if abstract_match:
            abstract_text = abstract_match.group(1)
            # Extract key sentences
            sentences = re.split(r'[.!?]+', abstract_text)
            for sentence in sentences:
                if any(word in sentence.lower() for word in ['propose', 'introduce', 'present', 'develop']):
                    claims.append(sentence.strip())
        
        return list(set(claims))  # Remove duplicates
    
    def _analyze_citations(self, paper_content: str, model: str, request_timeout: int) -> CitationAnalysis:
        """Analyze existing citations in the paper."""
        # Extract citations
        citations = re.findall(r'\\cite\{([^}]+)\}', paper_content)
        citation_keys = []
        for citation in citations:
            citation_keys.extend([key.strip() for key in citation.split(',')])
        
        total_citations = len(set(citation_keys))
        
        # Extract bibliography entries
        bib_entries = self._extract_bibliography_entries(paper_content)
        
        # Analyze citation relevance using AI
        relevant_citations = 0
        irrelevant_citations = []
        
        if bib_entries and total_citations > 0:
            relevance_analysis = self._analyze_citation_relevance(
                paper_content, bib_entries, model, request_timeout
            )
            relevant_citations = relevance_analysis.get('relevant_count', total_citations)
            irrelevant_citations = relevance_analysis.get('irrelevant', [])
        else:
            relevant_citations = total_citations  # Assume all are relevant if can't analyze
        
        # Calculate metrics
        word_count = len(paper_content.split())
        citation_density = total_citations / max(word_count, 1) * 1000  # Citations per 1000 words
        
        # Calculate recency score (based on years)
        years = self._extract_publication_years(bib_entries)
        current_year = datetime.now().year
        recency_score = self._calculate_recency_score(years, current_year)
        
        # Authority score (simplified - based on venue types)
        authority_score = self._calculate_authority_score(bib_entries)
        
        # Find missing key citations (would require API calls)
        missing_key_citations = []  # Placeholder - would need actual search
        
        return CitationAnalysis(
            total_citations=total_citations,
            relevant_citations=relevant_citations,
            irrelevant_citations=irrelevant_citations,
            missing_key_citations=missing_key_citations,
            citation_density=citation_density,
            recency_score=recency_score,
            authority_score=authority_score
        )
    
    def _analyze_citation_relevance(self, paper_content: str, bib_entries: List[Dict], 
                                  model: str, request_timeout: int) -> Dict[str, Any]:
        """Use AI to analyze citation relevance."""
        # Prepare sample of citations for analysis
        sample_size = min(10, len(bib_entries))
        sample_entries = bib_entries[:sample_size]
        
        prompt = [
            {
                "role": "system",
                "content": """You are analyzing the relevance of citations in a research paper.
                For each citation, determine if it's directly relevant to the paper's research topic
                and contributions. Respond with JSON format:
                
                {
                    "relevant_count": <number>,
                    "irrelevant": [<list of citation keys that seem irrelevant>],
                    "analysis": "<brief analysis>"
                }"""
            },
            {
                "role": "user",
                "content": f"""Analyze the relevance of these citations to this paper:

                PAPER EXCERPT (first 1000 words):
                {' '.join(paper_content.split()[:1000])}
                
                CITATIONS TO ANALYZE:
                {json.dumps(sample_entries, indent=2)}
                
                Provide relevance analysis in JSON format."""
            }
        ]
        
        try:
            response = self.universal_chat(
                prompt, model=model, request_timeout=request_timeout,
                prompt_type="citation_relevance"
            )
            return json.loads(response.strip())
        except Exception as e:
            print(f"⚠ Citation relevance analysis failed: {e}")
            return {'relevant_count': len(bib_entries), 'irrelevant': []}
    
    def _extract_bibliography_entries(self, paper_content: str) -> List[Dict]:
        """Extract bibliography entries from paper."""
        entries = []
        
        # Extract from filecontents BibTeX
        bib_pattern = r'\\begin\{filecontents\*?\}\{[^}]*\.bib\}(.*?)\\end\{filecontents\*?\}'
        bib_match = re.search(bib_pattern, paper_content, re.DOTALL)
        
        if bib_match:
            bib_content = bib_match.group(1)
            # Parse BibTeX entries
            entry_pattern = r'@(\w+)\{([^,]+),\s*(.*?)\n\}'
            matches = re.findall(entry_pattern, bib_content, re.DOTALL)
            
            for entry_type, key, fields in matches:
                entry = {'type': entry_type, 'key': key}
                
                # Extract fields
                field_pattern = r'(\w+)\s*=\s*\{([^}]+)\}'
                field_matches = re.findall(field_pattern, fields)
                for field_name, field_value in field_matches:
                    entry[field_name] = field_value
                
                entries.append(entry)
        
        return entries
    
    def _extract_publication_years(self, bib_entries: List[Dict]) -> List[int]:
        """Extract publication years from bibliography entries."""
        years = []
        for entry in bib_entries:
            if 'year' in entry:
                try:
                    year = int(entry['year'])
                    if 1900 <= year <= 2030:  # Reasonable year range
                        years.append(year)
                except ValueError:
                    continue
        return years
    
    def _calculate_recency_score(self, years: List[int], current_year: int) -> float:
        """Calculate recency score based on publication years."""
        if not years:
            return 0.5  # Neutral score if no years
        
        # Weight recent papers more heavily
        weights = []
        for year in years:
            age = current_year - year
            if age <= 2:
                weight = 1.0
            elif age <= 5:
                weight = 0.8
            elif age <= 10:
                weight = 0.6
            else:
                weight = 0.3
            weights.append(weight)
        
        return sum(weights) / len(weights)
    
    def _calculate_authority_score(self, bib_entries: List[Dict]) -> float:
        """Calculate authority score based on publication venues."""
        if not bib_entries:
            return 0.5
        
        high_impact_venues = [
            'nature', 'science', 'cell', 'nejm', 'lancet',
            'neurips', 'icml', 'iclr', 'aaai', 'ijcai',
            'acm', 'ieee', 'springer', 'elsevier'
        ]
        
        authority_scores = []
        for entry in bib_entries:
            journal = entry.get('journal', '').lower()
            booktitle = entry.get('booktitle', '').lower()
            venue = journal or booktitle
            
            if any(hiv in venue for hiv in high_impact_venues):
                authority_scores.append(1.0)
            elif 'conference' in venue or 'workshop' in venue:
                authority_scores.append(0.7)
            elif 'arxiv' in venue or 'preprint' in venue:
                authority_scores.append(0.4)
            else:
                authority_scores.append(0.6)  # Default score
        
        return sum(authority_scores) / len(authority_scores) if authority_scores else 0.5
    
    def _search_supporting_literature(self, claim: str, field: str) -> List[LiteraturePaper]:
        """Search for literature that supports the research claim."""
        # This would use APIs like Semantic Scholar
        # For now, return placeholder results
        return []
    
    def _search_competing_literature(self, claim: str, field: str) -> List[LiteraturePaper]:
        """Search for competing or contradictory literature."""
        # This would use APIs to find competing approaches
        # For now, return placeholder results
        return []
    
    def _identify_foundational_work(self, paper_content: str, field: str) -> List[LiteraturePaper]:
        """Identify foundational work that should be cited."""
        # This would identify seminal papers in the field
        # For now, return placeholder results
        return []
    
    def _search_semantic_scholar(self, query: str, limit: int = 20) -> List[Dict]:
        """Search Semantic Scholar API."""
        if not self._check_rate_limit('semantic_scholar'):
            return []
        
        cache_key = hashlib.md5(f"ss_{query}_{limit}".encode()).hexdigest()
        cached_result = self._get_cached_result(cache_key)
        if cached_result:
            return cached_result
        
        try:
            url = f"{self.semantic_scholar_base}/paper/search"
            params = {
                'query': query,
                'limit': limit,
                'fields': 'title,authors,year,venue,abstract,citationCount,externalIds'
            }
            
            if HAS_REQUESTS:
                response = requests.get(url, params=params, timeout=10)
                response.raise_for_status()
                data = response.json()
            else:
                # Fallback using urllib
                query_string = urllib.parse.urlencode(params)
                request_url = f"{url}?{query_string}"
                with urllib.request.urlopen(request_url, timeout=10) as response:
                    data = json.loads(response.read().decode())
            
            papers = data.get('data', [])
            self._cache_result(cache_key, papers)
            return papers
            
        except Exception as e:
            print(f"⚠ Semantic Scholar search failed: {e}")
            return []
    
    def _check_rate_limit(self, api_name: str) -> bool:
        """Check if we can make an API call based on rate limits."""
        if api_name not in self.last_api_call:
            self.last_api_call[api_name] = 0
        
        time_since_last = time.time() - self.last_api_call[api_name]
        required_delay = self.api_delays.get(api_name, 1.0)
        
        if time_since_last < required_delay:
            time.sleep(required_delay - time_since_last)
        
        self.last_api_call[api_name] = time.time()
        return True
    
    def _get_cached_result(self, cache_key: str) -> Optional[Any]:
        """Get cached result if available and not expired."""
        cache_file = self.cache_dir / f"{cache_key}.json"
        
        if not cache_file.exists():
            return None
        
        try:
            # Check if cache is expired
            file_age = datetime.now() - datetime.fromtimestamp(cache_file.stat().st_mtime)
            if file_age > self.cache_ttl:
                cache_file.unlink()  # Remove expired cache
                return None
            
            with open(cache_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception:
            return None
    
    def _cache_result(self, cache_key: str, result: Any):
        """Cache API result."""
        cache_file = self.cache_dir / f"{cache_key}.json"
        
        try:
            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=2)
        except Exception as e:
            print(f"⚠ Failed to cache result: {e}")
    
    def _identify_literature_gaps(self, research_claims: List[str], 
                                supporting_work: List[LiteraturePaper],
                                competing_work: List[LiteraturePaper],
                                model: str, request_timeout: int) -> List[LiteratureGap]:
        """Identify potential research gaps using AI analysis."""
        if not research_claims:
            return []
        
        prompt = [
            {
                "role": "system",
                "content": """You are a research expert identifying literature gaps.
                Analyze research claims and existing work to identify potential gaps
                where this research could make novel contributions.
                
                Respond with JSON format:
                {
                    "gaps": [
                        {
                            "description": "<gap description>",
                            "supporting_evidence": ["<evidence>", "..."],
                            "potential_impact": "<impact assessment>",
                            "confidence_score": <0.0-1.0>
                        }
                    ]
                }"""
            },
            {
                "role": "user",
                "content": f"""Identify research gaps based on these claims and existing work:

                RESEARCH CLAIMS:
                {json.dumps(research_claims, indent=2)}
                
                SUPPORTING WORK COUNT: {len(supporting_work)}
                COMPETING WORK COUNT: {len(competing_work)}
                
                Identify 2-5 potential research gaps where this work could contribute."""
            }
        ]
        
        try:
            response = self.universal_chat(
                prompt, model=model, request_timeout=request_timeout,
                prompt_type="literature_gap_analysis"
            )
            
            data = json.loads(response.strip())
            gaps = []
            
            for gap_data in data.get('gaps', []):
                gap = LiteratureGap(
                    description=gap_data.get('description', ''),
                    supporting_evidence=gap_data.get('supporting_evidence', []),
                    potential_impact=gap_data.get('potential_impact', ''),
                    related_papers=[],  # Would be populated from search results
                    confidence_score=gap_data.get('confidence_score', 0.5)
                )
                gaps.append(gap)
            
            return gaps
            
        except Exception as e:
            print(f"⚠ Literature gap analysis failed: {e}")
            return []
    
    def _generate_literature_recommendations(self, citation_analysis: CitationAnalysis,
                                           supporting_work: List[LiteraturePaper],
                                           competing_work: List[LiteraturePaper],
                                           identified_gaps: List[LiteratureGap]) -> List[str]:
        """Generate literature-related recommendations."""
        recommendations = []
        
        # Citation density recommendations
        if citation_analysis.citation_density < 20:  # Less than 20 citations per 1000 words
            recommendations.append("Consider adding more citations to support claims (current density is low)")
        elif citation_analysis.citation_density > 100:
            recommendations.append("Consider reducing citation density - some citations may be unnecessary")
        
        # Recency recommendations
        if citation_analysis.recency_score < 0.6:
            recommendations.append("Include more recent publications (within last 5 years) to strengthen literature review")
        
        # Authority recommendations
        if citation_analysis.authority_score < 0.6:
            recommendations.append("Include more citations from high-impact venues and authoritative sources")
        
        # Irrelevant citations
        if citation_analysis.irrelevant_citations:
            recommendations.append(f"Review {len(citation_analysis.irrelevant_citations)} potentially irrelevant citations")
        
        # Missing work recommendations
        if len(competing_work) < 3:
            recommendations.append("Search for and cite more competing approaches to better position your work")
        
        if len(supporting_work) < 5:
            recommendations.append("Add more supporting citations to strengthen theoretical foundation")
        
        # Gap-based recommendations
        high_confidence_gaps = [gap for gap in identified_gaps if gap.confidence_score > 0.7]
        if high_confidence_gaps:
            recommendations.append("Emphasize identified research gaps to strengthen novelty claims")
        
        return recommendations if recommendations else ["Literature integration appears comprehensive"]
    
    def _generate_analysis_summary(self, citation_analysis: CitationAnalysis,
                                 identified_gaps: List[LiteratureGap],
                                 supporting_count: int, competing_count: int,
                                 foundational_count: int) -> str:
        """Generate summary of literature analysis."""
        summary = []
        
        summary.append(f"Literature Analysis Summary:")
        summary.append(f"- Citations: {citation_analysis.total_citations} total, {citation_analysis.relevant_citations} relevant")
        summary.append(f"- Citation density: {citation_analysis.citation_density:.1f} per 1000 words")
        summary.append(f"- Recency score: {citation_analysis.recency_score:.2f}")
        summary.append(f"- Authority score: {citation_analysis.authority_score:.2f}")
        summary.append(f"- Supporting work found: {supporting_count} papers")
        summary.append(f"- Competing work found: {competing_count} papers")
        summary.append(f"- Foundational work identified: {foundational_count} papers")
        summary.append(f"- Research gaps identified: {len(identified_gaps)}")
        
        # Quality assessment
        if (citation_analysis.recency_score >= 0.7 and 
            citation_analysis.authority_score >= 0.7 and
            citation_analysis.citation_density >= 15):
            summary.append("- Overall literature quality: EXCELLENT")
        elif (citation_analysis.recency_score >= 0.5 and 
              citation_analysis.authority_score >= 0.5 and
              citation_analysis.citation_density >= 10):
            summary.append("- Overall literature quality: GOOD")
        else:
            summary.append("- Overall literature quality: NEEDS IMPROVEMENT")
        
        return "\n".join(summary)
    
    def _deduplicate_papers(self, papers: List[LiteraturePaper]) -> List[LiteraturePaper]:
        """Remove duplicate papers from list."""
        seen_titles = set()
        unique_papers = []
        
        for paper in papers:
            title_key = paper.title.lower().strip()
            if title_key not in seen_titles:
                seen_titles.add(title_key)
                unique_papers.append(paper)
        
        return unique_papers
    
    def generate_literature_report(self, analysis: LiteratureAnalysisResult) -> str:
        """Generate comprehensive literature analysis report."""
        report = []
        report.append("=" * 80)
        report.append("ADVANCED LITERATURE ANALYSIS REPORT")
        report.append("=" * 80)
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")
        
        # Analysis summary
        report.append("ANALYSIS SUMMARY:")
        for line in analysis.analysis_summary.split('\n'):
            report.append(f"  {line}")
        report.append("")
        
        # Citation analysis
        report.append("CITATION ANALYSIS:")
        ca = analysis.citation_analysis
        report.append(f"  Total Citations: {ca.total_citations}")
        report.append(f"  Relevant Citations: {ca.relevant_citations}")
        report.append(f"  Citation Density: {ca.citation_density:.1f} per 1000 words")
        report.append(f"  Recency Score: {ca.recency_score:.2f}")
        report.append(f"  Authority Score: {ca.authority_score:.2f}")
        
        if ca.irrelevant_citations:
            report.append(f"  Potentially Irrelevant: {len(ca.irrelevant_citations)} citations")
        report.append("")
        
        # Literature gaps
        if analysis.identified_gaps:
            report.append("IDENTIFIED RESEARCH GAPS:")
            for i, gap in enumerate(analysis.identified_gaps, 1):
                report.append(f"  {i}. {gap.description}")
                report.append(f"     Confidence: {gap.confidence_score:.2f}")
                report.append(f"     Impact: {gap.potential_impact}")
                if gap.supporting_evidence:
                    report.append(f"     Evidence: {'; '.join(gap.supporting_evidence)}")
            report.append("")
        
        # Related work summary
        report.append("RELATED WORK SUMMARY:")
        report.append(f"  Supporting Work: {len(analysis.supporting_work)} papers")
        report.append(f"  Competing Work: {len(analysis.competing_work)} papers")
        report.append(f"  Foundational Work: {len(analysis.foundational_work)} papers")
        report.append("")
        
        # Recommendations
        if analysis.recommendations:
            report.append("RECOMMENDATIONS:")
            for rec in analysis.recommendations:
                report.append(f"  • {rec}")
            report.append("")
        
        report.append("=" * 80)
        
        return "\n".join(report)
