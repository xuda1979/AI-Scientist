"""
Research Tools — literature search, web search, and academic discovery.

Provides tools for:
- Semantic Scholar API (paper search, citation graphs, abstracts)
- arXiv API (preprint search, full-text access)
- Web search (via free APIs)
- Paper summarization
"""
from __future__ import annotations

import json
import logging
import re
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from typing import Any, Dict, List, Optional

from agent.tools import ToolDefinition, ToolParameter, ToolResult, ToolRegistry

logger = logging.getLogger(__name__)


# ═════════════════════════════════════════════════════════════════════════
# Semantic Scholar Search
# ═════════════════════════════════════════════════════════════════════════

def search_semantic_scholar(
    query: str,
    max_results: int = 10,
    year_range: str = "",
    fields_of_study: str = "",
    api_key: str = "",
) -> ToolResult:
    """
    Search Semantic Scholar for academic papers.

    Args:
        query: Search query string.
        max_results: Maximum number of results to return.
        year_range: Optional year range filter, e.g. "2020-2024".
        fields_of_study: Comma-separated fields, e.g. "Computer Science,Physics".
        api_key: Optional Semantic Scholar API key for higher rate limits.
    """
    try:
        base_url = "https://api.semanticscholar.org/graph/v1/paper/search"
        params = {
            "query": query,
            "limit": min(max_results, 50),
            "fields": "title,authors,year,abstract,venue,externalIds,citationCount,url,tldr",
        }
        if year_range:
            params["year"] = year_range
        if fields_of_study:
            params["fieldsOfStudy"] = fields_of_study

        url = f"{base_url}?{urllib.parse.urlencode(params)}"
        headers = {"Accept": "application/json"}
        if api_key:
            headers["x-api-key"] = api_key

        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read().decode())

        papers = []
        for item in data.get("data", []):
            authors = [a.get("name", "") for a in item.get("authors", [])]
            ext_ids = item.get("externalIds", {}) or {}
            tldr = item.get("tldr", {}) or {}

            papers.append({
                "title": item.get("title", ""),
                "authors": authors,
                "year": item.get("year"),
                "venue": item.get("venue", ""),
                "abstract": item.get("abstract", ""),
                "doi": ext_ids.get("DOI", ""),
                "arxiv_id": ext_ids.get("ArXiv", ""),
                "citation_count": item.get("citationCount", 0),
                "url": item.get("url", ""),
                "tldr": tldr.get("text", ""),
            })

        return ToolResult(True, data={
            "total": data.get("total", len(papers)),
            "papers": papers,
        })

    except urllib.error.HTTPError as e:
        if e.code == 429:
            return ToolResult(False, error="Rate limited by Semantic Scholar API. Wait and retry.")
        return ToolResult(False, error=f"HTTP {e.code}: {e.reason}")
    except Exception as exc:
        return ToolResult(False, error=f"Semantic Scholar search failed: {exc}")


def get_paper_details(paper_id: str, api_key: str = "") -> ToolResult:
    """
    Get detailed information about a specific paper from Semantic Scholar.

    Args:
        paper_id: Semantic Scholar paper ID, DOI, or ArXiv ID.
        api_key: Optional API key.
    """
    try:
        fields = "title,authors,year,abstract,venue,externalIds,citationCount,referenceCount,references,citations,tldr"
        url = f"https://api.semanticscholar.org/graph/v1/paper/{paper_id}?fields={fields}"
        headers = {"Accept": "application/json"}
        if api_key:
            headers["x-api-key"] = api_key

        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read().decode())

        return ToolResult(True, data=data)
    except Exception as exc:
        return ToolResult(False, error=f"Failed to get paper details: {exc}")


def get_citing_papers(paper_id: str, max_results: int = 10, api_key: str = "") -> ToolResult:
    """
    Get papers that cite a given paper (citation graph traversal).

    Args:
        paper_id: Semantic Scholar paper ID.
        max_results: Maximum number of citing papers.
        api_key: Optional API key.
    """
    try:
        fields = "title,authors,year,abstract,venue,citationCount"
        url = f"https://api.semanticscholar.org/graph/v1/paper/{paper_id}/citations?fields={fields}&limit={max_results}"
        headers = {"Accept": "application/json"}
        if api_key:
            headers["x-api-key"] = api_key

        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read().decode())

        citations = []
        for item in data.get("data", []):
            citing = item.get("citingPaper", {})
            citations.append({
                "title": citing.get("title", ""),
                "authors": [a.get("name", "") for a in citing.get("authors", [])],
                "year": citing.get("year"),
                "venue": citing.get("venue", ""),
                "citation_count": citing.get("citationCount", 0),
            })

        return ToolResult(True, data={"citations": citations})
    except Exception as exc:
        return ToolResult(False, error=f"Failed to get citing papers: {exc}")


# ═════════════════════════════════════════════════════════════════════════
# arXiv Search
# ═════════════════════════════════════════════════════════════════════════

def search_arxiv(
    query: str,
    max_results: int = 10,
    sort_by: str = "relevance",
    category: str = "",
) -> ToolResult:
    """
    Search arXiv for preprints and papers.

    Args:
        query: Search query string.
        max_results: Maximum results to return.
        sort_by: Sort order — "relevance" or "lastUpdatedDate" or "submittedDate".
        category: Optional arXiv category filter, e.g. "cs.AI", "quant-ph".
    """
    try:
        search_query = query
        if category:
            search_query = f"cat:{category} AND all:{query}"
        else:
            search_query = f"all:{query}"

        params = {
            "search_query": search_query,
            "max_results": min(max_results, 50),
            "sortBy": sort_by,
            "sortOrder": "descending",
        }

        url = f"http://export.arxiv.org/api/query?{urllib.parse.urlencode(params)}"
        req = urllib.request.Request(url)
        with urllib.request.urlopen(req, timeout=30) as resp:
            xml_data = resp.read().decode()

        # Parse Atom XML
        ns = {"atom": "http://www.w3.org/2005/Atom", "arxiv": "http://arxiv.org/schemas/atom"}
        root = ET.fromstring(xml_data)

        papers = []
        for entry in root.findall("atom:entry", ns):
            title = entry.findtext("atom:title", "", ns).strip().replace("\n", " ")
            abstract = entry.findtext("atom:summary", "", ns).strip().replace("\n", " ")
            published = entry.findtext("atom:published", "", ns)[:10]

            authors = []
            for author in entry.findall("atom:author", ns):
                name = author.findtext("atom:name", "", ns)
                if name:
                    authors.append(name)

            # Extract arXiv ID from the entry id URL
            entry_id = entry.findtext("atom:id", "", ns)
            arxiv_id = entry_id.split("/abs/")[-1] if "/abs/" in entry_id else entry_id

            # Get categories
            categories = [
                c.get("term", "")
                for c in entry.findall("atom:category", ns)
            ]

            # Get PDF link
            pdf_url = ""
            for link in entry.findall("atom:link", ns):
                if link.get("title") == "pdf":
                    pdf_url = link.get("href", "")

            papers.append({
                "title": title,
                "authors": authors,
                "abstract": abstract,
                "arxiv_id": arxiv_id,
                "published": published,
                "categories": categories,
                "pdf_url": pdf_url,
                "url": entry_id,
            })

        return ToolResult(True, data={"papers": papers})
    except Exception as exc:
        return ToolResult(False, error=f"arXiv search failed: {exc}")


# ═════════════════════════════════════════════════════════════════════════
# BibTeX Generation
# ═════════════════════════════════════════════════════════════════════════

def generate_bibtex(
    title: str,
    authors: str,
    year: str,
    venue: str = "",
    doi: str = "",
    arxiv_id: str = "",
    entry_type: str = "article",
) -> ToolResult:
    """
    Generate a BibTeX entry from paper metadata.

    Args:
        title: Paper title.
        authors: Authors as comma-separated string.
        year: Publication year.
        venue: Journal or conference name.
        doi: DOI if available.
        arxiv_id: arXiv ID if available.
        entry_type: BibTeX entry type (article, inproceedings, misc, etc.).
    """
    # Generate a cite key: firstauthor_year_firstword
    first_author = authors.split(",")[0].strip().split()[-1].lower() if authors else "unknown"
    first_word = re.sub(r"[^a-z]", "", title.split()[0].lower()) if title else "paper"
    cite_key = f"{first_author}{year}{first_word}"

    lines = [f"@{entry_type}{{{cite_key},"]
    lines.append(f"  title = {{{title}}},")
    lines.append(f"  author = {{{authors}}},")
    lines.append(f"  year = {{{year}}},")
    if venue:
        field_name = "journal" if entry_type == "article" else "booktitle"
        lines.append(f"  {field_name} = {{{venue}}},")
    if doi:
        lines.append(f"  doi = {{{doi}}},")
    if arxiv_id:
        lines.append(f"  eprint = {{{arxiv_id}}},")
        lines.append(f"  archivePrefix = {{arXiv}},")
    lines.append("}")

    bibtex = "\n".join(lines)
    return ToolResult(True, data={
        "bibtex": bibtex,
        "cite_key": cite_key,
    })


# ═════════════════════════════════════════════════════════════════════════
# Paper Summarization (via LLM)
# ═════════════════════════════════════════════════════════════════════════

def summarize_paper(
    title: str,
    abstract: str,
    research_context: str = "",
) -> ToolResult:
    """
    Prepare a structured summary request for a paper.
    (Actual summarization happens via the LLM in the agent loop.)

    Args:
        title: Paper title.
        abstract: Paper abstract.
        research_context: How this paper relates to the current research.
    """
    summary_prompt = (
        f"Summarize the following paper for use in a literature review:\n\n"
        f"**Title**: {title}\n"
        f"**Abstract**: {abstract}\n\n"
        f"Context: {research_context}\n\n"
        f"Provide:\n"
        f"1. Key contributions (2-3 sentences)\n"
        f"2. Methodology used\n"
        f"3. Main results\n"
        f"4. Relevance to our research\n"
        f"5. Potential gaps or limitations"
    )
    return ToolResult(True, data={"summary_prompt": summary_prompt})


# ═════════════════════════════════════════════════════════════════════════
# Registration
# ═════════════════════════════════════════════════════════════════════════

def register_research_tools(registry: ToolRegistry, api_key: str = "") -> None:
    """Register all research tools with the tool registry."""

    # Semantic Scholar search
    registry.register(ToolDefinition(
        name="search_papers",
        description="Search Semantic Scholar for academic papers matching a query. Returns titles, authors, abstracts, citation counts, and DOIs.",
        parameters=[
            ToolParameter("query", "string", "Search query for finding papers"),
            ToolParameter("max_results", "integer", "Maximum number of results (default 10)", required=False, default=10),
            ToolParameter("year_range", "string", "Year range filter, e.g. '2020-2024'", required=False, default=""),
            ToolParameter("fields_of_study", "string", "Comma-separated fields, e.g. 'Computer Science'", required=False, default=""),
        ],
        handler=lambda **kwargs: search_semantic_scholar(api_key=api_key, **kwargs),
        category="research",
    ))

    # Paper details
    registry.register(ToolDefinition(
        name="get_paper_details",
        description="Get detailed information about a specific paper including references and citations. Use with a Semantic Scholar ID, DOI, or ArXiv ID.",
        parameters=[
            ToolParameter("paper_id", "string", "Paper identifier (Semantic Scholar ID, DOI, or ArXiv ID)"),
        ],
        handler=lambda **kwargs: get_paper_details(api_key=api_key, **kwargs),
        category="research",
    ))

    # Citation graph
    registry.register(ToolDefinition(
        name="get_citing_papers",
        description="Get papers that cite a given paper. Useful for finding follow-up work and understanding impact.",
        parameters=[
            ToolParameter("paper_id", "string", "Semantic Scholar paper ID"),
            ToolParameter("max_results", "integer", "Maximum results", required=False, default=10),
        ],
        handler=lambda **kwargs: get_citing_papers(api_key=api_key, **kwargs),
        category="research",
    ))

    # arXiv search
    registry.register(ToolDefinition(
        name="search_arxiv",
        description="Search arXiv for preprints and papers. Good for finding the latest research not yet indexed elsewhere.",
        parameters=[
            ToolParameter("query", "string", "Search query"),
            ToolParameter("max_results", "integer", "Maximum results", required=False, default=10),
            ToolParameter("sort_by", "string", "Sort by: relevance, lastUpdatedDate, submittedDate", required=False, default="relevance"),
            ToolParameter("category", "string", "arXiv category filter, e.g. 'cs.AI', 'quant-ph'", required=False, default=""),
        ],
        handler=search_arxiv,
        category="research",
    ))

    # BibTeX generation
    registry.register(ToolDefinition(
        name="generate_bibtex",
        description="Generate a properly formatted BibTeX entry from paper metadata.",
        parameters=[
            ToolParameter("title", "string", "Paper title"),
            ToolParameter("authors", "string", "Authors as comma-separated string"),
            ToolParameter("year", "string", "Publication year"),
            ToolParameter("venue", "string", "Journal or conference name", required=False, default=""),
            ToolParameter("doi", "string", "DOI", required=False, default=""),
            ToolParameter("arxiv_id", "string", "arXiv ID", required=False, default=""),
            ToolParameter("entry_type", "string", "BibTeX type: article, inproceedings, misc", required=False, default="article"),
        ],
        handler=generate_bibtex,
        category="research",
    ))

    # Paper summarization
    registry.register(ToolDefinition(
        name="summarize_paper",
        description="Prepare a structured summary of a paper for the literature review.",
        parameters=[
            ToolParameter("title", "string", "Paper title"),
            ToolParameter("abstract", "string", "Paper abstract text"),
            ToolParameter("research_context", "string", "How this relates to current research", required=False, default=""),
        ],
        handler=summarize_paper,
        category="research",
    ))
