"""Novelty vetting utilities for research ideation and manuscript validation."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, asdict
from datetime import datetime
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

try:
    import requests  # type: ignore

    HAS_REQUESTS = True
except Exception:  # pragma: no cover - requests optional in runtime env
    HAS_REQUESTS = False
import urllib.parse
import urllib.request


@dataclass
class RetrievedWork:
    """Metadata about a retrieved paper that may compete with a proposal."""

    title: str
    url: str
    year: Optional[int]
    doi: Optional[str]
    similarity: float
    source: str

    def short_label(self) -> str:
        year_display = str(self.year) if self.year else "n.d."
        return f"{self.title} ({year_display})"


@dataclass
class NoveltyAssessment:
    """Result of novelty vetting for a concept or manuscript."""

    query: str
    field: str
    novelty_score: float
    retrieved_works: List[RetrievedWork]
    diagnostics: str
    timestamp: str

    def to_dict(self) -> Dict[str, Any]:
        """Serialize the assessment for persistence."""

        return {
            "query": self.query,
            "field": self.field,
            "novelty_score": self.novelty_score,
            "diagnostics": self.diagnostics,
            "timestamp": self.timestamp,
            "retrieved_works": [asdict(work) for work in self.retrieved_works],
        }


class NoveltyVetter:
    """Perform retrieval-backed novelty checks and summarise diagnostics."""

    CROSSREF_ENDPOINT = "https://api.crossref.org/works"

    def __init__(
        self,
        cache_dir: Optional[Path] = None,
        max_results: int = 6,
        request_timeout: int = 12,
    ) -> None:
        self.max_results = max_results
        self.request_timeout = request_timeout
        self.cache_dir = Path(cache_dir or Path.home() / ".sciresearch_cache" / "novelty")
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def assess_concept(
        self,
        title: str,
        question: str,
        field: str,
        extra_queries: Optional[Iterable[str]] = None,
    ) -> NoveltyAssessment:
        """Assess novelty for an ideation concept before blueprinting."""

        queries = self._build_queries(title, question, field, extra_queries)
        retrieved = self._collect_unique_results(queries)
        novelty_score, diagnostics = self._score_novelty(queries, retrieved)

        return NoveltyAssessment(
            query=" | ".join(queries[:3]),
            field=field,
            novelty_score=novelty_score,
            retrieved_works=retrieved,
            diagnostics=diagnostics,
            timestamp=datetime.utcnow().isoformat(timespec="seconds"),
        )

    def assess_manuscript(self, content: str, field: str) -> NoveltyAssessment:
        """Assess novelty for an in-progress manuscript."""

        claims = self._extract_claims(content)
        queries = list(dict.fromkeys(claims))  # Stable dedupe
        retrieved = self._collect_unique_results(queries)
        novelty_score, diagnostics = self._score_novelty(queries, retrieved)

        dominant_query = queries[0] if queries else "manuscript"
        return NoveltyAssessment(
            query=dominant_query,
            field=field,
            novelty_score=novelty_score,
            retrieved_works=retrieved,
            diagnostics=diagnostics,
            timestamp=datetime.utcnow().isoformat(timespec="seconds"),
        )

    def blueprint_digest(self, assessment: NoveltyAssessment, top_n: int = 3) -> str:
        """Create a succinct digest for blueprint planning prompts."""

        if not assessment.retrieved_works:
            return (
                "No competing publications were discovered for the current idea. "
                "Emphasize articulating a unique contribution path."
            )

        closest = assessment.retrieved_works[:top_n]
        lines = [
            f"Baseline novelty score: {assessment.novelty_score:.2f} (1.0 = highly novel).",
            "Closest retrieved works:",
        ]
        for work in closest:
            lines.append(
                f"• {work.short_label()} — similarity {work.similarity:.2f}; {work.url or 'no url'}"
            )
        lines.append("Explicitly differentiate from these studies when outlining sections.")
        return "\n".join(lines)

    def revision_diagnostics(self, assessment: NoveltyAssessment, max_items: int = 3) -> List[str]:
        """Create diagnostic messages suitable for revision loops."""

        issues: List[str] = []
        if not assessment.retrieved_works:
            issues.append(
                "INFO: Novelty scan found no close matches; maintain rigorous differentiation."  # noqa: E501
            )
            return issues

        threshold = 0.75
        for work in assessment.retrieved_works[:max_items]:
            if work.similarity >= threshold:
                issues.append(
                    (
                        f"CRITICAL: Manuscript overlaps with {work.short_label()} (similarity"
                        f" {work.similarity:.2f}). Clarify novel mechanisms."
                    )
                )
            else:
                issues.append(
                    (
                        f"WARNING: Related work {work.short_label()} shares key themes "
                        f"(similarity {work.similarity:.2f}). Highlight differentiation."
                    )
                )

        issues.append(f"INFO: Novelty diagnostics summary — {assessment.diagnostics}")
        return issues

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _build_queries(
        self,
        title: str,
        question: str,
        field: str,
        extra_queries: Optional[Iterable[str]] = None,
    ) -> List[str]:
        queries = [title.strip(), question.strip(), f"{field} {title}".strip()]
        if extra_queries:
            for value in extra_queries:
                if value and value.strip():
                    queries.append(value.strip())

        # Truncate overly long queries to keep API requests compact
        compact_queries = []
        for query in queries:
            if not query:
                continue
            compact_queries.append(query[:180])
        return compact_queries[:8]

    def _collect_unique_results(self, queries: Iterable[str]) -> List[RetrievedWork]:
        seen: Dict[str, RetrievedWork] = {}
        for query in queries:
            if not query:
                continue
            for work in self._search_crossref(query):
                key = work.title.lower()
                if key not in seen or work.similarity > seen[key].similarity:
                    seen[key] = work

        # Sort by highest similarity (strongest overlap first)
        results = sorted(seen.values(), key=lambda w: w.similarity, reverse=True)
        return results

    def _search_crossref(self, query: str) -> List[RetrievedWork]:
        cache_path = self._cache_path(query)
        if cache_path.exists():
            try:
                with cache_path.open("r", encoding="utf-8") as fh:
                    cached = json.load(fh)
                return [self._work_from_dict(item) for item in cached]
            except Exception:
                pass

        params = {
            "query": query,
            "rows": str(self.max_results),
            "select": "title,DOI,URL,issued",
        }

        try:
            data = self._perform_request(params)
        except Exception as exc:  # pragma: no cover - network failures vary
            return [
                RetrievedWork(
                    title=f"[retrieval failure] {query[:40]}...",
                    url="",
                    year=None,
                    doi=None,
                    similarity=0.0,
                    source=f"crossref:{exc}",
                )
            ]

        message = data.get("message", {})
        items = message.get("items", [])
        works: List[RetrievedWork] = []
        for item in items:
            title_list = item.get("title") or []
            if not title_list:
                continue
            title = title_list[0].strip()
            similarity = SequenceMatcher(None, query.lower(), title.lower()).ratio()
            issued = item.get("issued", {})
            year = None
            if isinstance(issued, dict):
                parts = issued.get("date-parts")
                if parts and isinstance(parts, list) and parts[0]:
                    year = parts[0][0]

            works.append(
                RetrievedWork(
                    title=title,
                    url=item.get("URL", ""),
                    year=year,
                    doi=item.get("DOI"),
                    similarity=similarity,
                    source="crossref",
                )
            )

        try:
            with cache_path.open("w", encoding="utf-8") as fh:
                json.dump([asdict(work) for work in works], fh, indent=2)
        except Exception:
            pass

        return works

    def _perform_request(self, params: Dict[str, str]) -> Dict[str, Any]:
        encoded = urllib.parse.urlencode(params)
        url = f"{self.CROSSREF_ENDPOINT}?{encoded}"
        if HAS_REQUESTS:
            response = requests.get(url, timeout=self.request_timeout)
            response.raise_for_status()
            return response.json()

        with urllib.request.urlopen(url, timeout=self.request_timeout) as handle:
            data = handle.read()
        return json.loads(data.decode("utf-8"))

    def _score_novelty(
        self,
        queries: Iterable[str],
        retrieved: List[RetrievedWork],
    ) -> Tuple[float, str]:
        if not retrieved:
            return 0.85, "No overlapping works retrieved."

        if any(work.source.startswith("crossref:") for work in retrieved):
            return 0.4, "Retrieval failure prevented overlap analysis."

        max_similarity = max(work.similarity for work in retrieved)
        avg_similarity = sum(work.similarity for work in retrieved) / max(len(retrieved), 1)

        current_year = datetime.utcnow().year
        recent_penalty = 0.0
        for work in retrieved:
            if work.year and current_year - work.year <= 2 and work.similarity >= 0.6:
                recent_penalty = max(recent_penalty, 0.1)

        novelty_raw = max(0.0, 1.0 - max_similarity - recent_penalty)
        novelty_score = round(min(1.0, novelty_raw + 0.15), 3)

        diagnostics = (
            f"max similarity {max_similarity:.2f}, avg similarity {avg_similarity:.2f}, "
            f"recent overlap penalty {recent_penalty:.2f}"
        )

        return novelty_score, diagnostics

    def _extract_claims(self, content: str) -> List[str]:
        sentences: List[str] = []

        # Simple heuristics: use sentences containing contribution phrases
        triggers = ["we propose", "we introduce", "we present", "our contribution", "this paper"]
        for sentence in content.split(". "):
            if any(trigger in sentence.lower() for trigger in triggers):
                sentences.append(sentence.strip())

        # Fallback to title-like heuristics if no sentences found
        if not sentences:
            sentences = content.splitlines()[:3]

        trimmed = []
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            trimmed.append(sentence[:200])

        return trimmed or [content[:200]]

    def _cache_path(self, query: str) -> Path:
        digest = hashlib.sha1(query.encode("utf-8")).hexdigest()
        return self.cache_dir / f"{digest}.json"

    def _work_from_dict(self, data: Dict[str, Any]) -> RetrievedWork:
        return RetrievedWork(
            title=data.get("title", ""),
            url=data.get("url", ""),
            year=data.get("year"),
            doi=data.get("doi"),
            similarity=float(data.get("similarity", 0.0)),
            source=data.get("source", "crossref"),
        )


__all__ = [
    "NoveltyAssessment",
    "NoveltyVetter",
    "RetrievedWork",
]

