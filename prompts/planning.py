"""Prompt helpers for research blueprint planning."""

from __future__ import annotations

from textwrap import dedent
from typing import Dict, List

from document_types import DocumentType, get_document_template


def get_blueprint_prompt(
    doc_type: DocumentType,
    topic: str,
    field: str,
    question: str,
) -> List[Dict[str, str]]:
    """Create a planning prompt that yields a structured research blueprint."""

    template = get_document_template(doc_type)

    system_prompt = dedent(
        f"""
        You are a principal investigator who specializes in outlining high-impact {template.doc_type.value.replace('_', ' ')}s.
        Design a complete research blueprint that a writing model will follow when drafting the manuscript.

        Planning expectations:
        • Provide a hierarchical outline covering every mandatory section for a {template.doc_type.value.replace('_', ' ')} in {field}.
        • Summarize the core thesis and 2–3 concrete contributions that answer the research question.
        • Detail the methodology plan, including datasets or simulation ingredients, key techniques, and validation checkpoints.
        • Specify quantitative evaluation strategy: metrics, baselines, ablation studies, and statistical tests that must appear.
        • Identify figures, tables, and diagrams that must be produced, with one line describing the insight each visual conveys.
        • Highlight potential risks, open questions, or assumptions that the draft must address explicitly.

        Constraints:
        • Keep the blueprint under 600 words.
        • Use Markdown with numbered sections and sub-bullets.
        • Label each major section with an explicit priority tag in brackets (e.g., [CRITICAL], [IMPORTANT], [NICE-TO-HAVE]).
        • Reference recent seminal works or datasets by name only—do not invent citations.
        • Emphasize reproducibility requirements relevant to {field} and the {template.prompt_focus} focus area.
        """
    ).strip()

    user_prompt = dedent(
        f"""
        TOPIC: {topic}
        FIELD: {field}
        RESEARCH QUESTION: {question}

        Produce the blueprint now. Close with a concise checklist summarizing the top five items the drafting model must satisfy.
        """
    ).strip()

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

