"""Workflow step for generating a structured research blueprint."""

from __future__ import annotations

from typing import Optional


def generate_research_blueprint(
    topic: str,
    field: str,
    question: str,
    model: str,
    request_timeout: int,
    config,
    novelty_digest: Optional[str] = None,
) -> Optional[str]:
    """Generate a research blueprint prior to drafting the paper."""

    if not getattr(config, "enable_blueprint_planning", True):
        return None

    from document_types import infer_document_type
    from prompts.planning import get_blueprint_prompt
    from sciresearch_workflow import _universal_chat

    doc_type = infer_document_type(topic=topic, field=field, question=question)
    prompt_messages = get_blueprint_prompt(
        doc_type,
        topic,
        field,
        question,
        novelty_digest=novelty_digest,
    )

    blueprint_text = _universal_chat(
        prompt_messages,
        model=model,
        request_timeout=request_timeout,
        prompt_type="research_blueprint",
        fallback_models=getattr(config, "fallback_models", None),
    )

    return blueprint_text.strip() or None

