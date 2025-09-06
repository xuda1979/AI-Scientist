from __future__ import annotations
from typing import Optional


def generate_initial_draft(
    topic: str,
    field: str,
    question: str,
    user_prompt: Optional[str],
    model: str,
    request_timeout: int,
    config,
) -> str:
    """Generate the initial paper draft."""
    from sciresearch_workflow import (
        _generate_best_initial_draft_candidate,
        _initial_draft_prompt,
        _universal_chat,
    )

    if getattr(config, "use_test_time_scaling", False) and getattr(config, "initial_draft_candidates", 1) > 1:
        return _generate_best_initial_draft_candidate(
            topic,
            field,
            question,
            user_prompt,
            model,
            request_timeout,
            config,
            config.initial_draft_candidates,
        )

    return _universal_chat(
        _initial_draft_prompt(topic, field, question, user_prompt),
        model=model,
        request_timeout=request_timeout,
        prompt_type="initial_draft",
        fallback_models=config.fallback_models,
    )
