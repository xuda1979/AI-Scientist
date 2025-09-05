"""Run validation checks in parallel using inexpensive models."""
from __future__ import annotations
import asyncio
from typing import Dict, Any


async def run_parallel_checks(paper_content: str, model: str) -> Dict[str, Any]:
    """Run grammar, style, figure, and bibliography checks concurrently."""
    from sciresearch_workflow import (
        _universal_chat,
        _validate_figures_tables,
        _validate_bibliography,
    )

    async def grammar() -> str:
        messages = [{"role": "user", "content": "List grammar issues in bullet points:\n" + paper_content[:4000]}]
        return await asyncio.to_thread(_universal_chat, messages, model, None, "grammar", None)

    async def style() -> str:
        messages = [{"role": "user", "content": "List style issues in bullet points:\n" + paper_content[:4000]}]
        return await asyncio.to_thread(_universal_chat, messages, model, None, "style", None)

    async def figures() -> Any:
        return await asyncio.to_thread(_validate_figures_tables, paper_content)

    async def bibliography() -> Any:
        return await asyncio.to_thread(_validate_bibliography, paper_content)

    results = await asyncio.gather(grammar(), style(), figures(), bibliography())
    return {
        "grammar": results[0],
        "style": results[1],
        "figures": results[2],
        "bibliography": results[3],
    }
