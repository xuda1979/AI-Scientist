"""
AI chat interface and utilities for SciResearch Workflow.
"""
from __future__ import annotations
import logging
from typing import List, Dict, Optional, Any
from pathlib import Path

logger = logging.getLogger(__name__)


class AIChat:
    """Centralized AI chat interface with fallback support."""
    
    def __init__(self, primary_model: str, fallback_models: Optional[List[str]] = None):
        self.primary_model = primary_model
        self.fallback_models = fallback_models or []
        self.stats = {"calls": 0, "failures": 0, "fallbacks": 0}
    
    def chat(
        self, 
        messages: List[Dict[str, str]], 
        model: Optional[str] = None,
        request_timeout: Optional[int] = None,
        prompt_type: str = "general",
        pdf_path: Optional[Path] = None
    ) -> str:
        """Send chat request with automatic fallback handling."""
        target_model = model or self.primary_model
        self.stats["calls"] += 1
        
        try:
            response = self._make_api_call(messages, target_model, request_timeout, pdf_path)
            logger.info(f"SUCCESS: {prompt_type} call successful with {target_model}")
            return response
        except Exception as e:
            logger.warning(f"FAILED: {prompt_type} call failed with {target_model}: {e}")
            self.stats["failures"] += 1
            
            # Try fallback models
            for fallback_model in self.fallback_models:
                try:
                    self.stats["fallbacks"] += 1
                    response = self._make_api_call(messages, fallback_model, request_timeout, pdf_path)
                    logger.info(f"SUCCESS: {prompt_type} call successful with fallback {fallback_model}")
                    return response
                except Exception as fallback_error:
                    logger.warning(f"FAILED: Fallback {fallback_model} failed: {fallback_error}")
                    continue
            
            # All models failed
            raise RuntimeError(f"All models failed for {prompt_type}. Last error: {e}")
    
    def _make_api_call(
        self, 
        messages: List[Dict[str, str]], 
        model: str, 
        timeout: Optional[int],
        pdf_path: Optional[Path]
    ) -> str:
        """Make the actual API call."""
        try:
            # Try to import and use OpenAI
            from openai import OpenAI
            import os

            # Determine which API to call and configure the client
            requested_model = model
            if model == "oss-120b":
                requested_model = "openai/gpt-oss-120b:free"

            openrouter_key = os.getenv("OPENROUTER_API_KEY")
            use_openrouter = bool(openrouter_key and (
                requested_model.startswith("openai/") or "oss" in requested_model
            ))

            if use_openrouter:
                client = OpenAI(
                    base_url="https://openrouter.ai/api/v1",
                    api_key=openrouter_key,
                )

                extra_headers = {}
                referer = os.getenv("OPENROUTER_HTTP_REFERER")
                if referer:
                    extra_headers["HTTP-Referer"] = referer

                site_title = os.getenv("OPENROUTER_X_TITLE")
                if site_title:
                    extra_headers["X-Title"] = site_title

                response = client.chat.completions.create(
                    model=requested_model,
                    messages=messages,
                    max_tokens=4000,
                    temperature=0.7,
                    timeout=timeout or 3600,
                    extra_headers=extra_headers,
                    extra_body={},
                )
            else:
                # Initialize OpenAI client
                api_key = os.getenv('OPENAI_API_KEY')
                if not api_key:
                    raise ValueError("OPENAI_API_KEY environment variable not set")

                client = OpenAI(api_key=api_key)

                # Make the API call
                if requested_model.startswith('gpt-5') or requested_model.startswith('o1'):
                    # GPT-5 and o1 models use max_completion_tokens and only support default temperature
                    response = client.chat.completions.create(
                        model=requested_model,
                        messages=messages,
                        max_completion_tokens=4000,
                        timeout=timeout or 3600
                        # Note: temperature not specified for GPT-5 (uses default of 1.0)
                    )
                else:
                    # GPT-4 and earlier models use max_tokens and support custom temperature
                    response = client.chat.completions.create(
                        model=requested_model,
                        messages=messages,
                        max_tokens=4000,
                        temperature=0.7,
                        timeout=timeout or 3600
                    )

            return response.choices[0].message.content
            
        except ImportError:
            logger.warning("OpenAI library not available, using placeholder response")
            return self._placeholder_response(messages)
        except Exception as e:
            logger.error(f"OpenAI API call failed: {e}")
            return self._placeholder_response(messages)
    
    def _placeholder_response(self, messages: List[Dict[str, str]]) -> str:
        """Generate placeholder response when API is not available."""
        if isinstance(messages, list) and messages:
            content = messages[0].get("content", "")
            if "ideation" in content.lower():
                return """## IDEATION ANALYSIS

This placeholder response demonstrates the structure of an ideation and drafting reply
when the language model API is unavailable. Replace the bracketed sections below with
domain-specific insight once live model access is restored.

### Research Idea Options
1. **Idea A**: [Summarize a potential direction and why it might matter.]
2. **Idea B**: [Describe an alternative approach highlighting a distinct methodology.]
3. **Idea C**: [Outline a complementary avenue or comparative baseline study.]

### SELECTED RESEARCH DIRECTION
**Title**: [Placeholder title for the preferred idea]
**Rationale**: [Brief justification describing novelty, impact, and feasibility.]

## COMPLETE LATEX PAPER
```latex
\documentclass{article}
\usepackage{amsmath}

\begin{document}
\title{Placeholder Research Manuscript}
\author{Automated Workflow}
\maketitle

\begin{abstract}
This placeholder manuscript illustrates the expected paper structure produced by the
workflow when model calls succeed. Replace this text with a concise summary of the
actual study, including the problem statement, approach, and primary findings.
\end{abstract}

\section{Introduction}
[Introduce the broad research problem, motivate its importance, and summarise key
contributions without referencing any specific prior paper.]

\section{Methodology}
[Outline the general methodology, experimental setup, or theoretical framework that the
final paper should contain once generated.]

\section{Results}
[Describe the types of results, evaluations, or analyses that will support the claims in
the completed manuscript.]

\section{Conclusion}
[Summarise anticipated takeaways and future work directions in a domain-agnostic manner.]

\end{document}
```"""
            else:
                return (
                    "Structured placeholder response for research paper refinement. "
                    "Actual reviewer insights will appear once model access is restored."
                )

        return (
            "Placeholder response generated without referencing a specific paper. "
            "Connect a language model provider to receive full results."
        )

    
    def get_stats(self) -> Dict[str, Any]:
        """Get usage statistics."""
        return {
            **self.stats,
            "success_rate": (self.stats["calls"] - self.stats["failures"]) / max(1, self.stats["calls"]),
            "fallback_rate": self.stats["fallbacks"] / max(1, self.stats["calls"])
        }


def create_chat_interface(config) -> AIChat:
    """Create AI chat interface from configuration."""
    return AIChat(
        primary_model=getattr(config, 'primary_model', 'gpt-4'),
        fallback_models=getattr(config, 'fallback_models', [])
    )
