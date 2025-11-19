
from __future__ import annotations
from typing import Dict, Any, Optional

class BaseLLMDriver:
    def generate(self, prompt: str, max_tokens: int = 256, temperature: float = 0.7) -> Dict[str, Any]:
        raise NotImplementedError

class OpenAIChatDriver(BaseLLMDriver):
    """Thin wrapper around openai ChatCompletions API. Requires OPENAI_API_KEY in env.
    Not imported unless used to avoid hard dependency.
    """
    def __init__(self, model: str = "gpt-4o-mini"):
        self.model = model

    def generate(self, prompt: str, max_tokens: int = 256, temperature: float = 0.7) -> Dict[str, Any]:
        try:
            import os
            import openai  # type: ignore
            client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            r = client.chat.completions.create(
                model=self.model,
                messages=[{"role":"user","content":prompt}],
                max_tokens=max_tokens,
                temperature=temperature,
            )
            text = r.choices[0].message.content
            return {"text": text, "finish_reason": r.choices[0].finish_reason}
        except Exception as e:
            return {"text": f"[driver-error] {e}", "finish_reason": "error"}

class EchoDriver(BaseLLMDriver):
    """Offline stub: echoes a templated response for development."""
    def generate(self, prompt: str, max_tokens: int = 256, temperature: float = 0.7) -> Dict[str, Any]:
        sample = prompt[:max_tokens]
        return {"text": f"Answer: {sample}\n[stub]", "finish_reason": "stop"}
