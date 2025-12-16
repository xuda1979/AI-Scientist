"""Utilities for managing Yunwu API connections (OpenAI-compatible endpoint)."""
from __future__ import annotations

import os
import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

__all__ = [
    "YunwuClient",
    "YunwuConnectionError",
    "get_yunwu_client",
    "YUNWU_API_BASE",
    "YUNWU_MODEL_ALIASES",
]

logger = logging.getLogger(__name__)

# Yunwu API endpoint (OpenAI-compatible)
YUNWU_API_BASE = "https://yunwu.ai/v1"

# Model name aliases - map user-friendly names to actual API model names
# Users can use either the alias or the full model name
YUNWU_MODEL_ALIASES = {
    # Claude 4 models (Anthropic 2025)
    "opus-4.5": "claude-opus-4-20250514",  # Claude Opus 4
    "opus-4": "claude-opus-4-20250514",
    "claude-4-opus": "claude-opus-4-20250514",
    "sonnet-4": "claude-sonnet-4-20250514",
    "claude-4-sonnet": "claude-sonnet-4-20250514",
    # Claude 3.5 models
    "sonnet-3.5": "claude-3-5-sonnet-20241022",
    "claude-3.5-sonnet": "claude-3-5-sonnet-20241022",
    # Claude 3 models
    "opus-3": "claude-3-opus-20240229",
    "claude-3-opus": "claude-3-opus-20240229",
    # OpenAI models (pass through)
    "gpt-4o": "gpt-4o",
    "gpt-4": "gpt-4",
}

# Global state
_YUNWU_CLIENT: Optional["YunwuClient"] = None
_YUNWU_ENABLED: bool = False


class YunwuConnectionError(RuntimeError):
    """Raised when Yunwu API connectivity cannot be established."""


@dataclass
class YunwuClient:
    """Client for interacting with Yunwu API (OpenAI-compatible endpoint)."""
    
    api_key: str
    base_url: str = YUNWU_API_BASE
    
    def __post_init__(self):
        """Validate configuration on initialization."""
        if not self.api_key:
            raise YunwuConnectionError("Yunwu API key is required")
    
    def _get_openai_client(self):
        """Create an OpenAI client configured for Yunwu API."""
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise YunwuConnectionError("The 'openai' package is not installed") from exc
        
        return OpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
        )
    
    def chat(
        self,
        messages: List[Dict[str, str]],
        model: str,
        temperature: float = 0.7,
        timeout: Optional[int] = None,
        max_tokens: Optional[int] = None,
    ) -> str:
        """
        Send a chat completion request to Yunwu API.
        
        Args:
            messages: List of chat messages
            model: Model name (e.g., 'opus-4.5', 'claude-3-opus', etc.)
            temperature: Sampling temperature
            timeout: Request timeout in seconds
            max_tokens: Maximum tokens in response
            
        Returns:
            The assistant's response text
        """
        client = self._get_openai_client()
        
        # Map model aliases to actual API model names
        actual_model = YUNWU_MODEL_ALIASES.get(model, model)
        if actual_model != model:
            print(f"[Yunwu API] Mapping model alias '{model}' -> '{actual_model}'")
        
        try:
            kwargs: Dict[str, Any] = {
                "model": actual_model,
                "messages": messages,
                "temperature": temperature,
            }
            
            if timeout:
                kwargs["timeout"] = timeout
            if max_tokens:
                kwargs["max_tokens"] = max_tokens
            
            print(f"[Yunwu API] Sending request to {self.base_url} with model={actual_model}...")
            
            response = client.chat.completions.create(**kwargs)
            
            content = response.choices[0].message.content
            print(f"[Yunwu API] Response received: {len(content):,} characters")
            
            return content
            
        except Exception as e:
            logger.error(f"Yunwu API request failed: {e}")
            raise YunwuConnectionError(f"Yunwu API request failed: {e}") from e
    
    def ping(self) -> bool:
        """Check if the Yunwu API is reachable."""
        try:
            client = self._get_openai_client()
            # Try to list models as a connectivity check
            client.models.list()
            return True
        except Exception as e:
            logger.warning(f"Yunwu API ping failed: {e}")
            return False
    
    def list_models(self) -> List[str]:
        """List available models from Yunwu API."""
        try:
            client = self._get_openai_client()
            models = client.models.list()
            return [m.id for m in models.data]
        except Exception as e:
            logger.warning(f"Failed to list Yunwu models: {e}")
            return []


def configure_yunwu(api_key: Optional[str] = None, base_url: Optional[str] = None) -> YunwuClient:
    """
    Configure and return the global Yunwu client.
    
    Args:
        api_key: Yunwu API key (defaults to YUNWU_API_KEY env var)
        base_url: API base URL (defaults to YUNWU_API_BASE)
        
    Returns:
        Configured YunwuClient instance
    """
    global _YUNWU_CLIENT, _YUNWU_ENABLED
    
    # Get API key from parameter or environment
    key = api_key or os.environ.get("YUNWU_API_KEY")
    if not key:
        raise YunwuConnectionError(
            "Yunwu API key not provided. Set YUNWU_API_KEY environment variable "
            "or pass --yunwu-api-key parameter."
        )
    
    url = base_url or os.environ.get("YUNWU_API_BASE", YUNWU_API_BASE)
    
    _YUNWU_CLIENT = YunwuClient(api_key=key, base_url=url)
    _YUNWU_ENABLED = True
    
    print(f"✓ Yunwu API configured with endpoint: {url}")
    
    return _YUNWU_CLIENT


def get_yunwu_client() -> Optional[YunwuClient]:
    """Get the configured Yunwu client, or None if not configured."""
    return _YUNWU_CLIENT


def is_yunwu_enabled() -> bool:
    """Check if Yunwu API is enabled."""
    return _YUNWU_ENABLED


def set_yunwu_enabled(enabled: bool) -> None:
    """Set whether Yunwu API should be used."""
    global _YUNWU_ENABLED
    _YUNWU_ENABLED = enabled
