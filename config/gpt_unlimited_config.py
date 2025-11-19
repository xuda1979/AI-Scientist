"""
GPT Unlimited Output Configuration
===================================
Ensures no token limits for all GPT models, especially GPT-5-Pro.
"""

import os
from typing import Optional, Dict, Any

# =============================================================================
# UNLIMITED OUTPUT CONFIGURATION
# =============================================================================

UNLIMITED_OUTPUT_CONFIG = {
    # Model-specific configurations
    "gpt-5-pro": {
        "max_tokens": None,  # None = unlimited
        "max_completion_tokens": None,  # For newer API versions
        "timeout": 600,  # 10 minutes
        "stream": True,  # Stream long responses
        "temperature": 0.7,
        "top_p": 0.95,
    },
    
    "gpt-5": {
        "max_tokens": None,
        "max_completion_tokens": None,
        "timeout": 600,
        "stream": True,
        "temperature": 0.7,
        "top_p": 0.95,
    },
    
    "gpt-4": {
        "max_tokens": None,
        "max_completion_tokens": None,
        "timeout": 300,
        "stream": True,
        "temperature": 0.7,
    },
    
    "o1-preview": {
        "max_completion_tokens": 32768,  # o1 models have fixed limits
        "timeout": 600,
        "temperature": 1.0,  # o1 doesn't support temp control
    },
    
    "o1-mini": {
        "max_completion_tokens": 65536,
        "timeout": 300,
        "temperature": 1.0,
    },
    
    # Default for any model
    "default": {
        "max_tokens": None,
        "max_completion_tokens": None,
        "timeout": 300,
        "stream": True,
        "temperature": 0.7,
    }
}

# Retry configuration
RETRY_CONFIG = {
    "max_retries": 5,
    "initial_delay": 2.0,
    "max_delay": 60.0,
    "exponential_base": 2.0,
    "jitter": True,
}

# Chunking configuration for very large documents
CHUNKING_CONFIG = {
    "enable_auto_chunking": True,
    "chunk_threshold_lines": 1000,  # Chunk if > this many lines
    "chunk_size_lines": 400,
    "overlap_lines": 50,
    "preserve_structure": True,  # Keep preamble/bib intact
}

# Validation configuration
VALIDATION_CONFIG = {
    "validate_completeness": True,
    "check_latex_structure": True,
    "check_size_ratio": True,
    "min_size_ratio": 0.7,  # New content should be >= 70% of original
    "max_size_ratio": 2.0,  # New content should be <= 200% of original
    "check_truncation_markers": True,
    "require_end_document": True,
}

def get_model_config(
    model: str,
    override_max_tokens: Optional[int] = None,
    custom_overrides: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Get configuration for a specific model with unlimited output.
    
    Args:
        model: Model name (e.g., "gpt-5-pro", "gpt-4")
        override_max_tokens: If provided, overrides the default None
        custom_overrides: Additional config overrides
        
    Returns:
        Configuration dictionary
    """
    # Get model-specific config or default
    config = UNLIMITED_OUTPUT_CONFIG.get(
        model,
        UNLIMITED_OUTPUT_CONFIG["default"]
    ).copy()
    
    # Override max_tokens if specified
    if override_max_tokens is not None:
        config["max_tokens"] = override_max_tokens
        config["max_completion_tokens"] = override_max_tokens
    
    # Apply custom overrides
    if custom_overrides:
        config.update(custom_overrides)
    
    return config

def get_api_params(
    model: str,
    messages: list,
    **kwargs
) -> Dict[str, Any]:
    """
    Get API parameters with unlimited output enabled.
    
    Args:
        model: Model name
        messages: Chat messages
        **kwargs: Additional parameters to override
        
    Returns:
        Dictionary of API parameters
    """
    config = get_model_config(model)
    
    # Build API parameters
    params = {
        "model": model,
        "messages": messages,
    }
    
    # Add optional parameters
    if "temperature" in config and model not in ["o1-preview", "o1-mini"]:
        params["temperature"] = config.get("temperature", 0.7)
    
    if "top_p" in config:
        params["top_p"] = config.get("top_p", 0.95)
    
    if "stream" in config:
        params["stream"] = config.get("stream", True)
    
    # Handle max_tokens - use EITHER max_completion_tokens OR max_tokens, not both
    # Prefer max_completion_tokens for newer models
    if config.get("max_completion_tokens") is not None:
        params["max_completion_tokens"] = config["max_completion_tokens"]
    elif config.get("max_tokens") is not None:
        params["max_tokens"] = config["max_tokens"]
    
    # Override with any provided kwargs
    params.update(kwargs)
    
    # Safety check: remove conflicting parameters
    if "max_tokens" in params and "max_completion_tokens" in params:
        # Keep max_completion_tokens, remove max_tokens
        del params["max_tokens"]
    
    return params

def estimate_tokens(text: str, chars_per_token: float = 4.0) -> int:
    """
    Estimate token count for text.
    
    Args:
        text: Input text
        chars_per_token: Average characters per token
        
    Returns:
        Estimated token count
    """
    return int(len(text) / chars_per_token)

def log_token_usage(
    model: str,
    input_text: str,
    output_text: str,
    actual_tokens: Optional[Dict[str, int]] = None
) -> None:
    """
    Log token usage for monitoring.
    
    Args:
        model: Model name
        input_text: Input prompt
        output_text: Model response
        actual_tokens: Actual token counts if available
    """
    estimated_input = estimate_tokens(input_text)
    estimated_output = estimate_tokens(output_text)
    
    print(f"\n📊 Token Usage ({model}):")
    print(f"   Input:  ~{estimated_input:,} tokens")
    print(f"   Output: ~{estimated_output:,} tokens")
    print(f"   Total:  ~{estimated_input + estimated_output:,} tokens")
    
    if actual_tokens:
        print(f"   Actual: {actual_tokens}")

# Export main configurations
__all__ = [
    'UNLIMITED_OUTPUT_CONFIG',
    'RETRY_CONFIG',
    'CHUNKING_CONFIG',
    'VALIDATION_CONFIG',
    'get_model_config',
    'get_api_params',
    'estimate_tokens',
    'log_token_usage',
]
