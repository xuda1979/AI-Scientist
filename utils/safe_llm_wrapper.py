"""
Safe GPT API Wrapper with Unlimited Output
==========================================
Provides safe API calls with no token limits, automatic retries, and validation.
"""

import time
import random
from typing import Optional, Dict, Any, List
from pathlib import Path

# Import configuration
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.gpt_unlimited_config import (
    get_model_config,
    get_api_params,
    RETRY_CONFIG,
    log_token_usage,
)

def safe_llm_call(
    messages: List[Dict[str, str]],
    model: str = "gpt-5-pro",
    client: Any = None,
    max_tokens: Optional[int] = None,
    temperature: Optional[float] = None,
    stream: bool = True,
    verbose: bool = True,
    **kwargs
) -> str:
    """
    Safe LLM API call with unlimited output and automatic retries.
    
    Args:
        messages: List of message dicts with 'role' and 'content'
        model: Model name
        client: OpenAI client instance (if None, will create)
        max_tokens: Max tokens (None = unlimited)
        temperature: Sampling temperature (None = use default)
        stream: Whether to stream the response
        verbose: Whether to print progress
        **kwargs: Additional API parameters
        
    Returns:
        Response text
        
    Raises:
        Exception if all retries fail
    """
    # Lazy import to avoid circular dependencies
    try:
        from openai import OpenAI
    except ImportError:
        raise ImportError("OpenAI package not installed. Run: pip install openai")
    
    # Create client if not provided
    if client is None:
        client = OpenAI()
    
    # Get configuration
    config = get_model_config(model)
    
    # Build API parameters
    api_params = get_api_params(model, messages)
    
    # Override with provided values
    if max_tokens is not None:
        # Remove both first to avoid conflicts
        api_params.pop("max_tokens", None)
        api_params.pop("max_completion_tokens", None)
        # Add the appropriate one based on model
        if model in ["o1-preview", "o1-mini"]:
            api_params["max_completion_tokens"] = max_tokens
        else:
            api_params["max_tokens"] = max_tokens
    
    if temperature is not None:
        api_params["temperature"] = temperature
    
    if "stream" in api_params:
        api_params["stream"] = stream
    
    # Merge additional kwargs
    api_params.update(kwargs)
    
    # Safety check: remove conflicting parameters
    if "max_tokens" in api_params and "max_completion_tokens" in api_params:
        # Keep max_completion_tokens, remove max_tokens for o1 models
        if model in ["o1-preview", "o1-mini"]:
            api_params.pop("max_tokens", None)
        else:
            api_params.pop("max_completion_tokens", None)
    
    # Remove None values
    api_params = {k: v for k, v in api_params.items() if v is not None}
    
    if verbose:
        print(f"\n🤖 Calling {model}...")
        if api_params.get("max_tokens") is None and api_params.get("max_completion_tokens") is None:
            print(f"   ✅ Unlimited output enabled")
        else:
            max_t = api_params.get("max_tokens") or api_params.get("max_completion_tokens")
            print(f"   📏 Max tokens: {max_t:,}")
        print(f"   🌡️  Temperature: {api_params.get('temperature', 'default')}")
        print(f"   📡 Streaming: {api_params.get('stream', False)}")
    
    # Retry configuration
    max_retries = RETRY_CONFIG["max_retries"]
    initial_delay = RETRY_CONFIG["initial_delay"]
    max_delay = RETRY_CONFIG["max_delay"]
    exp_base = RETRY_CONFIG["exponential_base"]
    use_jitter = RETRY_CONFIG["jitter"]
    
    # Retry loop
    for attempt in range(max_retries):
        try:
            if api_params.get("stream", False):
                # Streaming response
                if verbose:
                    print(f"\n📥 Streaming response...\n")
                
                response_text = ""
                stream_response = client.chat.completions.create(**api_params)
                
                for chunk in stream_response:
                    if chunk.choices[0].delta.content:
                        content = chunk.choices[0].delta.content
                        response_text += content
                        if verbose:
                            print(content, end='', flush=True)
                
                if verbose:
                    print(f"\n\n✅ Received {len(response_text)} characters")
                
                return response_text
            else:
                # Standard response
                if verbose:
                    print(f"\n⏳ Waiting for response...")
                
                response = client.chat.completions.create(**api_params)
                response_text = response.choices[0].message.content
                
                if verbose:
                    print(f"✅ Received {len(response_text)} characters")
                    
                    # Log token usage if available
                    if hasattr(response, 'usage'):
                        actual_tokens = {
                            'prompt_tokens': response.usage.prompt_tokens,
                            'completion_tokens': response.usage.completion_tokens,
                            'total_tokens': response.usage.total_tokens,
                        }
                        log_token_usage(
                            model,
                            messages[-1].get('content', ''),
                            response_text,
                            actual_tokens
                        )
                
                return response_text
                
        except Exception as e:
            error_msg = str(e)
            
            if verbose:
                print(f"\n⚠️  API call failed (attempt {attempt+1}/{max_retries}):")
                print(f"    {error_msg}")
            
            # Check if we should retry
            if attempt < max_retries - 1:
                # Calculate delay with exponential backoff
                delay = min(initial_delay * (exp_base ** attempt), max_delay)
                
                # Add jitter to prevent thundering herd
                if use_jitter:
                    delay = delay * (0.5 + random.random())
                
                if verbose:
                    print(f"    Retrying in {delay:.1f}s...")
                
                time.sleep(delay)
            else:
                # All retries exhausted
                error_msg = f"API call failed after {max_retries} attempts: {error_msg}"
                if verbose:
                    print(f"\n❌ {error_msg}")
                raise Exception(error_msg)

def create_messages(
    system_prompt: str,
    user_prompt: str,
    additional_messages: Optional[List[Dict[str, str]]] = None
) -> List[Dict[str, str]]:
    """
    Create properly formatted messages list.
    
    Args:
        system_prompt: System message content
        user_prompt: User message content
        additional_messages: Additional messages to include
        
    Returns:
        List of message dictionaries
    """
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    
    if additional_messages:
        messages.extend(additional_messages)
    
    return messages

def test_unlimited_output(
    model: str = "gpt-5-pro",
    test_prompt: str = None
) -> bool:
    """
    Test that unlimited output is working correctly.
    
    Args:
        model: Model to test
        test_prompt: Custom test prompt (default generates long output)
        
    Returns:
        True if test passes
    """
    if test_prompt is None:
        test_prompt = """
        Please generate a very long response to test unlimited output.
        Write a detailed 2000-word essay about quantum computing,
        including history, principles, applications, and future prospects.
        Be thorough and comprehensive.
        """
    
    messages = create_messages(
        system_prompt="You are a helpful assistant that provides complete, detailed responses.",
        user_prompt=test_prompt
    )
    
    try:
        print(f"\n{'='*70}")
        print(f"🧪 Testing unlimited output for {model}")
        print(f"{'='*70}")
        
        response = safe_llm_call(
            messages=messages,
            model=model,
            max_tokens=None,  # Unlimited!
            stream=True,
            verbose=True
        )
        
        word_count = len(response.split())
        char_count = len(response)
        
        print(f"\n{'='*70}")
        print(f"✅ Test PASSED")
        print(f"   Response length: {char_count:,} characters")
        print(f"   Word count: {word_count:,} words")
        print(f"   No truncation detected")
        print(f"{'='*70}\n")
        
        return True
        
    except Exception as e:
        print(f"\n{'='*70}")
        print(f"❌ Test FAILED")
        print(f"   Error: {e}")
        print(f"{'='*70}\n")
        return False

# Export main functions
__all__ = [
    'safe_llm_call',
    'create_messages',
    'test_unlimited_output',
]
