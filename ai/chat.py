#!/usr/bin/env python3
"""
AI chat interface and model management for the sciresearch workflow.
"""
from __future__ import annotations
import base64
import json
import os
import time
import urllib.request
import urllib.parse
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Google AI SDK
try:
    import google.generativeai as genai
    GOOGLE_AI_AVAILABLE = True
except ImportError:
    GOOGLE_AI_AVAILABLE = False


def _classify_error(error: Exception) -> Tuple[str, Optional[int]]:
    """Classify errors for retry logic."""
    error_str = str(error).lower()
    
    # Rate limit errors
    if any(term in error_str for term in ['rate limit', 'quota', 'too many requests', '429']):
        return "rate_limit", 60
    
    # Temporary API issues
    if any(term in error_str for term in ['502', '503', '504', 'server error', 'timeout']):
        return "temporary", 30
    
    # Permanent errors (don't retry)
    if any(term in error_str for term in ['401', '403', 'unauthorized', 'forbidden', 'invalid api key']):
        return "permanent", None
    
    # Default: temporary with short wait
    return "temporary", 10


def _model_supports_vision(model: str) -> bool:
    """Check if a model supports vision/image input."""
    vision_models = {
        'gpt-4-vision-preview', 'gpt-4-turbo', 'gpt-4o', 'gpt-4o-mini',
        'claude-3-opus', 'claude-3-sonnet', 'claude-3-haiku',
        'gemini-pro-vision', 'gemini-1.5-pro', 'gemini-1.5-flash'
    }
    return any(vm in model.lower() for vm in vision_models)


def _offline_response(prompt_type: str) -> str:
    """Provide offline fallback response."""
    return f"OFFLINE MODE: Unable to connect to AI service for {prompt_type} task. Please check your internet connection and API keys."


def _openai_chat(messages: List[Dict[str, str]], model: str, request_timeout: Optional[int] = None, 
                prompt_type: str = "general", fallback_models: Optional[List[str]] = None, 
                pdf_path: Optional[Path] = None) -> str:
    """Chat with OpenAI models with vision support."""
    import openai
    
    client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    
    # Handle PDF attachment for vision-capable models
    if pdf_path and _model_supports_vision(model):
        try:
            with open(pdf_path, "rb") as pdf_file:
                base64_pdf = base64.b64encode(pdf_file.read()).decode('utf-8')
            
            # Modify the last user message to include the PDF
            if messages and messages[-1]["role"] == "user":
                messages[-1]["content"] = [
                    {"type": "text", "text": messages[-1]["content"]},
                    {"type": "image_url", "image_url": {"url": f"data:application/pdf;base64,{base64_pdf}"}}
                ]
        except Exception as e:
            print(f"Warning: Failed to attach PDF to vision model: {e}")
    
    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=1.0,
            timeout=request_timeout or 3600
        )
        return response.choices[0].message.content
    
    except Exception as e:
        error_type, wait_time = _classify_error(e)
        
        if error_type == "rate_limit" and wait_time:
            print(f"Rate limit hit. Waiting {wait_time} seconds...")
            time.sleep(wait_time)
            # Retry once after rate limit
            try:
                response = client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=1.0,
                    timeout=request_timeout or 3600
                )
                return response.choices[0].message.content
            except Exception as retry_e:
                print(f"Retry failed: {retry_e}")
                raise retry_e
        
        print(f"OpenAI API error with {model}: {e}")
        raise e


def _google_chat(messages: List[Dict[str, str]], model: str, request_timeout: Optional[int] = None,
                prompt_type: str = "general", fallback_models: Optional[List[str]] = None,
                pdf_path: Optional[Path] = None) -> str:
    """Chat with Google AI models."""
    if not GOOGLE_AI_AVAILABLE:
        raise ImportError("Google AI SDK not available")
    
    genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
    
    try:
        # Convert OpenAI format to Google format
        prompt_text = ""
        for msg in messages:
            if msg["role"] == "system":
                prompt_text += f"System: {msg['content']}\n\n"
            elif msg["role"] == "user":
                prompt_text += f"User: {msg['content']}\n\n"
            elif msg["role"] == "assistant":
                prompt_text += f"Assistant: {msg['content']}\n\n"
        
        prompt_text += "Assistant: "
        
        model_instance = genai.GenerativeModel(model)
        response = model_instance.generate_content(
            prompt_text,
            generation_config=genai.types.GenerationConfig(
                temperature=1.0,
            )
        )
        
        return response.text
    
    except Exception as e:
        print(f"Google AI error with {model}: {e}")
        raise e


def _try_openai_model(messages: List[Dict[str, str]], model: str, temp: float, 
                     request_timeout: int, prompt_type: str, pdf_path: Optional[Path] = None,
                     max_retries: int = 3) -> str:
    """Try OpenAI model with retries and error handling."""
    import logging
    logger = logging.getLogger('sciresearch_workflow')
    
    for attempt in range(max_retries):
        try:
            logger.info(f"Making API call to {model} for {prompt_type}")
            print(f"Making API call to {model} for {prompt_type}")
            
            # Log message details
            total_chars = sum(len(str(msg.get('content', ''))) for msg in messages)
            logger.info(f"Messages: {len(messages)} total, {total_chars:,} characters")
            print(f"Messages: {len(messages)} total, {total_chars:,} characters")
            
            # Log PDF attachment info
            if pdf_path and pdf_path.exists():
                file_size = pdf_path.stat().st_size
                logger.info(f"PDF file attached: {pdf_path.name} ({file_size:,} bytes)")
                print(f"PDF file attached: {pdf_path.name} ({file_size:,} bytes)")
            else:
                logger.info("No PDF file attached (PDF review disabled)")
                print("No PDF file attached (PDF review disabled)")
            
            # Log individual message lengths for debugging
            for i, msg in enumerate(messages, 1):
                content_len = len(str(msg.get('content', '')))
                logger.info(f"Message {i} ({msg.get('role', 'unknown')}): {content_len:,} chars")
            
            logger.info(f"Making API call to {model} for {prompt_type}")
            print(f"[API] Making API call to {model} for {prompt_type}...")
            print(f"Sending request with temperature={temp}, timeout={request_timeout}s (attempt {attempt + 1}/{max_retries})...")
            
            result = _openai_chat(messages, model, request_timeout, prompt_type, pdf_path=pdf_path)
            
            logger.info(f"API call successful for {model} ({prompt_type}) - response: {len(result):,} chars")
            print("INFO: API call successful.")
            print(f"✓ API call successful for {model} - response: {len(result):,} characters")
            
            return result
            
        except Exception as e:
            error_type, wait_time = _classify_error(e)
            logger.error(f"Attempt {attempt + 1} failed for {model}: {e}")
            
            if attempt < max_retries - 1:
                wait_time = wait_time or (2 ** attempt * 10)  # Exponential backoff
                logger.info(f"Retrying in {wait_time} seconds...")
                time.sleep(wait_time)
            else:
                logger.error(f"All {max_retries} attempts failed for {model}")
                raise e


def _universal_chat(messages: List[Dict[str, str]], model: str, request_timeout: Optional[int] = None,
                   prompt_type: str = "general", fallback_models: Optional[List[str]] = None,
                   pdf_path: Optional[Path] = None) -> str:
    """Universal chat interface that handles multiple AI providers."""
    request_timeout = request_timeout or 3600
    fallback_models = fallback_models or ["gpt-4o", "gpt-4"]
    
    # Try primary model
    try:
        if model.startswith("gpt") or model.startswith("o1"):
            return _try_openai_model(messages, model, 1.0, request_timeout, prompt_type, pdf_path)
        elif model.startswith("gemini"):
            return _google_chat(messages, model, request_timeout, prompt_type, fallback_models, pdf_path)
        else:
            # Default to OpenAI for unknown models
            return _try_openai_model(messages, model, 1.0, request_timeout, prompt_type, pdf_path)
    
    except Exception as primary_error:
        print(f"Primary model {model} failed: {primary_error}")
        
        # Try fallback models
        for fallback_model in fallback_models:
            if fallback_model == model:
                continue  # Skip if same as primary
                
            try:
                print(f"Trying fallback model: {fallback_model}")
                if fallback_model.startswith("gpt") or fallback_model.startswith("o1"):
                    return _try_openai_model(messages, fallback_model, 1.0, request_timeout, prompt_type, pdf_path)
                elif fallback_model.startswith("gemini"):
                    return _google_chat(messages, fallback_model, request_timeout, prompt_type, fallback_models, pdf_path)
            except Exception as fallback_error:
                print(f"Fallback model {fallback_model} failed: {fallback_error}")
                continue
        
        # All models failed
        print("All models failed, returning offline response")
        return _offline_response(prompt_type)
