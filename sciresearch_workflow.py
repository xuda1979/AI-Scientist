#!/usr/bin/env python3
"""
Extended workflow:
 - Enforce single paper.tex and simulation.py per project
 - Extract simulation code from LaTeX; run it; pass results to LLM during review/revision
 - Sanitize LaTeX to prevent overflow; compile-check; auto-fix on failure
"""
from __future__ import annotations
import argparse
import base64
import difflib
import json
import os
import re
import subprocess
import sys
import time
import urllib.request
import urllib.parse
import functools
import logging
import signal
import threading
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple

# Google AI SDK
try:
    import google.generativeai as genai
    GOOGLE_AI_AVAILABLE = True
except ImportError:
    GOOGLE_AI_AVAILABLE = False
    print("WARNING: Google AI SDK not installed. Run: pip install google-generativeai")

# Local helpers
from utils.sim_runner import ensure_single_tex_py, extract_simulation_from_tex

# Content protection system
from utils.content_protection import ContentProtector

# Shared workflow configuration
from core.config import WorkflowConfig

# Workflow step modules
from workflow_steps.initial_draft import generate_initial_draft
from workflow_steps.simulation import run_simulation_step
from workflow_steps.review_revision import run_review_revision_step

# Document type system
from document_types import DocumentType, get_document_template, infer_document_type, get_available_document_types
from document_prompts import DocumentPromptGenerator

# Quality enhancement system
from prompts.experimental_rigor_prompts import (
    detect_paper_type, 
    enhance_prompt_with_rigor,
    get_statistical_rigor_requirements,
    get_theoretical_rigor_requirements
)
from quality_enhancements.quality_validator import PaperQualityValidator

# Review-driven enhancements  
from prompts.review_driven_enhancements import (
    enhance_prompt_for_review_quality,
    detect_review_issues,
    get_review_driven_requirements
)

def timeout_input(prompt: str, timeout: int = 30, default: str = "") -> str:
    """
    Get user input with a timeout. If no input is received within the timeout,
    return the default value.
    
    Args:
        prompt: The prompt to display to the user
        timeout: Timeout in seconds (default: 30)
        default: Default value to return if timeout occurs
        
    Returns:
        User input or default value if timeout
    """
    import sys
    import select
    import os
    
    # On Windows, we need a different approach
    if os.name == 'nt':  # Windows
        import msvcrt
        import time
        
        print(f"{prompt}", end="", flush=True)
        if default:
            print(f" [default: {default}]", end="", flush=True)
        print(" ", end="", flush=True)
        
        start_time = time.time()
        input_chars = []
        
        while time.time() - start_time < timeout:
            if msvcrt.kbhit():
                char = msvcrt.getch()
                if char == b'\r':  # Enter key
                    print()  # New line
                    return ''.join(input_chars)
                elif char == b'\x08':  # Backspace
                    if input_chars:
                        input_chars.pop()
                        print('\b \b', end='', flush=True)
                else:
                    char_str = char.decode('utf-8', errors='ignore')
                    if char_str.isprintable():
                        input_chars.append(char_str)
                        print(char_str, end='', flush=True)
            time.sleep(0.1)
        
        print(f"\nTimeout reached. Using default: {default}")
        return default
    
    else:  # Unix/Linux/Mac
        print(f"{prompt}", end="", flush=True)
        if default:
            print(f" [default: {default}]", end="", flush=True)
        print(" ", end="", flush=True)
        
        ready, _, _ = select.select([sys.stdin], [], [], timeout)
        if ready:
            return sys.stdin.readline().strip()
        else:
            print(f"\nTimeout reached. Using default: {default}")
            return default

DEFAULT_MODEL = os.environ.get("SCI_MODEL", "gpt-5")

def setup_workflow_logging(log_level=logging.INFO, log_dir: Optional[Path] = None) -> logging.Logger:
    """Set up structured logging for the workflow."""
    if log_dir is None:
        log_dir = Path("logs")
    
    log_dir.mkdir(exist_ok=True)
    log_file = log_dir / f'workflow_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
    
    # Create logger
    logger = logging.getLogger('sciresearch_workflow')
    logger.setLevel(log_level)
    
    # Remove existing handlers
    logger.handlers.clear()
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # File handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(log_level)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    # Console handler (mirror file logging)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    return logger

# Global logger instance
logger = setup_workflow_logging()

class APIError(Exception):
    """Custom exception for API-related errors"""
    pass

class SecurityError(Exception):
    """Custom exception for security-related errors"""
    pass

class ResourceExhaustionError(Exception):
    """Custom exception for resource exhaustion"""
    pass

def _classify_error(error: Exception) -> Tuple[str, Optional[int]]:
    """Classify error and return error type and recommended wait time"""
    error_str = str(error).lower()
    
    if "rate_limit" in error_str or "429" in error_str:
        return "rate_limit", 60
    elif "timeout" in error_str:
        return "timeout", 10
    elif "content_policy" in error_str or "safety" in error_str:
        return "content_violation", None  # Don't retry
    elif "authentication" in error_str or "401" in error_str:
        return "auth_error", None
    elif "network" in error_str or "connection" in error_str:
        return "network", 30
    else:
        return "unknown", 30

def _openai_chat(messages: List[Dict[str, str]], model: str, request_timeout: Optional[int] = None, prompt_type: str = "general", fallback_models: Optional[List[str]] = None, pdf_path: Optional[Path] = None) -> str:
    """
    Enhanced chat wrapper with error classification, fallback models, intelligent retry, and PDF support.
    Includes retry logic for timeout errors and configurable temperature based on prompt type.
    
    Args:
        messages: List of chat messages
        model: AI model to use
        request_timeout: Request timeout in seconds
        prompt_type: Type of prompt for temperature selection
        fallback_models: List of fallback models if primary fails
        pdf_path: Optional PDF file to include in the request
    """
    logger.info(f"Making API call to {model} for {prompt_type}")
    print(f"[API] Making API call to {model} for {prompt_type}...")

    if os.getenv("OFFLINE_MODE") == "1":
        print("OFFLINE_MODE enabled - returning stub response without contacting API")
        return _offline_response(prompt_type)

    if False and pdf_path and pdf_path.exists():  # PDF upload disabled
        print(f"[PDF] Including PDF in request: {pdf_path.name}")
    
    # Set longer timeout for GPT-5
    if request_timeout is None:
        request_timeout = 3600 if "gpt-5" in model.lower() else 1800  # 1 hour for GPT-5, 30 min for others
    
    # Configure temperature based on prompt type and model
    if "gpt-5" in model.lower():
        temp = 1.0  # GPT-5 only supports temperature=1
    else:
        # Different temperatures for different prompt types
        temp_map = {
            "initial_draft": 0.7,    # Creative for initial drafting
            "review": 0.3,           # Conservative for reviewing
            "revise": 0.5,           # Balanced for revisions
            "editor": 0.2,           # Very conservative for decisions
            "simulation_fix": 0.4,   # Moderate for code fixes
            "general": 0.2           # Default conservative
        }
        temp = temp_map.get(prompt_type, 0.2)
    
    # Try primary model first
    try:
        result = _try_openai_model(messages, model, temp, request_timeout, prompt_type, pdf_path)
        logger.info(f"API call successful for {model} ({prompt_type}) - response: {len(result):,} chars")
        print(f"✓ API call successful for {model} - response: {len(result):,} characters")
        return result
    except Exception as primary_error:
        error_type, wait_time = _classify_error(primary_error)
        logger.warning(f"Primary model {model} failed", extra={'error': str(primary_error), 'error_type': error_type})
        
        # Don't retry for certain error types
        if wait_time is None:
            print(f"ERROR: Non-retryable error with {model}: {primary_error}")
            raise APIError(f"Primary model {model} failed with non-retryable error: {primary_error}")
        
        # Try fallback models if available
        if fallback_models:
            print(f"WARNING: Primary model {model} failed, trying fallback models...")
            for fallback_model in fallback_models:
                try:
                    print(f"INFO: Attempting fallback model: {fallback_model}")
                    return _try_openai_model(messages, fallback_model, temp, request_timeout, prompt_type, pdf_path)
                except Exception as fallback_error:
                    print(f"WARNING: Fallback model {fallback_model} also failed: {fallback_error}")
                    continue

        # If all models failed, raise the original error
        raise APIError(f"All models failed. Primary error: {primary_error}")


def _offline_response(prompt_type: str) -> str:
    """Return minimal placeholder responses when OFFLINE_MODE is enabled."""
    if prompt_type == "combined_review_edit_revise":
        return "## REVIEW\nOffline review skipped\n## REVISION DIFFS\n"
    return ""

def _try_openai_model(messages: List[Dict[str, str]], model: str, temp: float, request_timeout: int, prompt_type: str, pdf_path: Optional[Path] = None, max_retries: int = 3) -> str:
    """
    Try a specific OpenAI model with intelligent retry logic and optional PDF support.
    
    Args:
        messages: List of chat messages
        model: OpenAI model to use
        temp: Temperature for generation
        request_timeout: Request timeout in seconds
        prompt_type: Type of prompt
        pdf_path: Optional PDF file to include
        max_retries: Maximum number of retry attempts
    """
    
    for attempt in range(max_retries):
        try:
            # Newer SDK
            from openai import OpenAI
            client = OpenAI()
            
            # Process messages to include PDF if provided
            processed_messages = messages.copy()
            
            # Add PDF to the last user message if provided and model supports vision
            if False and pdf_path and pdf_path.exists() and _model_supports_vision(model):  # PDF upload disabled
                try:
                    # For now, we'll add a note about the PDF but not include the binary data
                    # OpenAI's vision models typically work better with images than PDFs
                    # In the future, this could be enhanced to convert PDF to images
                    
                    # Find the last user message and add PDF notice
                    for i in range(len(processed_messages) - 1, -1, -1):
                        if processed_messages[i]["role"] == "user":
                            # Add note about PDF availability
                            original_content = processed_messages[i]["content"]
                            processed_messages[i]["content"] = f"{original_content}\n\n**Note: A PDF version of the paper ({pdf_path.name}, {pdf_path.stat().st_size // 1024} KB) has been generated and is available for reference. Please provide feedback as if you can see the rendered document layout, figure placement, and visual formatting.**"
                            break
                    
                    print(f" PDF reference added to request: {pdf_path.name} ({pdf_path.stat().st_size // 1024} KB)")
                    
                except Exception as pdf_error:
                    print(f"WARNING: Failed to process PDF reference: {pdf_error}")
                    print("Continuing with text-only request...")
            
            elif False and pdf_path and pdf_path.exists() and not _model_supports_vision(model):  # PDF upload disabled
                print(f"INFO: Model {model} does not support vision input. Adding PDF reference note...")
                # Even for non-vision models, we can mention that a PDF was generated
                for i in range(len(processed_messages) - 1, -1, -1):
                    if processed_messages[i]["role"] == "user":
                        original_content = processed_messages[i]["content"]
                        processed_messages[i]["content"] = f"{original_content}\n\n**Note: A PDF version of the paper ({pdf_path.name}) has been generated successfully, indicating that the LaTeX compiles properly and produces a readable document.**"
                        break
            
            # Use configured temperature based on prompt type
            print(f"Sending request with temperature={temp}, timeout={request_timeout}s (attempt {attempt + 1}/{max_retries})...")
            
            resp = client.chat.completions.create(
                model=model, 
                messages=processed_messages, 
                temperature=temp, 
                timeout=request_timeout
            )
            print("INFO: API call successful.")
            return resp.choices[0].message.content
            
        except KeyboardInterrupt:
            print("ERROR: User interrupted the process.")
            raise
        except Exception as e:
            error_type, wait_time = _classify_error(e)
            print(f"WARNING: API Error (attempt {attempt + 1}): {e} (Type: {error_type})")
            
            # Don't retry for certain error types
            if wait_time is None:
                raise e
            
            # Only retry for retryable errors and if we have attempts left
            if attempt < max_retries - 1 and wait_time is not None:
                print(f"INFO: Retrying in {wait_time} seconds...")
                time.sleep(wait_time)
                continue
            else:
                raise e

def _model_supports_vision(model: str) -> bool:
    """Check if the given model supports vision/image inputs."""
    vision_models = [
        'gpt-4-vision', 'gpt-4-vision-preview', 'gpt-4-turbo', 'gpt-4-turbo-vision',
        'gpt-4o', 'gpt-4o-mini', 'gpt-5'  # Add more vision-capable models as they become available
    ]
    return any(vm in model.lower() for vm in vision_models)

def _validate_code_security(code: str) -> None:
    """Validate simulation code for security risks before execution."""
    dangerous_patterns = [
        (r'import\s+os\b', "OS module import detected"),
        (r'import\s+subprocess\b', "Subprocess module import detected"),
        (r'exec\s*\(', "exec() function call detected"),
        (r'eval\s*\(', "eval() function call detected"),
        (r'__import__\s*\(', "__import__() function call detected"),
        (r'open\s*\(["\'][^"\']*["\'][^)]*["\']w', "File writing operation detected"),
        (r'urllib\.request', "Network request detected"),
        (r'socket\.', "Socket operation detected"),
        (r'shutil\.rmtree', "Directory deletion detected"),
        (r'os\.system', "System command execution detected"),
        (r'os\.popen', "Process execution detected"),
    ]
    
    security_issues = []
    for pattern, description in dangerous_patterns:
        if re.search(pattern, code):
            security_issues.append(description)
    
    if security_issues:
        raise SecurityError(f"Security risks detected: {'; '.join(security_issues)}")

def _create_simulation_fixer(model: str, request_timeout: Optional[int] = None):
    """
    Returns a callable that can analyze simulation errors and decide what to do.
    """
    def _fix_simulation(code: str, stdout: str, stderr: str, return_code: int) -> Dict[str, str]:
        # Validate code security before processing
        try:
            _validate_code_security(code)
        except SecurityError as e:
            return {"action": "reject", "reason": str(e)}
        
        sys_prompt = (
            "You are a Python expert helping with simulation code. Given a Python script and its error output, "
            "decide what action to take. Respond with JSON only.\n\n"
            "Response format:\n"
            "- If the error is acceptable (e.g., encoding issues that don't affect results): {\"action\": \"accept\"}\n"
            "- If the code needs fixing: {\"action\": \"fix_code\", \"fixed_code\": \"<complete fixed code>\"}\n"
            "- If modules need installing: {\"action\": \"install_modules\", \"modules\": [\"module1\", \"module2\"]}\n"
            "- If the code is unsafe or malicious: {\"action\": \"reject\", \"reason\": \"<security reason>\"}\n\n"
            "SECURITY CONSTRAINTS:\n"
            "- Do not include any file system operations beyond reading data files\n"
            "- Do not include network operations\n"
            "- Do not include system command execution\n"
            "- Only allow safe computational libraries (numpy, matplotlib, scipy, pandas, etc.)\n\n"
            "Common fixes:\n"
            "- Unicode encoding errors: replace Greek letters with ASCII\n"
            "- Missing imports: add import statements\n"
            "- Module not found: suggest installation\n"
            "- Syntax errors: fix the syntax"
        )
        
        user = (
            f"Python script execution failed with return code {return_code}.\n\n"
            "=== CODE ===\n" + code + "\n\n"
            "=== STDOUT ===\n" + stdout + "\n\n"
            "=== STDERR ===\n" + stderr + "\n\n"
            "Analyze the error and respond with appropriate action as JSON."
        )
        
        try:
            fallback_models = ["gpt-4o", "gpt-4"] if "gpt-5" in model else ["gpt-3.5-turbo"]
            response = _universal_chat(
                [{"role": "system", "content": sys_prompt}, {"role": "user", "content": user}],
                model=model,
                request_timeout=request_timeout,
                prompt_type="simulation_fix",
                fallback_models=fallback_models
            )
            # Try to parse JSON response
            parsed_response = json.loads(response.strip())
            
            # Validate response structure
            if "action" not in parsed_response:
                print(f"LLM fixer response missing 'action' field: {response[:100]}")
                return {"action": "accept", "reason": "Invalid response format"}
            
            # Validate fixed code if provided
            if parsed_response.get("action") == "fix_code" and "fixed_code" in parsed_response:
                try:
                    _validate_code_security(parsed_response["fixed_code"])
                except SecurityError as e:
                    return {"action": "reject", "reason": f"Fixed code still unsafe: {e}"}
            
            # Classify error types for better handling
            if return_code != 0:
                if "import" in stderr.lower() or "modulenotfounderror" in stderr.lower():
                    # Missing module error - suggest installation
                    if parsed_response["action"] == "accept":
                        parsed_response["action"] = "fix"
                        parsed_response["reason"] = "Missing module detected - installation may be needed"
                elif "syntaxerror" in stderr.lower():
                    # Syntax error - should definitely try to fix
                    if parsed_response["action"] == "accept":
                        parsed_response["action"] = "fix"
                        parsed_response["reason"] = "Syntax error detected - code needs fixing"
            
            return parsed_response
            
        except json.JSONDecodeError as e:
            print(f"LLM fixer JSON parsing error: {e}")
            print(f"Response was: {response[:200]}...")
            return {"action": "accept", "reason": "JSON parsing failed"}
        except Exception as e:
            print(f"LLM fixer error: {e}")
            return {"action": "accept", "reason": f"Unexpected error: {str(e)}"}
    
    return _fix_simulation

def _select_best_revision_candidate_with_llm(
    candidates: List[str], 
    original_content: str, 
    review_text: str, 
    model: str, 
    request_timeout: int, 
    config: Any,
    pdf_path: Optional[Path] = None,  # Keep parameter for compatibility but not used
    project_dir: Optional[Path] = None,
) -> str:
    """
    Use an LLM to evaluate and select the best revision candidate from multiple options.
    
    Args:
        candidates: List of revision candidates to evaluate
        original_content: Original paper content for context
        review_text: Review feedback for context
        model: AI model to use for evaluation
        request_timeout: Request timeout
        config: Configuration object
        pdf_path: Not used (kept for compatibility) - PDFs skipped due to size
        project_dir: Project directory to read local files for context
    
    Returns:
        LLM response indicating which candidate is best and why
    """
    # Collect lightweight attachments (in-memory, truncated) for evaluation context
    # Note: PDF files are skipped due to size constraints
    attachments_note = ""
    try:
        if project_dir is not None:
            # Include main paper file
            paper_file = project_dir / "paper.tex"
            if paper_file.exists():
                paper_text = paper_file.read_text(encoding="utf-8", errors="ignore")
                attach_preview = (paper_text[:1500] + ("\n...[CONTENT TRUNCATED]...\n" if len(paper_text) > 2200 else "") + paper_text[-700:]) if len(paper_text) > 2200 else paper_text
                attachments_note += f"\nATTACHMENT: paper.tex (preview)\n{attach_preview}\n"
            
            # Include simulation code
            sim_file = project_dir / "simulation.py"
            if sim_file.exists():
                sim_text = sim_file.read_text(encoding="utf-8", errors="ignore")
                sim_preview = sim_text[:1200] + ("\n...[CONTENT TRUNCATED]...\n" if len(sim_text) > 1600 else "") + (sim_text[-400:] if len(sim_text) > 1600 else "")
                attachments_note += f"\nATTACHMENT: simulation.py (preview)\n{sim_preview}\n"
            
            # Include simulation output
            out_file = project_dir / "simulation_output.txt"
            if out_file.exists():
                out_text = out_file.read_text(encoding="utf-8", errors="ignore")
                out_preview = out_text[:1000] + ("\n...[CONTENT TRUNCATED]...\n" if len(out_text) > 1400 else "") + (out_text[-300:] if len(out_text) > 1400 else "")
                attachments_note += f"\nATTACHMENT: simulation_output.txt (preview)\n{out_preview}\n"
            
            # Include results summary CSV
            results_csv_file = project_dir / "results_summary.csv"
            if results_csv_file.exists():
                results_csv_text = results_csv_file.read_text(encoding="utf-8", errors="ignore")
                results_csv_preview = results_csv_text[:800] + ("\n...[CONTENT TRUNCATED]..." if len(results_csv_text) > 800 else "")
                attachments_note += f"\nATTACHMENT: results_summary.csv (preview)\n{results_csv_preview}\n"
            
            # Include bibliography if separate
            refs_file = project_dir / "refs.bib"
            if refs_file.exists():
                refs_text = refs_file.read_text(encoding="utf-8", errors="ignore")
                refs_preview = refs_text[:600] + ("\n...[CONTENT TRUNCATED]..." if len(refs_text) > 600 else "")
                attachments_note += f"\nATTACHMENT: refs.bib (preview)\n{refs_preview}\n"
                
    except Exception as _e:
        print(f"WARNING: Failed to attach project files for evaluation: {_e}")

    # Prepare candidates summary for evaluation
    candidates_summary = ""
    for i, candidate in enumerate(candidates):
        candidates_summary += f"\n{'='*50}\nREVISION CANDIDATE {i+1}:\n{'='*50}\n"
        # Show first 1500 chars and last 500 chars to capture key changes
        if len(candidate) > 2000:
            candidates_summary += candidate[:1500] + "\n...[CONTENT TRUNCATED]...\n" + candidate[-500:]
        else:
            candidates_summary += candidate
        candidates_summary += "\n"
    
    evaluation_prompt = [
        {
            "role": "system",
            "content": f"""You are an expert academic reviewer tasked with selecting the best paper revision from multiple candidates.

Your task is to:
1. Analyze each revision candidate against the original paper and review feedback
2. Evaluate improvements in: technical rigor, clarity, completeness, addressing review concerns
3. Select the single best revision
4. Provide clear reasoning for your choice

ORIGINAL PAPER LENGTH: {len(original_content)} characters

REVIEW FEEDBACK:
{review_text[:800]}...

EVALUATION CRITERIA:
- How well does it address the review feedback?
- Technical accuracy and depth improvements
- Clarity and readability enhancements
- Completeness of methodology and results
- Quality of new content additions
- Overall scientific contribution improvement

OUTPUT FORMAT:
REASONING: [Detailed analysis comparing candidates and explaining your choice]

SELECTED: [NUMBER] (just the number, e.g., "2")"""
        },
        {
            "role": "user", 
            "content": f"""Please evaluate these {len(candidates)} revision candidates and select the best one that most effectively improves the original paper:

{candidates_summary}

In addition to the candidate texts above, use these attachments for context:
{attachments_note if attachments_note else '(no local attachments found)'}

Note: PDF attachment is skipped due to size constraints, but assume the LaTeX compiles correctly if mentioned.

Which revision candidate provides the highest quality improvements? Focus on scientific rigor, addressing review concerns, and overall enhancement of the paper."""
        }
    ]
    
    try:
        response = _universal_chat(
            evaluation_prompt, 
            model=model, 
            request_timeout=request_timeout, 
            prompt_type="revision_candidate_evaluation",
            fallback_models=config.fallback_models,
            pdf_path=None  # Skip PDF due to size constraints
        )
        return response
    except Exception as e:
        print(f"ERROR: LLM revision evaluation failed: {e}")
        return "SELECTED: 1\nREASONING: Evaluation failed, defaulting to first candidate."

def _select_best_candidate_with_llm(
    candidates: List[str], 
    original_content: str, 
    sim_summary: str, 
    model: str, 
    request_timeout: int, 
    config: Any,
    pdf_path: Optional[Path] = None,  # Keep parameter for compatibility but not used
    project_dir: Optional[Path] = None,
) -> str:
    """
    Use an LLM to evaluate and select the best candidate from multiple options.
    
    Args:
        candidates: List of candidate responses to evaluate
        original_content: Original paper content for context
        sim_summary: Simulation summary for context
        model: AI model to use for evaluation
        request_timeout: Request timeout
        config: Configuration object
        pdf_path: Not used (kept for compatibility) - PDFs skipped due to size
        project_dir: Project directory to read local files for context
    
    Returns:
        LLM response indicating which candidate is best and why
    """
    # Collect lightweight attachments (in-memory, truncated) for evaluation context
    # Note: PDF files are skipped due to size constraints
    attachments_note = ""
    try:
        if project_dir is not None:
            # Include main paper file
            paper_file = project_dir / "paper.tex"
            if paper_file.exists():
                paper_text = paper_file.read_text(encoding="utf-8", errors="ignore")
                attach_preview = (paper_text[:1200] + ("\n...[CONTENT TRUNCATED]...\n" if len(paper_text) > 1800 else "") + paper_text[-400:]) if len(paper_text) > 1800 else paper_text
                attachments_note += f"\nATTACHMENT: paper.tex (preview)\n{attach_preview}\n"
            
            # Include simulation code
            sim_file = project_dir / "simulation.py"
            if sim_file.exists():
                sim_text = sim_file.read_text(encoding="utf-8", errors="ignore")
                sim_preview = sim_text[:800] + ("\n...[CONTENT TRUNCATED]...\n" if len(sim_text) > 1200 else "") + (sim_text[-300:] if len(sim_text) > 1200 else "")
                attachments_note += f"\nATTACHMENT: simulation.py (preview)\n{sim_preview}\n"
            
            # Include simulation output
            sim_output_file = project_dir / "simulation_output.txt"
            if sim_output_file.exists():
                sim_output_text = sim_output_file.read_text(encoding="utf-8", errors="ignore")
                sim_output_preview = sim_output_text[:1000] + ("\n...[CONTENT TRUNCATED]..." if len(sim_output_text) > 1000 else "")
                attachments_note += f"\nATTACHMENT: simulation_output.txt (preview)\n{sim_output_preview}\n"
            
            # Include results summary CSV
            results_csv_file = project_dir / "results_summary.csv"
            if results_csv_file.exists():
                results_csv_text = results_csv_file.read_text(encoding="utf-8", errors="ignore")
                results_csv_preview = results_csv_text[:800] + ("\n...[CONTENT TRUNCATED]..." if len(results_csv_text) > 800 else "")
                attachments_note += f"\nATTACHMENT: results_summary.csv (preview)\n{results_csv_preview}\n"
            
            # Include bibliography if separate
            refs_file = project_dir / "refs.bib"
            if refs_file.exists():
                refs_text = refs_file.read_text(encoding="utf-8", errors="ignore")
                refs_preview = refs_text[:600] + ("\n...[CONTENT TRUNCATED]..." if len(refs_text) > 600 else "")
                attachments_note += f"\nATTACHMENT: refs.bib (preview)\n{refs_preview}\n"
                
    except Exception as _e:
        print(f"WARNING: Failed to attach project files for evaluation: {_e}")

    # Prepare candidates summary for evaluation
    candidates_summary = ""
    for i, candidate in enumerate(candidates):
        candidates_summary += f"\n{'='*50}\nCANDIDATE {i+1}:\n{'='*50}\n"
        candidates_summary += candidate[:2000] + ("...[TRUNCATED]" if len(candidate) > 2000 else "")
        candidates_summary += "\n"
    
    evaluation_prompt = [
        {
            "role": "system",
            "content": f"""You are an expert academic reviewer tasked with selecting the best revision candidate from multiple options. 

Your task is to:
1. Carefully analyze each candidate's proposed changes
2. Evaluate quality based on: scientific rigor, clarity, completeness, technical depth, and addressing of quality issues
3. Select the single best candidate
4. Provide clear reasoning for your choice

ORIGINAL PAPER CONTEXT:
- Length: {len(original_content)} characters
- Simulation summary: {sim_summary[:500]}...

EVALUATION CRITERIA:
- Technical accuracy and rigor
- Clarity of presentation
- Completeness of methodology and results
- Quality of experimental validation
- Addressing of identified issues
- Overall contribution to scientific knowledge

OUTPUT FORMAT:
REASONING: [Explain your analysis of each candidate and comparison]

SELECTED: [NUMBER] (just the number, e.g., "2")"""
        },
        {
            "role": "user", 
            "content": f"""Please evaluate these {len(candidates)} revision candidates and select the best one:

{candidates_summary}

In addition to the candidate texts above, use these attachments for context:
{attachments_note if attachments_note else '(no local attachments found)'}

Note: PDF attachment is skipped due to size constraints, but assume the LaTeX compiles correctly if mentioned.

Which candidate provides the highest quality revision? Consider scientific rigor, clarity, completeness, and overall improvement over the original."""
        }
    ]
    
    try:
        response = _universal_chat(
            evaluation_prompt, 
            model=model, 
            request_timeout=request_timeout, 
            prompt_type="candidate_evaluation",
            fallback_models=config.fallback_models,
            pdf_path=None  # Skip PDF due to size constraints
        )
        return response
    except Exception as e:
        print(f"ERROR: LLM evaluation failed: {e}")
        return "SELECTED: 1\nREASONING: Evaluation failed, defaulting to first candidate."

def _save_candidate_diff(old_content: str, new_content: str, candidate_num: int, prefix: str = "candidate"):
    """
    Generate and display a git-style diff for candidate comparison.
    
    Args:
        old_content: Original content
        new_content: Candidate content
        candidate_num: Candidate number
        prefix: Prefix for display
    """
    try:
        # Generate git-style diff
        diff_lines = list(difflib.unified_diff(
            old_content.splitlines(keepends=True),
            new_content.splitlines(keepends=True),
            fromfile=f"a/original.tex",
            tofile=f"b/{prefix}_{candidate_num}.tex",
            lineterm=""
        ))
        
        if diff_lines:
            print(f"\n{'='*60}")
            print(f"CANDIDATE {candidate_num} DIFF")
            print(f"{'='*60}")
            
            # Count changes
            adds = sum(1 for line in diff_lines if line.startswith('+') and not line.startswith('+++'))
            dels = sum(1 for line in diff_lines if line.startswith('-') and not line.startswith('---'))
            print(f"Changes: +{adds} -{dels} lines")
            
            # DEBUG: Show content length comparison
            print(f"DEBUG: Content comparison:")
            print(f"  Original: {len(old_content)} chars")
            print(f"  New: {len(new_content)} chars")
            print(f"  Diff lines: {len(diff_lines)}")
            
            # Show first 25 lines of diff
            diff_preview = diff_lines[:25]
            for line in diff_preview:
                if not line.startswith('---') and not line.startswith('+++'):
                    print(line, end='')
            
            if len(diff_lines) > 25:
                print(f"\n... ({len(diff_lines) - 25} more lines)")
            
            print(f"{'='*60}\n")
        else:
            print(f"INFO: No changes detected in candidate {candidate_num}")
            # DEBUG: Show why no diff was detected
            print(f"DEBUG: No diff details:")
            print(f"  Original content length: {len(old_content)}")
            print(f"  New content length: {len(new_content)}")
            print(f"  Contents identical: {old_content == new_content}")
            if len(old_content) == len(new_content) and old_content != new_content:
                print(f"  Same length but different - showing first difference:")
                for i, (orig_char, new_char) in enumerate(zip(old_content, new_content)):
                    if orig_char != new_char:
                        print(f"    First diff at position {i}: '{orig_char}' -> '{new_char}'")
                        break
            
    except Exception as e:
        print(f"WARNING: Failed to generate candidate {candidate_num} diff: {e}")

def _save_iteration_diff(old_content: str, new_content: str, output_dir: Path, iteration: int, filename: str = "paper.tex"):
    """
    Generate and display a git-style diff between old and new file content to terminal only.
    
    Args:
        old_content: Content before revision
        new_content: Content after revision  
        output_dir: Directory (unused - no files saved)
        iteration: Current iteration number
        filename: Name of the file being diffed (default: paper.tex)
    """
    try:
        # Generate git-style diff
        diff_lines = list(difflib.unified_diff(
            old_content.splitlines(keepends=True),
            new_content.splitlines(keepends=True),
            fromfile=f"a/{filename}",
            tofile=f"b/{filename}",
            lineterm=""
        ))
        
        if diff_lines:
            print(f"\n{'='*80}")
            print(f"GIT DIFF FOR ITERATION {iteration} - {filename}")
            print(f"{'='*80}")
            
            # Add git diff header
            print(f"diff --git a/{filename} b/{filename}")
            print(f"index 1234567..abcdefg 100644")
            print(f"--- a/{filename}")
            print(f"+++ b/{filename}")
            
            # Print the actual diff content (skip the file headers from difflib)
            in_header = True
            for line in diff_lines:
                if line.startswith('@@') and in_header:
                    in_header = False
                if not in_header:
                    print(line, end='')
            
            print(f"{'='*80}\n")
        else:
            # Even if no diff, show content length comparison
            if len(old_content) != len(new_content):
                print(f"Subtle changes detected in iteration {iteration}: {len(old_content)} -> {len(new_content)} chars")
            else:
                print(f"No changes detected in iteration {iteration}")
            
            # Show first 200 chars of both for manual comparison
            print(f"DEBUG: Content comparison (first 200 chars):")
            print(f"   Before: {old_content[:200]}...")
            print(f"   After:  {new_content[:200]}...")
            
    except Exception as e:
        print(f"WARNING: Failed to generate diff for iteration {iteration}: {e}")

def _universal_chat(messages: List[Dict[str, str]], model: str, request_timeout: Optional[int] = None, prompt_type: str = "general", fallback_models: Optional[List[str]] = None, pdf_path: Optional[Path] = None) -> str:
    """
    Universal chat function that automatically detects whether to use OpenAI or Google AI
    based on the model name and routes the request accordingly.
    
    Args:
        messages: List of chat messages
        model: AI model to use
        request_timeout: Request timeout in seconds
        prompt_type: Type of prompt for temperature selection
        fallback_models: List of fallback models if primary fails
        pdf_path: Optional PDF file to include in the request
    """
    # Log detailed information about what's being sent to the model
    logger = logging.getLogger(__name__)
    
    # Log basic call information
    logger.info(f"Making API call to {model} for {prompt_type}")
    print(f"Making API call to {model} for {prompt_type}")
    
    # Log message information
    total_chars = sum(len(msg.get('content', '')) for msg in messages)
    logger.info(f"Messages: {len(messages)} total, {total_chars:,} characters")
    print(f"Messages: {len(messages)} total, {total_chars:,} characters")
    
    # Log file information
    files_info = []
    if pdf_path and pdf_path.exists():
        file_size = pdf_path.stat().st_size
        files_info.append(f"PDF: {pdf_path.name} ({file_size:,} bytes)")
        logger.info(f"PDF file attached: {pdf_path.name} ({file_size:,} bytes)")
        print(f"PDF file attached: {pdf_path.name} ({file_size:,} bytes)")
    else:
        logger.info("No PDF file attached (PDF review disabled)")
        print("No PDF file attached (PDF review disabled)")
    
    # Log prompt content summary
    for i, msg in enumerate(messages):
        content = msg.get('content', '')
        role = msg.get('role', 'unknown')
        logger.info(f"Message {i+1} ({role}): {len(content):,} chars")
        
        # Log key sections in the content
        if 'CURRENT PAPER (LATEX)' in content:
            print("  • LaTeX source code included")
        if 'SIMULATION CODE & OUTPUTS' in content:
            print("  • Simulation results included")
        if 'DETECTED QUALITY ISSUES' in content:
            print("  • Quality issues list included")
        if 'LATEX COMPILATION ERRORS' in content:
            print("  • LaTeX compilation errors included")
        if 'ALL PROJECT FILES' in content:
            print("  • Project context files included")
    
    # Detect provider based on model name
    if model.startswith(('gemini', 'models/gemini')):
        # Google AI model
        return _google_chat(messages, model, request_timeout, prompt_type, fallback_models, pdf_path)
    else:
        # OpenAI model
        return _openai_chat(messages, model, request_timeout, prompt_type, fallback_models, pdf_path)

def _google_chat(messages: List[Dict[str, str]], model: str, request_timeout: Optional[int] = None, prompt_type: str = "general", fallback_models: Optional[List[str]] = None, pdf_path: Optional[Path] = None) -> str:
    """
    Google AI chat wrapper with similar interface to OpenAI chat and PDF support.
    Based on working reference implementation.
    Sets HTTPS_PROXY specifically for Gemini API calls.
    
    Args:
        messages: List of chat messages
        model: Gemini model to use
        request_timeout: Request timeout in seconds
        prompt_type: Type of prompt
        fallback_models: List of fallback models if primary fails
        pdf_path: Optional PDF file to include in the request
    """
    if not GOOGLE_AI_AVAILABLE:
        raise APIError("Google AI SDK not available. Please install with: pip install google-generativeai")
    
    # Set proxy specifically for Google AI API (not needed for OpenAI)
    original_proxy = os.environ.get("HTTPS_PROXY")
    os.environ["HTTPS_PROXY"] = "http://127.0.0.1:7078"
    print(f"Set proxy for Gemini API: {os.environ['HTTPS_PROXY']}")
    
    if False and pdf_path and pdf_path.exists():  # PDF upload disabled
        print(f"INFO: Including PDF in Gemini request: {pdf_path.name}")
    
    try:
        # Configure API key - use hardcoded key as in reference
        api_key = "AIzaSyCXhoRyRmp_6Rpbp9eZjjwEvE11KrKIJII"
        genai.configure(api_key=api_key)
        
        logger.info(f"Making Google AI API call to {model} for {prompt_type}")
        print(f"INFO: Making Google AI API call to {model} for {prompt_type}...")
        
        # Set timeout
        if request_timeout is None:
            request_timeout = 1800  # 30 minutes default
        
        # Convert OpenAI messages to a single prompt and handle PDF
        combined_prompt = ""
        content_parts = []
        
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            
            if role == "system":
                combined_prompt += f"System: {content}\n\n"
            elif role == "user":
                combined_prompt += f"User: {content}\n\n"
            elif role == "assistant":
                combined_prompt += f"Assistant: {content}\n\n"
        
        # Add PDF content if provided
        if False and pdf_path and pdf_path.exists():  # PDF upload disabled
            try:
                # For Google AI, we need to upload the file using the file API
                # This is a simplified approach - in production you might want to use the proper file upload API
                import google.generativeai as genai
                
                # Upload the PDF file
                uploaded_file = genai.upload_file(path=str(pdf_path), mime_type="application/pdf")
                
                # Add PDF note to text prompt
                combined_prompt += "\n\n**Please also review the attached PDF document which shows the rendered output of the LaTeX paper.**\n"
                
                # Create content parts for multimodal input
                content_parts = [
                    combined_prompt.strip(),
                    uploaded_file
                ]
                
                print(f"PDF uploaded to Gemini: {pdf_path.name} ({pdf_path.stat().st_size // 1024} KB)")
                
            except Exception as pdf_error:
                print(f"WARNING: Failed to upload PDF to Gemini: {pdf_error}")
                print("Continuing with text-only request...")
                content_parts = [combined_prompt.strip()]
        else:
            content_parts = [combined_prompt.strip()]
        
        # Create model instance using the reference approach
        genai_model = genai.GenerativeModel(model)
        
        print(f"Sending Google AI request with timeout={request_timeout}s...")
        
        # Generate content with multimodal support
        if len(content_parts) > 1:
            # Multimodal request (text + PDF)
            response = genai_model.generate_content(content_parts)
        else:
            # Text-only request
            response = genai_model.generate_content(content_parts[0])
        
        print(f"✓ Google AI API call successful - response: {len(response.text):,} characters")
        logger.info(f"Google AI API call successful for {model} ({prompt_type}) - response: {len(response.text):,} chars")
        return response.text
        
    except Exception as e:
        error_msg = str(e)
        print(f"ERROR: Google AI API Error: {error_msg}")
        
        # Provide helpful proxy setup guidance
        if "503" in error_msg or "connect" in error_msg.lower() or "timeout" in error_msg.lower():
            print("Connection issue detected. Proxy is set for Gemini API.")
            print("   Check your network connection to Google AI API")
        
        # Try fallback models if available
        if fallback_models:
            print(f"WARNING: Primary model {model} failed, trying fallback models...")
            for fallback_model in fallback_models:
                try:
                    print(f"INFO: Attempting fallback model: {fallback_model}")
                    if fallback_model.startswith(('gemini', 'models/gemini')):
                        return _google_chat(messages, fallback_model, request_timeout, prompt_type, None, pdf_path)
                    else:
                        # For OpenAI fallback, restore original proxy
                        if original_proxy is not None:
                            os.environ["HTTPS_PROXY"] = original_proxy
                        elif "HTTPS_PROXY" in os.environ:
                            del os.environ["HTTPS_PROXY"]
                        print("INFO: Removed proxy for OpenAI fallback")
                        return _openai_chat(messages, fallback_model, request_timeout, prompt_type, None, pdf_path)
                except Exception as fallback_error:
                    print(f"WARNING: Fallback model {fallback_model} also failed: {fallback_error}")
                    continue
        
        raise APIError(f"Google AI model {model} failed: {error_msg}")
    
    finally:
        # Restore original proxy setting after Google AI call
        if original_proxy is not None:
            os.environ["HTTPS_PROXY"] = original_proxy
        elif "HTTPS_PROXY" in os.environ:
            del os.environ["HTTPS_PROXY"]
    print("INFO: Restored original proxy settings")

def _nowstamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")

def _prepare_project_dir(output_dir: Path, modify_existing: bool) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    # Check for any .tex file, not just paper.tex
    if modify_existing and any(output_dir.glob("*.tex")):
        return output_dir
    project_dir = output_dir / _nowstamp()
    project_dir.mkdir(parents=True, exist_ok=True)
    return project_dir

def _generate_research_ideas(
    topic: str, 
    field: str, 
    question: str, 
    model: str,
    prompt_generator: DocumentPromptGenerator,
    doc_type: DocumentType,
    num_ideas: int = 15,
    request_timeout: Optional[int] = 3600,
    fallback_models: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Generate and rank multiple research ideas for the given topic/field/question.
    
    Args:
        topic: Research topic
        field: Research field
        question: Research question
        model: AI model to use
        num_ideas: Number of ideas to generate (10-20)
        request_timeout: Timeout for API calls
        fallback_models: Fallback models if primary fails
        
    Returns:
        Dictionary containing ranked ideas with analysis
    """
    print(f"INFO: Generating {num_ideas} research ideas for '{topic}' in {field}...")
    
    ideation_prompt = prompt_generator.get_ideation_prompt(
        doc_type=doc_type,
        topic=topic,
        field=field,
        question=question,
        num_ideas=num_ideas
    )
    try:
        print("  Sending ideation request to AI...")
        messages = [{"role": "user", "content": ideation_prompt}]
        
        response = _universal_chat(
            messages=messages,
            model=model,
            request_timeout=request_timeout,
            prompt_type="ideation",
            fallback_models=fallback_models or []
        )
        
        print("INFO: Ideas generated successfully.")
        
        # Parse the response to extract structured data
        ideas = _parse_ideation_response(response)
        
        print(f"INFO: Parsed {len(ideas)} ideas from response")
        
        # Display summary
        print("INFO: Research Ideas Summary:")
        print("" * 60)
        
        for i, idea in enumerate(ideas[:5], 1):  # Show top 5
            originality = idea.get('originality', 0)
            impact = idea.get('impact', 0) 
            feasibility = idea.get('feasibility', 0)
            overall_score = (originality + impact + feasibility) / 3
            
            print(f"{i}. {idea.get('title', f'Idea {i}')}")
            print(f"   Scores: O={originality}, I={impact}, F={feasibility}, Overall={overall_score:.1f}")
            print(f"   {idea.get('core_concept', '')[:80]}...")
            print()
        
        return {
            "ideas": ideas,
            "raw_response": response,
            "selected_idea": ideas[0] if ideas else None,
            "topic": topic,
            "field": field,
            "question": question
        }
        
    except Exception as e:
        print(f"ERROR: Ideation failed: {e}")
        print("INFO: Proceeding with original topic/question...")
        
        # Fallback: return original topic as single idea
        return {
            "ideas": [{
                "title": topic,
                "core_concept": question,
                "originality": 7,
                "impact": 7,
                "feasibility": 8,
                "pros": ["Clear research direction", "Well-defined scope"],
                "cons": ["No alternative exploration", "Limited creativity"]
            }],
            "raw_response": f"Fallback response for {topic}",
            "selected_idea": {
                "title": topic,
                "core_concept": question,
                "originality": 7,
                "impact": 7,
                "feasibility": 8
            },
            "topic": topic,
            "field": field,
            "question": question
        }

def _parse_ideation_response(response: str) -> List[Dict[str, Any]]:
    """
    Parse the ideation response to extract structured idea data.
    
    Returns:
        List of dictionaries, each containing idea details
    """
    ideas = []
    
    try:
        import re
        
        # Split response into individual ideas
        idea_sections = re.split(r'## Research Idea #\d+', response)
        
        for section in idea_sections[1:]:  # Skip first empty split
            idea = {}
            
            # Extract title
            title_match = re.search(r'\*\*Title\*\*:\s*(.+)', section)
            if title_match:
                idea['title'] = title_match.group(1).strip()
            
            # Extract core concept
            concept_match = re.search(r'\*\*Core Concept\*\*:\s*(.+?)(?=\*\*|$)', section, re.DOTALL)
            if concept_match:
                idea['core_concept'] = concept_match.group(1).strip()
            
            # Extract scores
            for score_type in ['Originality', 'Impact', 'Feasibility']:
                pattern = fr'\*\*{score_type}\*\*:\s*(\d+)'
                match = re.search(pattern, section)
                if match:
                    idea[score_type.lower()] = int(match.group(1))
            
            # Extract pros and cons
            pros_match = re.search(r'\*\*Pros\*\*:\s*(.+?)(?=\*\*Cons|$)', section, re.DOTALL)
            if pros_match:
                pros_text = pros_match.group(1)
                idea['pros'] = [line.strip('- ').strip() for line in pros_text.split('\n') if line.strip().startswith('-')]
            
            cons_match = re.search(r'\*\*Cons\*\*:\s*(.+?)(?=\*\*|$)', section, re.DOTALL)
            if cons_match:
                cons_text = cons_match.group(1)
                idea['cons'] = [line.strip('- ').strip() for line in cons_text.split('\n') if line.strip().startswith('-')]
            
            if idea.get('title'):  # Only add if we got at least a title
                ideas.append(idea)
        
        # Sort by overall score (originality + impact + feasibility)
        def calculate_score(idea):
            return (idea.get('originality', 0) + idea.get('impact', 0) + idea.get('feasibility', 0)) / 3
        
        ideas.sort(key=calculate_score, reverse=True)
        
    except Exception as e:
        print(f"WARNING: Parsing error: {e}")
        # Return a single fallback idea
        ideas = [{
            "title": "Research Analysis",
            "core_concept": "Comprehensive analysis of the research topic",
            "originality": 6,
            "impact": 6,
            "feasibility": 8,
            "pros": ["Systematic approach", "Clear methodology"],
            "cons": ["Limited novelty", "Standard approach"]
        }]
    
    return ideas

def test_time_compute_scaling(
    model: str,
    candidate_counts: List[int] = [3, 5, 7, 10],
    timeout_base: int = 1800,
    test_prompt: str = None
) -> Dict[str, Any]:
    """
    Implement proper test-time compute scaling with candidate generation and selection.
    
    This method generates multiple candidate responses for each iteration and selects
    the best one using quality metrics, demonstrating how additional compute at test time
    can improve performance quality.
    
    Args:
        model: AI model to test (e.g., 'gpt-5', 'gemini-1.5-pro', 'gpt-4o')
        candidate_counts: List of candidate counts to test for scaling analysis
        timeout_base: Base timeout in seconds for API calls
        test_prompt: Custom prompt for testing (uses default if None)
    
    Returns:
        Dictionary containing quality results, scaling analysis, and best candidates
    """
    import time
    import statistics
    import re
    
    print(f"INFO: Starting test-time compute scaling with candidate generation for {model}")
    print(f"INFO: Testing candidate counts: {candidate_counts}")
    
    # Default test prompt if none provided
    if test_prompt is None:
        test_prompt = (
            "Design an efficient algorithm for solving the traveling salesman problem "
            "for graphs with 20-50 nodes. Provide a detailed explanation of your approach, "
            "analyze its time complexity, and discuss potential optimizations. "
            "Include pseudocode and explain why your solution is superior to basic approaches."
        )
    
    results = {
        "model": model,
        "test_config": {
            "candidate_counts_tested": candidate_counts,
            "base_timeout": timeout_base,
            "test_prompt_length": len(test_prompt)
        },
        "candidate_results": {},
        "quality_analysis": {},
        "scaling_analysis": {},
        "best_candidates": {}
    }
    
    def _evaluate_response_quality(response: str) -> Dict[str, float]:
        """Evaluate response quality using multiple metrics."""
        metrics = {}
        
        # Length and detail score (longer responses often more detailed)
        metrics['length_score'] = min(len(response) / 2000, 1.0)
        
        # Technical depth (presence of technical terms)
        technical_terms = ['algorithm', 'complexity', 'optimization', 'efficiency', 
                          'implementation', 'analysis', 'performance', 'solution', 
                          'approach', 'method', 'technique', 'strategy']
        tech_count = sum(1 for term in technical_terms if term.lower() in response.lower())
        metrics['technical_depth'] = min(tech_count / 8, 1.0)
        
        # Structure score (presence of organized sections)
        structure_indicators = ['step', 'first', 'second', 'third', 'finally', 
                               'algorithm:', 'approach:', 'solution:', 'analysis:',
                               '1.', '2.', '3.', 'pseudocode', 'complexity:']
        structure_count = sum(1 for indicator in structure_indicators 
                             if indicator.lower() in response.lower())
        metrics['structure_score'] = min(structure_count / 6, 1.0)
        
        # Code/pseudocode presence
        code_indicators = ['def ', 'for ', 'while ', 'if ', 'return ', 'function',
                          'procedure', 'begin', 'end', '```', 'algorithm']
        code_count = sum(1 for indicator in code_indicators 
                        if indicator.lower() in response.lower())
        metrics['code_presence'] = min(code_count / 4, 1.0)
        
        # Mathematical notation (for algorithmic problems)
        math_indicators = ['O(', '(', '(', 'log', 'n', 'n^2', 'exponential', 
                          'polynomial', 'linear', 'quadratic', 'complexity']
        math_count = sum(1 for indicator in math_indicators 
                        if indicator.lower() in response.lower())
        metrics['math_notation'] = min(math_count / 3, 1.0)
        
        # Overall quality score (weighted average)
        metrics['overall_quality'] = (
            metrics['length_score'] * 0.2 +
            metrics['technical_depth'] * 0.25 +
            metrics['structure_score'] * 0.25 +
            metrics['code_presence'] * 0.15 +
            metrics['math_notation'] * 0.15
        )
        
        return metrics
    
    def _select_best_candidate(candidates: List[str]) -> Tuple[str, Dict[str, float], int]:
        """Select the best candidate based on quality metrics."""
        best_candidate = ""
        best_score = -1
        best_index = -1
        best_metrics = {}
        
        for i, candidate in enumerate(candidates):
            metrics = _evaluate_response_quality(candidate)
            if metrics['overall_quality'] > best_score:
                best_score = metrics['overall_quality']
                best_candidate = candidate
                best_index = i
                best_metrics = metrics
        
        return best_candidate, best_metrics, best_index
    
    # Test each candidate count
    for candidate_count in candidate_counts:
        print(f"INFO: Testing {candidate_count} candidates...")
        
        # Prepare test messages
        test_messages = [
            {"role": "system", "content": "You are an expert computer scientist and algorithm designer. Provide detailed, technically accurate solutions with clear explanations."},
            {"role": "user", "content": test_prompt}
        ]
        
        # Generate multiple candidates
        candidates = []
        generation_times = []
        
        print(f"INFO: Generating {candidate_count} candidate responses...")
        
        for i in range(candidate_count):
            print(f"    Generating candidate {i + 1}/{candidate_count}...")
            
            start_time = time.time()
            
            try:
                # Add slight variation to encourage diverse responses
                varied_messages = test_messages.copy()
                if i > 0:
                    variation_prompts = [
                        " Focus on a different algorithmic approach.",
                        " Emphasize implementation details and optimizations.",
                        " Provide alternative solutions and compare them.",
                        " Include more mathematical analysis and proofs.",
                        " Focus on practical considerations and real-world applications."
                    ]
                    variation = variation_prompts[i % len(variation_prompts)]
                    varied_messages[-1]["content"] += variation
                
                # Make API call
                response = _universal_chat(
                    messages=varied_messages,
                    model=model,
                    request_timeout=timeout_base,
                    prompt_type="test_scaling"
                )
                
                generation_time = time.time() - start_time
                candidates.append(response)
                generation_times.append(generation_time)
                
                print(f"       Candidate {i + 1} generated: {generation_time:.2f}s, {len(response)} chars")
                
            except Exception as e:
                print(f"       Candidate {i + 1} failed: {e}")
                candidates.append("")
                generation_times.append(None)
        
        # Evaluate all candidates and select the best
        valid_candidates = [c for c in candidates if c.strip()]
        
        if valid_candidates:
            print(f"  Selecting best candidate from {len(valid_candidates)} valid responses...")
            
            best_candidate, best_metrics, best_index = _select_best_candidate(valid_candidates)
            
            # Evaluate all candidates for comparison
            all_metrics = []
            for candidate in valid_candidates:
                metrics = _evaluate_response_quality(candidate)
                all_metrics.append(metrics['overall_quality'])
            
            results["candidate_results"][candidate_count] = {
                "total_candidates": candidate_count,
                "successful_candidates": len(valid_candidates),
                "best_candidate_index": best_index,
                "best_quality_score": best_metrics['overall_quality'],
                "average_quality_score": statistics.mean(all_metrics),
                "quality_improvement": best_metrics['overall_quality'] - statistics.mean(all_metrics) if len(all_metrics) > 1 else 0,
                "std_dev_quality": statistics.stdev(all_metrics) if len(all_metrics) > 1 else 0,
                "generation_times": [t for t in generation_times if t is not None],
                "total_compute_time": sum(t for t in generation_times if t is not None),
                "best_metrics_breakdown": best_metrics
            }
            
            results["best_candidates"][candidate_count] = {
                "content": best_candidate,
                "quality_score": best_metrics['overall_quality'],
                "metrics": best_metrics
            }
            
            print(f"    Best candidate quality: {best_metrics['overall_quality']:.3f}")
            print(f"    Average quality: {statistics.mean(all_metrics):.3f}")
            print(f"      Quality improvement: {best_metrics['overall_quality'] - statistics.mean(all_metrics):.3f}")
            
        else:
            results["candidate_results"][candidate_count] = {
                "error": "All candidates failed",
                "total_candidates": candidate_count,
                "successful_candidates": 0
            }
    
    # Analyze quality scaling with compute
    print(f"\nAnalyzing quality scaling with test-time compute...")
    
    successful_results = {k: v for k, v in results["candidate_results"].items() 
                         if isinstance(v, dict) and "best_quality_score" in v}
    
    if len(successful_results) >= 2:
        candidate_counts_sorted = sorted(successful_results.keys())
        quality_scores = [successful_results[k]["best_quality_score"] for k in candidate_counts_sorted]
        compute_times = [successful_results[k]["total_compute_time"] for k in candidate_counts_sorted]
        
        # Calculate quality improvement per additional candidate
        quality_improvements = []
        compute_ratios = []
        
        for i in range(1, len(candidate_counts_sorted)):
            quality_improvement = quality_scores[i] - quality_scores[i-1]
            candidate_ratio = candidate_counts_sorted[i] / candidate_counts_sorted[i-1]
            compute_ratio = compute_times[i] / compute_times[i-1] if compute_times[i-1] > 0 else 1
            
            quality_improvements.append(quality_improvement)
            compute_ratios.append(compute_ratio)
        
        avg_quality_per_candidate = statistics.mean(quality_improvements) if quality_improvements else 0
        avg_compute_scaling = statistics.mean(compute_ratios) if compute_ratios else 1
        
        # Determine scaling efficiency
        efficiency_ratio = avg_quality_per_candidate * 10 / avg_compute_scaling  # Scale for readability
        
        if efficiency_ratio > 0.5:
            scaling_efficiency = "excellent"
        elif efficiency_ratio > 0.2:
            scaling_efficiency = "good"
        elif efficiency_ratio > 0.1:
            scaling_efficiency = "moderate"
        else:
            scaling_efficiency = "poor"
        
        results["scaling_analysis"] = {
            "quality_improvement_per_candidate": avg_quality_per_candidate,
            "compute_scaling_factor": avg_compute_scaling,
            "efficiency_ratio": efficiency_ratio,
            "scaling_efficiency": scaling_efficiency,
            "max_quality_achieved": max(quality_scores),
            "quality_variance": statistics.stdev(quality_scores) if len(quality_scores) > 1 else 0,
            "optimal_candidate_count": candidate_counts_sorted[quality_scores.index(max(quality_scores))]
        }
        
        # Generate recommendations for test-time compute scaling
        results["recommendations"] = {
            "recommended_candidate_count": max(3, min(7, results["scaling_analysis"]["optimal_candidate_count"])),
            "quality_ceiling": max(quality_scores),
            "compute_budget_per_query": statistics.mean(compute_times),
            "scaling_notes": f"Test-time compute scaling shows {scaling_efficiency} efficiency with quality improvements of {avg_quality_per_candidate:.3f} per additional candidate"
        }
    
    else:
        results["scaling_analysis"] = {"error": "Insufficient data for scaling analysis"}
        results["recommendations"] = {"error": "Cannot generate recommendations due to insufficient data"}
    
    # Print comprehensive summary
    print(f"\nTest-Time Compute Scaling Summary for {model}")
    print(f"")
    
    for candidate_count, result in results["candidate_results"].items():
        if "best_quality_score" in result:
            print(f"{candidate_count:2d} candidates: Quality {result['best_quality_score']:.3f} "
                  f"(+{result['quality_improvement']:.3f} improvement) "
                  f"Time: {result['total_compute_time']:.1f}s")
        else:
            print(f"{candidate_count:2d} candidates: FAILED")
    
    if "quality_improvement_per_candidate" in results["scaling_analysis"]:
        analysis = results["scaling_analysis"]
        print(f"\nQuality scaling: +{analysis['quality_improvement_per_candidate']:.3f} per candidate")
        print(f"Efficiency: {analysis['scaling_efficiency']} ({analysis['efficiency_ratio']:.2f})")
        print(f"Peak quality: {analysis['max_quality_achieved']:.3f}")
        print(f"Optimal count: {analysis['optimal_candidate_count']} candidates")
    
    if "recommended_candidate_count" in results["recommendations"]:
        rec = results["recommendations"]
        print(f"\nRecommended candidates: {rec['recommended_candidate_count']}")
        print(f"Expected compute time: {rec['compute_budget_per_query']:.1f}s per query")
    
    print(f"")
    
    return results

def _generate_best_revision_candidate(
    current_tex: str, 
    sim_summary: str, 
    review_text: str, 
    latex_errors: str, 
    project_dir: Path, 
    user_prompt: Optional[str], 
    model: str, 
    request_timeout: int, 
    config: Any, 
    candidate_count: int = 3,
    output_diffs: bool = False,
    pdf_path: Optional[Path] = None,
) -> str:
    """
    Generate multiple revision candidates and select the best one using test-time compute scaling.
    
    Args:
        current_tex: Current LaTeX paper content
        sim_summary: Simulation summary
        review_text: Review feedback
        latex_errors: LaTeX compilation errors (if any)
        project_dir: Project directory path
        user_prompt: Optional user instructions
        model: AI model to use
        request_timeout: Request timeout
        config: Configuration object
        candidate_count: Number of revision candidates to generate
    
    Returns:
        Best revision candidate based on quality metrics
    """
    import time
    import hashlib
    
    print(f"  Generating {candidate_count} revision candidates...")
    
    candidates = []
    generation_times = []
    
    # Generate multiple revision candidates with aggressive variations
    for i in range(candidate_count):
        print(f"    Generating revision candidate {i + 1}/{candidate_count}...")
        
        start_time = time.time()
        
        try:
            # Create varied revision prompts to encourage diversity
            base_prompt = _revise_prompt(current_tex, sim_summary, review_text, latex_errors, project_dir, user_prompt, enable_quality_enhancements=config.enable_quality_enhancements)
            
            # Add MUCH MORE AGGRESSIVE variation instructions to force different approaches
            if i > 0:
                variation_instructions = [
                    "\n\nIMPORTANT: You MUST make SUBSTANTIAL changes. Rewrite at least 30% of the content. Add new sections, expand existing ones, change technical approaches, improve mathematical rigor.",
                    "\n\nCRITICAL: Focus on MAJOR restructuring. Reorganize sections, add missing methodology details, enhance experimental validation. Make this version significantly different from the original.",
                    "\n\nESSENTIAL: Prioritize COMPREHENSIVE improvements. Add new theoretical foundations, expand results discussion, include additional related work. Transform the paper substantially.",
                    "\n\nREQUIRED: Concentrate on FUNDAMENTAL enhancements. Strengthen mathematical formulations, add implementation details, improve practical applications. Create a markedly different version.",
                    "\n\nMANDATORY: Focus on EXTENSIVE modifications. Rewrite abstract and conclusion, add new figures/tables concepts, enhance technical depth throughout. Generate a substantially revised paper."
                ]
                
                variation = variation_instructions[(i - 1) % len(variation_instructions)]
                
                # Add variation to the system prompt with higher temperature equivalent instructions
                varied_prompt = base_prompt.copy()
                varied_prompt[0]["content"] += variation
                
                # Also modify the user prompt to be more aggressive
                if len(varied_prompt) > 1:
                    varied_prompt[1]["content"] += f"\n\nVariation {i}: " + variation
            else:
                varied_prompt = base_prompt
            
            # Generate revision candidate with increased temperature-equivalent randomness
            candidate = _universal_chat(
                varied_prompt, 
                model=model, 
                request_timeout=request_timeout, 
                prompt_type="revise", 
                fallback_models=config.fallback_models
            )
            
            generation_time = time.time() - start_time
            candidates.append(candidate)
            generation_times.append(generation_time)
            
            print(f"       Candidate {i + 1} generated: {generation_time:.2f}s, {len(candidate)} chars")
            
            # DEBUG: Show first 300 chars of each revision candidate
            print(f"      DEBUG - Revision candidate {i + 1} preview:")
            print(f"      {candidate[:300]}...")
            print(f"      {''*60}")
            
            # Show diff from original for each candidate
            if output_diffs:
                print(f"      Comparing candidate {i + 1} to original...")
                _save_candidate_diff(current_tex, candidate, i + 1, "candidate")
            
        except Exception as e:
            print(f"       Candidate {i + 1} failed: {e}")
            candidates.append("")
            generation_times.append(None)
    
    # Filter out empty candidates
    valid_candidates = [(i, c) for i, c in enumerate(candidates) if c.strip()]
    
    if not valid_candidates:
        print("      All revision candidates failed, using empty response")
        return ""
    
    print(f"  Selecting best candidate from {len(valid_candidates)} valid responses...")
    
    # Use LLM to evaluate and select the best revision candidate
    print(f"  Asking LLM to evaluate revision candidates...")
    
    # Prepare candidates for LLM evaluation (in-memory only)
    revision_candidates = [c for _, c in valid_candidates]

    # DEBUG: Show what we're sending to the LLM
    print(f"  DEBUG - Sending {len(revision_candidates)} candidates to LLM for evaluation:")
    for i, candidate in enumerate(revision_candidates):
        print(f"  - Candidate {i+1}: {len(candidate)} chars")
    print(f"  Original content: {len(current_tex)} chars")
    print(f"  Review text: {len(review_text)} chars")
    print(f"  {''*60}")
    
    best_candidate_response = _select_best_revision_candidate_with_llm(
        revision_candidates, current_tex, review_text, model, request_timeout, config, pdf_path=pdf_path, project_dir=project_dir
    )
    
    # DEBUG: Show full LLM evaluation response
    print(f"  DEBUG - LLM Revision Evaluation Response:")
    print(f"  {best_candidate_response}")
    print(f"  {''*80}")
    
    # Parse the LLM selection response
    try:
        import re
        selection_match = re.search(r'SELECTED:\s*(\d+)', best_candidate_response, re.IGNORECASE)
        if selection_match:
            selected_idx = int(selection_match.group(1)) - 1  # Convert to 0-based index
            if 0 <= selected_idx < len(valid_candidates):
                orig_idx, best_candidate = valid_candidates[selected_idx]
                print(f"  LLM selected candidate {selected_idx + 1} (original index {orig_idx + 1})")
                
                # Show why this candidate was selected
                reasoning_match = re.search(r'REASONING:\s*(.*?)(?=SELECTED:|$)', best_candidate_response, re.DOTALL | re.IGNORECASE)
                if reasoning_match:
                    reasoning = reasoning_match.group(1).strip()
                    print(f"   Selection reasoning: {reasoning[:200]}...")
                
                # Show final selected candidate diff
                if output_diffs:
                    print(f"\nFINAL SELECTED CANDIDATE DIFF:")
                    _save_candidate_diff(current_tex, best_candidate, selected_idx + 1, "SELECTED")
                
                # Calculate compute time
                total_time = sum(t for t in generation_times if t)
                print(f"    Total compute time: {total_time:.1f}s")
                
                return best_candidate
            else:
                print(f"   Invalid selection index {selected_idx + 1}, using first candidate")
                return valid_candidates[0][1]
        else:
            print(f"   Could not parse LLM selection, using first candidate")
            return valid_candidates[0][1]
    except Exception as e:
        print(f"   Error parsing LLM selection: {e}, using first candidate")
        return valid_candidates[0][1]

def _evaluate_revision_quality(revised_text: str, review_text: str, latex_errors: str) -> Dict[str, float]:
    """
    Evaluate the quality of a revision candidate.
    
    Args:
        revised_text: The revised paper content
        review_text: Original review feedback
        latex_errors: LaTeX compilation errors
    
    Returns:
        Dictionary of quality metrics
    """
    metrics = {}
    
    # Length and detail score (good revisions often add content)
    metrics['length_score'] = min(len(revised_text) / 15000, 1.0)  # Reasonable paper length
    
    # LaTeX structure and formatting
    latex_indicators = ['\\begin{', '\\end{', '\\section', '\\subsection', '\\cite{', 
                       '\\ref{', '\\label{', '\\begin{equation}', '\\begin{figure}', '\\begin{table}']
    latex_count = sum(1 for indicator in latex_indicators if indicator in revised_text)
    metrics['latex_structure'] = min(latex_count / 15, 1.0)
    
    # Academic content quality
    academic_terms = ['methodology', 'analysis', 'results', 'discussion', 'conclusion',
                     'evaluation', 'experiment', 'validation', 'algorithm', 'approach',
                     'performance', 'comparison', 'implementation', 'framework']
    academic_count = sum(1 for term in academic_terms if term.lower() in revised_text.lower())
    metrics['academic_depth'] = min(academic_count / 10, 1.0)
    
    # Reference and citation quality
    citation_patterns = ['\\cite{', '\\citep{', '\\citet{', '\\citeauthor{']
    citation_count = sum(revised_text.count(pattern) for pattern in citation_patterns)
    metrics['citation_quality'] = min(citation_count / 20, 1.0)
    
    # Mathematical content (for technical papers)
    math_indicators = ['\\begin{equation}', '\\begin{align}', '$', '\\(', 'O(', 'algorithm',
                      'complexity', 'theorem', 'proof', 'lemma', 'definition']
    math_count = sum(1 for indicator in math_indicators if indicator.lower() in revised_text.lower())
    metrics['mathematical_content'] = min(math_count / 8, 1.0)
    
    # Simulation quality assessment
    metrics['simulation_quality'] = _evaluate_simulation_content(revised_text)
    
    # Response to review concerns (keyword matching)
    if review_text:
        # Extract key concerns from review
        concern_keywords = ['clarity', 'methodology', 'validation', 'experiment', 'analysis',
                           'comparison', 'evaluation', 'limitation', 'discussion', 'related work']
        
        addressed_concerns = 0
        for keyword in concern_keywords:
            if keyword.lower() in review_text.lower() and keyword.lower() in revised_text.lower():
                addressed_concerns += 1
        
        metrics['review_responsiveness'] = min(addressed_concerns / 5, 1.0)
    else:
        metrics['review_responsiveness'] = 0.5  # Neutral score if no review
    
    # LaTeX compilation likelihood (penalty for obvious errors)
    error_indicators = ['\\begin{', '\\end}', 'undefined', 'missing', '\\\\\\', '}{']
    error_count = sum(1 for indicator in error_indicators if indicator in revised_text)
    metrics['compilation_score'] = max(0, 1.0 - error_count / 10)
    
    # Overall quality score (weighted combination)
    metrics['overall_quality'] = (
        metrics['length_score'] * 0.12 +
        metrics['latex_structure'] * 0.18 +
        metrics['academic_depth'] * 0.18 +
        metrics['citation_quality'] * 0.12 +
        metrics['mathematical_content'] * 0.10 +
        metrics['simulation_quality'] * 0.15 +
        metrics['review_responsiveness'] * 0.12 +
        metrics['compilation_score'] * 0.03
    )
    
    return metrics

def _evaluate_simulation_content(text: str) -> float:
    """
    Evaluate the quality of simulation content in the paper.
    
    Args:
        text: Paper content to evaluate
        
    Returns:
        Simulation quality score (0.0 to 1.0)
    """
    score = 0.0
    
    # Check for filecontents simulation blocks
    if '\\begin{filecontents' in text and 'simulation.py' in text:
        score += 0.3
        
        # Extract simulation content from filecontents
        sim_pattern = r'\\begin\{filecontents\*?\}\{simulation\.py\}(.*?)\\end\{filecontents\*?\}'
        matches = re.findall(sim_pattern, text, re.DOTALL | re.IGNORECASE)
        
        if matches:
            sim_content = matches[0]
            
            # Check for comprehensive simulation indicators
            quality_indicators = [
                ('class ', 0.15),  # Object-oriented structure
                ('def ', 0.1),     # Function definitions
                ('import ', 0.1),  # Module imports
                ('numpy', 0.05),   # Scientific computing
                ('matplotlib', 0.05),  # Visualization
                ('experiment', 0.1),   # Experimental design
                ('result', 0.05),  # Results generation
                ('main()', 0.1),   # Main execution
                ('if __name__', 0.1),  # Proper script structure
                ('test', 0.05),    # Testing/validation
                ('scaling', 0.1),  # Scaling analysis
                ('candidate', 0.1), # Candidate generation (test-time scaling)
                ('quality', 0.05), # Quality evaluation
                ('algorithm', 0.05), # Algorithmic content
                ('performance', 0.05)  # Performance analysis
            ]
            
            for indicator, weight in quality_indicators:
                if indicator.lower() in sim_content.lower():
                    score += weight
            
            # Length bonus for substantial simulations
            sim_lines = len([line for line in sim_content.split('\n') 
                           if line.strip() and not line.strip().startswith('#')])
            if sim_lines > 50:
                score += 0.1
            if sim_lines > 100:
                score += 0.1
            if sim_lines > 200:
                score += 0.1
    
    # Check for simulation discussion in paper text
    simulation_terms = ['simulation', 'experiment', 'implementation', 'algorithm',
                       'numerical', 'computational', 'benchmark', 'evaluation']
    term_count = sum(1 for term in simulation_terms if term.lower() in text.lower())
    score += min(term_count / 20, 0.2)  # Up to 0.2 bonus for simulation discussion
    
    return min(score, 1.0)  # Cap at 1.0

def _generate_best_initial_draft_candidate(
    topic: str, 
    field: str, 
    question: str, 
    user_prompt: Optional[str], 
    model: str, 
    request_timeout: int, 
    config: Any, 
    candidate_count: int = 3
) -> str:
    """
    Generate multiple initial draft candidates and select the best one using test-time compute scaling.
    
    Args:
        topic: Research topic
        field: Research field
        question: Research question
        user_prompt: Optional user instructions
        model: AI model to use
        request_timeout: Request timeout
        config: Configuration object
        candidate_count: Number of draft candidates to generate
    
    Returns:
        Best initial draft candidate based on quality metrics
    """
    import time
    
    print(f"  Generating {candidate_count} initial draft candidates...")
    
    candidates = []
    generation_times = []
    
    # Generate multiple initial draft candidates with slight variations
    for i in range(candidate_count):
        print(f"    Generating draft candidate {i + 1}/{candidate_count}...")
        
        start_time = time.time()
        
        try:
            # Create varied initial draft prompts to encourage diversity
            base_prompt = _initial_draft_prompt(topic, field, question, user_prompt, config.enable_quality_enhancements)
            
            # Add variation instructions to encourage different approaches
            if i > 0:
                variation_instructions = [
                    "\nEmphasize theoretical rigor and mathematical formulations in your approach.",
                    "\nFocus on practical applications and experimental validation methods.",
                    "\nPrioritize comprehensive literature review and related work analysis.",
                    "\nConcentrate on novel algorithmic contributions and implementation details.",
                    "\nHighlight the broader impact and interdisciplinary connections."
                ]
                
                variation = variation_instructions[(i - 1) % len(variation_instructions)]
                
                # Add variation to the system prompt
                varied_prompt = base_prompt.copy()
                varied_prompt[0]["content"] += variation
            else:
                varied_prompt = base_prompt
            
            # Generate draft candidate
            candidate = _universal_chat(
                varied_prompt, 
                model=model, 
                request_timeout=request_timeout, 
                prompt_type="initial_draft", 
                fallback_models=config.fallback_models
            )
            
            generation_time = time.time() - start_time
            candidates.append(candidate)
            generation_times.append(generation_time)
            
            print(f"       Candidate {i + 1} generated: {generation_time:.2f}s, {len(candidate)} chars")
            
        except Exception as e:
            print(f"       Candidate {i + 1} failed: {e}")
            candidates.append("")
            generation_times.append(None)
    
    # Filter out empty candidates
    valid_candidates = [(i, c) for i, c in enumerate(candidates) if c.strip()]
    
    if not valid_candidates:
        print("      All draft candidates failed, using empty response")
        return ""
    
    print(f"  Selecting best candidate from {len(valid_candidates)} valid responses...")
    
    # Evaluate each candidate using quality metrics
    candidate_scores = []
    
    for idx, (orig_idx, candidate) in enumerate(valid_candidates):
        try:
            # Calculate quality metrics for this candidate
            metrics = _evaluate_initial_draft_quality(candidate, topic, field, question)
            score = metrics['overall_quality']
            candidate_scores.append((orig_idx, candidate, score, metrics))
            
            print(f"    Candidate {orig_idx + 1}: Quality score {score:.3f}")
            
        except Exception as e:
            print(f"     Failed to evaluate candidate {orig_idx + 1}: {e}")
            candidate_scores.append((orig_idx, candidate, 0.0, {}))
    
    # Select the best candidate
    if candidate_scores:
        best_idx, best_candidate, best_score, best_metrics = max(candidate_scores, key=lambda x: x[2])
        avg_score = sum(score for _, _, score, _ in candidate_scores) / len(candidate_scores)
        improvement = best_score - avg_score
        
        print(f"  Selected candidate {best_idx + 1} with quality score {best_score:.3f}")
        print(f"  Quality improvement over average: +{improvement:.3f}")
        print(f"    Total compute time: {sum(t for t in generation_times if t):.1f}s")
        
        return best_candidate
    else:
        print("      No valid candidates scored, returning first valid candidate")
        return valid_candidates[0][1] if valid_candidates else ""

def _evaluate_initial_draft_quality(draft_text: str, topic: str, field: str, question: str) -> Dict[str, float]:
    """
    Evaluate the quality of an initial draft candidate.
    
    Args:
        draft_text: The draft paper content
        topic: Research topic
        field: Research field  
        question: Research question
    
    Returns:
        Dictionary of quality metrics
    """
    metrics = {}
    
    # Length and completeness score
    metrics['length_score'] = min(len(draft_text) / 12000, 1.0)  # Target ~12k chars for initial draft
    
    # LaTeX structure quality
    latex_indicators = ['\\documentclass', '\\begin{document}', '\\end{document}', 
                       '\\section{', '\\subsection{', '\\begin{abstract}', '\\end{abstract}',
                       '\\begin{filecontents}', '\\bibliography{', '\\cite{']
    latex_count = sum(1 for indicator in latex_indicators if indicator in draft_text)
    metrics['latex_structure'] = min(latex_count / 8, 1.0)
    
    # Academic sections presence
    required_sections = ['abstract', 'introduction', 'related work', 'methodology', 
                        'results', 'discussion', 'conclusion', 'references']
    section_count = sum(1 for section in required_sections if section.lower() in draft_text.lower())
    metrics['section_completeness'] = min(section_count / 6, 1.0)
    
    # Topic relevance (keyword matching)
    topic_keywords = topic.lower().split() if topic else []
    field_keywords = field.lower().split() if field else []
    question_keywords = question.lower().split() if question else []
    
    all_keywords = topic_keywords + field_keywords + question_keywords
    keyword_matches = sum(1 for keyword in all_keywords if len(keyword) > 3 and keyword in draft_text.lower())
    metrics['topic_relevance'] = min(keyword_matches / max(len(all_keywords), 1), 1.0)
    
    # Technical depth indicators
    technical_terms = ['algorithm', 'methodology', 'analysis', 'evaluation', 'implementation',
                      'experiment', 'validation', 'optimization', 'performance', 'framework']
    tech_count = sum(1 for term in technical_terms if term.lower() in draft_text.lower())
    metrics['technical_depth'] = min(tech_count / 8, 1.0)
    
    # Reference quality
    citation_patterns = ['\\cite{', '\\citep{', '\\citet{', '\\citeauthor{']
    citation_count = sum(draft_text.count(pattern) for pattern in citation_patterns)
    metrics['citation_quality'] = min(citation_count / 15, 1.0)
    
    # Mathematical content (for technical papers)
    math_indicators = ['\\begin{equation}', '\\begin{align}', '$', '\\(', 'theorem', 'proof', 'lemma']
    math_count = sum(1 for indicator in math_indicators if indicator.lower() in draft_text.lower())
    metrics['mathematical_content'] = min(math_count / 5, 1.0)
    
    # Simulation quality assessment
    metrics['simulation_quality'] = _evaluate_simulation_content(draft_text)
    
    # Overall quality score (weighted combination)
    metrics['overall_quality'] = (
        metrics['length_score'] * 0.14 +
        metrics['latex_structure'] * 0.18 +
        metrics['section_completeness'] * 0.18 +
        metrics['topic_relevance'] * 0.14 +
        metrics['technical_depth'] * 0.10 +
        metrics['citation_quality'] * 0.10 +
        metrics['mathematical_content'] * 0.08 +
        metrics['simulation_quality'] * 0.08
    )
    
    return metrics

def _initial_draft_prompt(topic: str, field: str, question: str, user_prompt: Optional[str] = None, enable_quality_enhancements: bool = True) -> List[Dict[str, str]]:
    sys_prompt = (
        "You are a meticulous scientist writing a LaTeX paper suitable for a top journal. "
        
        "CRITICAL REQUIREMENTS - NO EXCEPTIONS:\n"
        "1. SINGLE FILE ONLY: Create ONE LaTeX file with NO separate bibliography files\n"
        "2. EMBEDDED REFERENCES MANDATORY: Include ALL references directly in the paper.tex file using EITHER:\n"
        "   - \\begin{filecontents*}{refs.bib}...\\end{filecontents*} at the TOP of the file, OR\n"
        "   - \\begin{thebibliography}...\\end{thebibliography} at the END of the file\n"
        "   - NO separate refs.bib files are allowed - everything must be in paper.tex\n"
        "   - Use \\bibliography{refs} to reference the embedded filecontents\n"
        "3. COMPILABLE: The file must compile successfully with pdflatex\n"
        "4. NO FILENAMES IN TEXT: The paper text must NOT contain ANY references to specific filenames, code files, data files, or directory structures. Avoid:\n"
        "   - Specific filenames like 'simulation.py', 'results.txt', 'data.csv', 'model.pkl', etc.\n"
        "   - Directory paths like 'output/', 'src/', 'data/', etc.\n"
        "   - File extensions like '.py', '.txt', '.csv', '.pkl', etc.\n"
        "   - Use generic descriptions like 'our implementation', 'experimental setup', 'computational framework'\n"
        "   - The paper should describe methods and results without revealing the underlying file structure\n"
        "5. AUTHENTIC REFERENCES MANDATORY: All references must be real, published works with correct details:\n"
        "   - Minimum 15-20 authentic references from reputable sources\n"
        "   - Authors, titles, journals, years, DOIs must be accurate\n"
        "   - NO FAKE or PLACEHOLDER references\n"
        "   - All references must be cited in the text using \\cite{} commands\n"
        "6. SELF-CONTAINED CONTENT: ALL tables, figures, diagrams must be defined within the LaTeX file using TikZ, tabular, or other LaTeX constructs. NO external image files.\n"
        "7. DATA-DRIVEN RESULTS: All numerical values in tables/figures must come from actual simulation results, not made-up numbers.\n"
        "8. SIMULATION REQUIREMENTS: If the paper needs numerical results, you MUST include a comprehensive simulation.py file with:\n"
        "   - Complete, runnable Python code implementing the paper's main contribution\n"
        "   - Proper experiment design with multiple trials, statistical analysis, and reproducible results\n"
        "   - For ML/AI papers: Include test-time compute scaling where appropriate (generate multiple candidates, select best)\n"
        "   - For optimization papers: Include proper algorithm implementation with convergence analysis\n"
        "   - For system papers: Include performance benchmarks and comparative evaluation\n"
        "   - All experimental parameters clearly documented and configurable\n"
        "   - Results saved to 'results.txt' for verification and reproducibility\n"
        "9. APPROPRIATE STRUCTURE: The paper structure and sections must align with the paper type and field conventions:\n"
        "8. APPROPRIATE STRUCTURE: The paper structure and sections must align with the paper type and field conventions:\n"
        "   - Theoretical papers: Abstract, Introduction, Related Work, Theory/Methods, Analysis, Discussion, Conclusion\n"
        "   - Experimental papers: Abstract, Introduction, Related Work, Methodology, Experiments, Results, Discussion, Conclusion\n"
        "   - Survey papers: Abstract, Introduction, Background, Classification/Taxonomy, Comparative Analysis, Future Directions, Conclusion\n"
        "   - Systems papers: Abstract, Introduction, Related Work, System Design, Implementation, Evaluation, Discussion, Conclusion\n"
        "   - Algorithm papers: Abstract, Introduction, Related Work, Problem Definition, Algorithm Description, Analysis, Experiments, Conclusion\n"
        "9. RESULTS DOCUMENTATION: The numerical results from running simulation.py MUST be saved in 'results.txt' in the project folder for reproducibility and verification.\n"
        "10. FIGURE GENERATION: If the paper includes figures generated by code, ALL figure generation code must be included in simulation.py and figures must be saved in the local project folder.\n"
        "11. SINGLE CODE FILE: ALL computational code must be consolidated into ONE simulation.py file - no additional .py files, scripts, or code fragments.\n\n"
        
        "STRUCTURE ALIGNMENT REQUIREMENTS:\n"
        "- Identify the paper type based on the research question and field\n"
        "- Use appropriate section names and organization for that paper type\n"
        "- Include field-specific sections (e.g., 'Threat Model' for security papers, 'Clinical Validation' for medical papers)\n"
        "- Follow established conventions for the target journal/conference\n"
        "- Ensure logical flow appropriate for the paper's contribution type\n"
        "- Include appropriate evaluation methodology for the paper type\n\n"
        
        "📋 PROFESSIONAL LAYOUT AND STRUCTURE REQUIREMENTS:\n"
        "Design the paper with professional academic layout standards:\n"
        "- SECTION ORGANIZATION: Ensure logical section flow with appropriate subsection hierarchy\n"
        "- GLOSSARY PLACEMENT: If terminology section needed, place before references or as appendix\n"
        "- APPENDIX PLANNING: Reserve appendix for supplementary material like detailed proofs, large tables, or extended algorithms\n"
        "- PSEUDOCODE PLACEMENT: Keep essential algorithms in main text, move detailed implementations to appendix\n"
        "- BALANCED SECTIONS: Ensure no section is disproportionately long or short relative to its importance\n"
        "- REFERENCE ORGANIZATION: Place references at document end with consistent formatting\n"
        "- ACKNOWLEDGMENTS: Include acknowledgments section before references if applicable\n"
        "- PAGE FLOW: Design content flow to minimize awkward page breaks and orphaned elements\n"
        "- VISUAL HIERARCHY: Use consistent heading styles and proper sectioning commands\n\n"
        
        "🎯 LATEX FLOAT POSITIONING REQUIREMENTS:\n"
        "Use proper LaTeX positioning for all tables and figures from the start:\n"
        "- USE [htbp] POSITIONING: Always use \\begin{figure}[htbp] and \\begin{table}[htbp] for optimal placement\n"
        "  * NEVER use [H] which forces exact placement and creates poor page flow\n"
        "  * [htbp] allows LaTeX to optimize: h=here, t=top, b=bottom, p=page\n"
        "- PROPER SPACING: Add appropriate spacing around figures/tables using \\vspace{0.5em} if needed\n"
        "- CAPTION POSITIONING: Place captions ABOVE tables and BELOW figures (academic standard)\n"
        "- REFERENCE IN TEXT: Reference every figure/table in text using \\ref{} before it appears\n"
        "- CENTERED CONTENT: Use \\centering inside figure/table environments (not \\begin{center})\n"
        "- LOGICAL ORDERING: Ensure figures/tables appear in same order as referenced in text\n"
        "- SIZE OPTIMIZATION: Design tables to fit page width using \\small or \\footnotesize if needed\n"
        "- LABEL CONSISTENCY: Use consistent labeling scheme (fig:name, tab:name) for all floats\n\n"
        
        "📊 VISUAL QUALITY STANDARDS:\n"
        "Ensure all visual elements meet professional standards:\n"
        "- FIGURE SIZING: Design figures to fit properly within page margins without stretching\n"
        "- TEXT LEGIBILITY: Ensure all figure text (labels, legends, annotations) is appropriately sized and positioned\n"
        "- TABLE FORMATTING: Create tables that fit page width with readable font sizes\n"
        "- CONSISTENT STYLING: Maintain uniform visual style across all figures and tables\n"
        "- MARGIN COMPLIANCE: Ensure no content extends beyond standard page margins\n"
        "- CAPTION QUALITY: Write informative captions that explain figures/tables completely\n"
        "- REFERENCE INTEGRATION: Ensure smooth integration between text and visual elements\n\n"
        
        "RESULTS DOCUMENTATION REQUIREMENTS:\n"
        "- Ensure simulation.py writes key numerical results to 'results.txt'\n"
        "- Include timestamps, parameter values, and computed metrics\n"
        "- Format results clearly for easy reference and verification\n"
        "- Results file should be human-readable and well-structured\n"
        "- All numbers cited in the paper must be traceable to results.txt\n\n"
        
        "FIGURE GENERATION REQUIREMENTS:\n"
        "- If paper includes computational figures/plots, include generation code in simulation.py\n"
        "- Save generated figures to the local project folder (not external dependencies)\n"
        "- Use standard formats (PNG, PDF, SVG) that can be referenced in LaTeX\n"
        "- Include proper figure generation with matplotlib, seaborn, or similar libraries\n"
        "- Ensure figure files are saved with descriptive names\n"
        "- Reference saved figures in LaTeX using \\includegraphics with proper paths\n"
        "- GRAPH DRAWING CONSTRAINTS: For any graph, chart, or plot generation, ensure at least 5 data samples are included and use appropriate sizing (minimum 6x4 inches or equivalent)\n"
        "- DATA ADEQUACY: Graphs must contain sufficient data points to demonstrate meaningful patterns and trends\n"
        "- VISUALIZATION QUALITY: Use proper axis labels, legends, grid lines, and appropriate scaling for clarity\n\n"
        
        "REFERENCE REQUIREMENTS:\n"
        "- Minimum 15-20 authentic, recently published references (prefer 2018-2025)\n"
        "- Include proper DOIs where available\n"
        "- Verify all author names, journal names, and publication details\n"
        "- References must be directly relevant to the research topic\n"
        "- Use proper citation style throughout the paper\n\n"
        
        "CONTENT SELF-CONTAINMENT:\n"
        "- Figures: Use TikZ, PGFPlots, or pure LaTeX constructs for diagrams; \\includegraphics for generated plots\n"
        "- Tables: Create with tabular environment, populate with simulation data from results.txt\n"
        "- Diagrams: Use TikZ or similar LaTeX-native tools\n"
        "- Generated plots: Include generation code in simulation.py, save locally, reference properly\n"
        "- ALL visual content must either be LaTeX-generated or traceable to simulation.py\n\n"
        
        "TABLE/FIGURE POSITIONING REQUIREMENTS:\n"
        "- Use contextual positioning: [h] (here if possible), [ht] (here or top of section), or [H] (force here)\n"
        "- NEVER use [t] or [!t] positioning which forces floats to page tops regardless of context\n"
        "- Place tables/figures in relevant subsections immediately after first text mention\n"
        "- Ensure visual self-containment with comprehensive, descriptive captions\n"
        "- Each float must appear in logical context, not forced to arbitrary page positions\n"
        "- Tables/figures should enhance understanding within their specific subsection context\n"
        "- CRITICAL: Place **all figures and tables either inline in the main text where they are first cited, or at the very end of the document but strictly before the references section**\n"
        "- **Do not place any figures or tables after the references or between references**\n"
        "- If LaTeX floats push figures out of order, use [H] positioning (requires \\usepackage{float}) to force placement where cited\n"
        "- For figures/tables at document end, place them immediately before \\bibliography{} or \\begin{thebibliography}\n"
        "- CRITICAL: Table formatting for large tables that exceed page borders:\n"
        "  * Split long tables into multiple parts using \\begin{longtable} or manual splitting\n"
        "  * Repeat table headers/keys in each part: 'Table X (continued)' or 'Table X (part 2 of 3)'\n"
        "  * Add '(continued)' text between table parts to maintain continuity\n"
        "  * Use \\multicolumn spanning for section breaks in large tables\n"
        "  * Ensure each table part is self-explanatory with repeated column headers\n"
        "  * Consider landscape orientation (\\begin{landscape}) for very wide tables\n\n"
        
        "STRUCTURE REQUIREMENTS:\n"
        "- Start with \\begin{filecontents*}{refs.bib}...\\end{filecontents*} containing ALL bibliography entries\n"
        "- Follow with standard LaTeX document structure\n"
        "- Organize sections according to paper type and field conventions\n"
        "- End with \\bibliography{refs} to use the embedded references\n\n"
        
        "FORMATTING REQUIREMENTS:\n"
        "- All content must use width=\\linewidth constraints\n"
        "- Wrap wide tables using adjustbox width=\\linewidth\n"
        "- NO CODE BLOCKS: NEVER use \\begin{lstlisting}, \\begin{verbatim}, \\begin{code}, or any code listing environments\n"
        "- ALGORITHMS ONLY: Use \\begin{algorithm}, \\begin{algorithmic}, or algorithm2e environments for pseudocode/algorithms\n"
        "- Replace any existing code blocks with proper algorithm pseudocode descriptions\n"
        "- NO PROMPT REPETITION: Do not repeat or reference the instructions/prompts given to you directly in the paper content\n"
        "- SINGLE CODE FILE: Consolidate ALL computational code into simulation.py only\n"
        "- Ensure all mathematical notation is properly formatted\n\n"
        
        "EXAMPLE STRUCTURE - ALL REFERENCES IN PAPER.TEX:\n"
        "\\begin{filecontents*}{refs.bib}\n"
        "@article{RealAuthor2024,\n"
        "  author = {A. Real and B. Author},\n"
        "  title = {Actual Published Title},\n"
        "  journal = {Nature Communications},\n"
        "  year = {2024},\n"
        "  volume = {15},\n"
        "  pages = {1234},\n"
        "  doi = {10.1038/s41467-024-xxxxx}\n"
        "}\n"
        "% Add all 15-20 references here in paper.tex - NO separate .bib file!\n"
        "\\end{filecontents*}\n"
        "\\documentclass{...}\n"
        "...\n"
        "\\begin{document}\n"
        "\\begin{tikzpicture} % For diagrams\n"
        "...\n"
        "\\end{tikzpicture}\n"
        "\\includegraphics[width=\\linewidth]{generated_plot.pdf} % For simulation-generated figures\n"
        "\\begin{tabular} % With real simulation data from results.txt\n"
        "...\n"
        "\\end{tabular}\n"
        "\\bibliography{refs} % References the embedded filecontents above\n"
        "\\end{document}\n"
        "\n"
        "CRITICAL: NO separate refs.bib files - everything in paper.tex!"
    )
    
    # Add custom user prompt if provided - it takes priority
    if user_prompt:
        sys_prompt = (
            f"PRIORITY INSTRUCTION FROM USER: {user_prompt}\n\n"
            "The above user instruction takes precedence over any conflicting requirements below. "
            "However, still maintain the critical technical requirements (single file, embedded references, compilable LaTeX).\n\n"
            + sys_prompt
        )
    
    user_content = f"Topic: {topic}\nField: {field}\nResearch Question: {question}\n\nDraft the COMPLETE self-contained LaTeX paper with authentic references, proper results documentation, and appropriate figure generation code."
    
    # Enhance with quality requirements if enabled
    if enable_quality_enhancements:
        try:
            paper_type = detect_paper_type(question, field, topic)
            
            # Apply experimental/theoretical rigor enhancements
            enhanced_prompt = enhance_prompt_with_rigor(sys_prompt, paper_type)
            
            # Apply review-driven enhancements
            enhanced_prompt = enhance_prompt_for_review_quality(enhanced_prompt, paper_type)
            sys_prompt = enhanced_prompt
            
            # Add paper-type-specific requirements to user content
            if paper_type == "experimental":
                user_content += f"\n\nSTATISTICAL RIGOR REQUIREMENTS:\n{get_statistical_rigor_requirements()}"
            elif paper_type == "theoretical":
                user_content += f"\n\nTHEORETICAL RIGOR REQUIREMENTS:\n{get_theoretical_rigor_requirements()}"
                
            # Add review-driven requirements
            review_requirements = get_review_driven_requirements(paper_type)
            user_content += f"\n\n🔬 CRITICAL FOR POSITIVE REVIEWS - EMPIRICAL VALIDATION:\n" + "\n".join([f"• {req}" for req in review_requirements["empirical_validation"][:3]])
            user_content += f"\n\n📊 MANDATORY FIGURE REQUIREMENTS (0 FIGURES = REJECTION):\n" + "\n".join([f"• {req}" for req in review_requirements["figure_requirements"][:3]])
            
        except Exception as e:
            # Continue with standard prompt if enhancement fails
            print(f"Warning: Quality enhancement failed: {e}")
    
    return [{"role": "system", "content": sys_prompt}, {"role": "user", "content": user_content}]

def _collect_project_files(project_dir: Path) -> str:
    """Collect all relevant files in the project directory for review context."""
    file_contents = []
    
    # Define file extensions and patterns to include
    include_patterns = [
        "*.py", "*.tex", "*.bib", "*.txt", "*.csv", "*.json", "*.md", "*.yml", "*.yaml", 
        "*.png", "*.log", "*.dat", "*.tsv", "*.xml", "*.html", "*.js", "*.cpp", "*.c", 
        "*.h", "*.hpp", "*.java", "*.scala", "*.r", "*.R", "*.m", "*.sh", "*.bat", 
        "*.cfg", "*.conf", "*.ini", "*.properties", "*.toml", "*.result", "*.output", 
        "*.summary", "*.stats", "*.metrics", "*.benchmark", "*.trace", "*.report"
    ]
    
    # Define files/directories to exclude
    exclude_patterns = [
        "__pycache__", "*.aux", "*.bbl", "*.blg", "*.out", "*.pdf", 
        "*.npy", "*.npz", "*.pkl", "*.cache", ".git", "node_modules", "*.pyc", 
        "*.pyo", "*.class", "*.o", "*.obj", "*.so", "*.dll", "*.exe"
    ]
    
    try:
        for pattern in include_patterns:
            for file_path in project_dir.glob(pattern):
                if file_path.is_file():
                    # Check if file should be excluded
                    should_exclude = any(
                        file_path.match(exclude_pattern) for exclude_pattern in exclude_patterns
                    )
                    if should_exclude:
                        continue
                    
                    try:
                        # Try to read as text file
                        with open(file_path, 'r', encoding='utf-8') as f:
                            content = f.read()
                        
                        # Limit file size to avoid overwhelming the context
                        if len(content) > 50000:  # 50KB limit per file
                            content = content[:50000] + "\n... (file truncated for length)"
                        
                        relative_path = file_path.relative_to(project_dir)
                        file_contents.append(f"=== FILE: {relative_path} ===\n{content}\n")
                        
                    except (UnicodeDecodeError, PermissionError):
                        # Skip binary or inaccessible files
                        continue
                        
    except Exception as e:
        print(f"Warning: Error collecting project files: {e}")
    
    if file_contents:
        return "\n".join(file_contents)
    else:
        return "No additional project files found."

def _combined_review_edit_revise_prompt(paper_tex: str, sim_summary: str, latex_errors: str = "", project_dir: Path = None, user_prompt: Optional[str] = None, iteration_count: int = 1, quality_issues: Optional[List[str]] = None) -> List[Dict[str, str]]:
    """Combined prompt for review and revision with diff output."""
    sys_prompt = (
        "You are a combined AI system acting as: (1) Top-tier journal reviewer and (2) Paper author. "
        "Your task is to review the paper and provide complete file diffs for all revisions needed to improve it.\n\n"
        
        "🔒 CRITICAL CONTENT PRESERVATION REQUIREMENTS:\n"
        "- NEVER delete entire sections, subsections, or substantial content blocks\n"
        "- PRESERVE the paper's core content, findings, and methodology\n"
        "- MAINTAIN or INCREASE the paper's word count and substance\n"
        "- When fixing issues, ADD content rather than DELETE existing content\n"
        "- If content needs restructuring, REARRANGE rather than REMOVE\n"
        "- PRESERVE all figures, tables, equations, and references\n"
        "- ONLY delete content if it's clearly redundant, incorrect, or harmful\n"
        "- When in doubt, preserve existing content and add improvements around it\n\n"
        
        "REVIEW APPROACH:\n"
        "You will be provided with both the LaTeX source code AND the rendered PDF version of the paper (if available). "
        "Use the PDF to assess the visual presentation, layout, figure placement, and overall appearance, "
        "while using the LaTeX source to verify technical requirements and code structure.\n\n"
        
        "📊 CRITICAL PDF VISUAL INSPECTION REQUIREMENTS:\n"
        "If a PDF is provided, perform THOROUGH visual analysis of the rendered document:\n"
        "- GRAPH/FIGURE SIZING: Check if graphs, charts, plots are properly sized and not stretching beyond page margins\n"
        "- TEXT POSITIONING: Verify all text elements (axis labels, legends, captions, annotations) are in correct positions\n"
        "- TABLE FORMATTING: Ensure tables fit within page width, text is readable, no content is cut off\n"
        "- FONT CONSISTENCY: Check that font sizes are appropriate and consistent throughout figures\n"
        "- VISUAL CLARITY: Verify graphs are not pixelated, blurry, or distorted\n"
        "- MARGIN COMPLIANCE: Ensure no content extends beyond page margins or overlaps with other elements\n"
        "- FIGURE REFERENCES: Confirm all figures are properly numbered and referenced in text\n"
        "- CAPTION ALIGNMENT: Check that captions are properly aligned and positioned relative to their figures/tables\n\n"
        
        "📋 PAPER STRUCTURE AND LAYOUT ANALYSIS:\n"
        "Perform comprehensive structural review of the paper organization:\n"
        "- SECTION ORDERING: Verify logical flow - Abstract, Introduction, Methods, Results, Discussion, Conclusion\n"
        "- GLOSSARY PLACEMENT: If present, glossary should be positioned appropriately (typically before references or as appendix)\n"
        "- APPENDIX ORGANIZATION: Consider moving large tables, detailed algorithms, or pseudocode to appendix for better readability\n"
        "- PSEUDOCODE PLACEMENT: Evaluate if complex algorithms should be moved to appendix while keeping simplified versions in main text\n"
        "- REFERENCE POSITIONING: Ensure references section is at the very end, properly formatted\n"
        "- SUBSECTION BALANCE: Check that sections are well-balanced in length and content depth\n"
        "- FLOATING ELEMENTS: Verify figures/tables appear near their first mention, not orphaned on separate pages\n"
        "- PAGE BREAKS: Ensure logical page breaks that don't split related content inappropriately\n"
        "- HEADER/FOOTER: Check for consistent and appropriate header/footer formatting\n"
        "- ACKNOWLEDGMENTS: Verify acknowledgments section is properly placed (typically before references)\n\n"
        
        "🎯 CRITICAL LATEX FLOAT POSITIONING REQUIREMENTS:\n"
        "MANDATORY LaTeX float positioning rules to ensure proper table/figure placement:\n"
        "- USE [htbp] POSITIONING: Replace ALL \\begin{figure}[H] and \\begin{table}[H] with [htbp] for better flow\n"
        "  * [H] forces exact placement and often creates poor page breaks and large white spaces\n"
        "  * [htbp] allows LaTeX to optimize placement: h=here, t=top, b=bottom, p=page\n"
        "- PROPER SPACING: Add \\vspace{0.5em} before and after tables/figures if needed for visual separation\n"
        "- CAPTION POSITIONING: Ensure captions are ABOVE tables and BELOW figures (standard convention)\n"
        "- REFERENCE IN TEXT: Every figure/table MUST be referenced in text using \\ref{} before it appears\n"
        "- CENTERED CONTENT: Use \\centering (not \\begin{center}) inside figure/table environments\n"
        "- AVOID FORCED POSITIONING: Never use [!h] or multiple [H] in sequence - let LaTeX optimize\n"
        "- LOGICAL ORDERING: Figures/tables should appear in the same order as referenced in text\n"
        "- SIZE OPTIMIZATION: Ensure tables fit page width using \\small, \\footnotesize, or \\resizebox if needed\n"
        "- BREAK WIDE TABLES: For tables wider than page, consider splitting or rotating (\\rotatebox{90})\n\n"
        
        "WORKFLOW STEPS:\n"
        "1. REVIEW: Conduct a thorough peer review meeting top journal standards\n"
        "2. REVISION: Provide complete file diffs for ALL files that need changes to address review issues\n\n"
        
        "REVIEW CRITERIA (same as top-tier journals):\n"
        "- Scientific rigor, methodology soundness, and novel contribution\n"
        "- Proper literature review with 15-20 authentic references\n"
        "- Clear research question, appropriate experimental design\n"
        "- Results interpretation, limitations acknowledgment\n"
        "- LaTeX compilation success and proper formatting\n"
        "- Self-contained visuals with proper size constraints\n"
        "- No filename references in paper text\n"
        "- Authentic references (no fake citations)\n"
        "- Single file structure with embedded references\n"
        "- Real simulation data usage (no fake numbers)\n"
        "- Reproducible results documentation\n"
        "- CRITICAL: Tables/figures positioned contextually in relevant subsections (NOT forced to page tops)\n\n"
        
        "REVISION OUTPUT FORMAT:\n"
        "Always provide complete revised file contents in this exact format:\n\n"
        "```tex\n"
        "# File: paper.tex\n"
        "[Complete revised LaTeX content here]\n"
        "```\n\n"
        "```python\n"
        "# File: simulation.py\n"
        "[Complete revised Python code here]\n"
        "```\n\n"
        "For each file that needs changes, provide the COMPLETE file content (not just diffs).\n"
        "This ensures all changes are applied correctly without parsing errors.\n\n"
        
        "IMPORTANT: Your changes will be displayed in git diff format in the terminal. "
        "Make SUBSTANTIAL and MEANINGFUL changes that address the review issues. "
        "Avoid making cosmetic-only changes - focus on content improvements that will "
        "significantly enhance the paper's quality and fix identified problems.\n\n"
        
        "CRITICAL REVISION REQUIREMENTS:\n"
        "- Address ALL review concerns completely\n"
        "- Fix LaTeX compilation errors if any\n"
        "- Use only authentic references (no fake citations)\n"
        "- Ensure single file structure with embedded references\n"
        "- Apply proper size constraints to all visuals\n"
        "- Remove filename references from paper text\n"
        "- Use only real simulation data\n"
        "- Maintain paper structure appropriate for field\n"
        "- NO CODE BLOCKS: NEVER use \\begin{lstlisting}, \\begin{verbatim}, \\begin{code}, or any code listing environments\n"
        "- ALGORITHMS ONLY: Use \\begin{algorithm}, \\begin{algorithmic}, or algorithm2e environments for pseudocode/algorithms\n"
        "- Replace any existing code blocks with proper algorithm pseudocode descriptions\n"
        "- Include all necessary files in diffs (paper.tex, simulation.py, etc.)\n"
        "- ESSENTIAL: Fix table/figure positioning using contextual placement:\n"
        "  * Use [h] (here if possible), [ht] (here or top of section), or [H] (force here) for contextual positioning\n"
        "  * AVOID [t] and [!t] positioning which forces floats to page tops regardless of context\n"
        "  * Place tables/figures in relevant subsections after first text mention\n"
        "  * Ensure visual self-containment with descriptive captions\n"
        "  * Verify each float appears in logical context, not forced to random page positions\n"
        "- CRITICAL: Table formatting for large tables that exceed page borders:\n"
        "  * Split long tables into multiple parts using \\begin{longtable} or manual splitting\n"
        "  * Repeat table headers/keys in each part: 'Table X (continued)' or 'Table X (part 2 of 3)'\n"
        "  * Add '(continued)' text between table parts to maintain continuity\n"
        "  * Use \\multicolumn spanning for section breaks in large tables\n"
        "  * Ensure each table part is self-explanatory with repeated column headers\n"
        "  * Consider landscape orientation (\\begin{landscape}) for very wide tables\n\n"
    )
    
    # Add custom user prompt if provided
    if user_prompt:
        sys_prompt = (
            f"PRIORITY INSTRUCTION FROM USER: {user_prompt}\n\n"
            "The above user instruction takes precedence when evaluating and revising the paper. "
            "However, still maintain the critical technical requirements.\n\n"
            + sys_prompt
        )
    
    # Collect all project files for complete context
    project_files_content = ""
    if project_dir and project_dir.exists():
        project_files_content = _collect_project_files(project_dir)
    
    user = (
        f"This is iteration {iteration_count}. Please complete the 2-step workflow:\n\n"
        "STEP 1: REVIEW\n"
        "Conduct a thorough peer review of the paper using top journal standards.\n\n"
        "STEP 2: REVISION\n"
        "Provide complete file diffs for all necessary changes to address the review issues.\n\n"
        "----- CURRENT PAPER (LATEX) -----\n" + paper_tex + "\n"
        "----- SIMULATION CODE & OUTPUTS -----\n" + sim_summary + "\n"
        "----- ALL PROJECT FILES (FOR CONTEXT) -----\n" + project_files_content + "\n"
    )
    
    # Add quality issues if detected
    if quality_issues:
        user += (
            "\n----- DETECTED QUALITY ISSUES -----\n"
            "The following specific quality issues have been automatically detected and MUST be addressed:\n\n"
        )
        for issue in quality_issues:
            user += f"• {issue}\n"
        user += (
            "\n----- END QUALITY ISSUES -----\n\n"
            "CRITICAL: Your revision MUST specifically address ALL of the above quality issues. "
            "These are not suggestions - they are required fixes that must be implemented.\n"
        )
    
    # Add LaTeX compilation information
    if latex_errors:
        user += (
            "\n----- LATEX COMPILATION ERRORS (LAST 20 LINES OF .log) -----\n" + 
            latex_errors + 
            "\n----- END LATEX ERRORS -----\n\n"
            "CRITICAL: Fix ALL LaTeX compilation errors in your revision diffs.\n"
        )
    else:
        user += (
            "\n----- LATEX COMPILATION STATUS -----\n" +
            "Previous compilation was SUCCESSFUL. No errors detected.\n" +
            "----- END COMPILATION STATUS -----\n\n"
        )
    
    user += (
        "\nProvide your response in this format:\n\n"
        "## REVIEW\n"
        "[Your detailed review here]\n\n"
        "## REVISION DIFFS\n"
        "[Complete revised file contents for all files that need changes]\n"
    )
    
    return [{"role": "system", "content": sys_prompt}, {"role": "user", "content": user}]

def _parse_combined_response(response: str, project_dir: Path) -> tuple[str, str, dict]:
    """
    Parse the combined review/revision response.
    
    Returns:
        (review_text, "REVISE", file_changes)
        where file_changes is a dict {filename: new_content}
    """
    import re
    
    # Extract sections using regex
    review_match = re.search(r'## REVIEW\s*\n(.*?)(?=## REVISION DIFFS)', response, re.DOTALL | re.IGNORECASE)
    diffs_match = re.search(r'## REVISION DIFFS.*?\n(.*)', response, re.DOTALL | re.IGNORECASE)
    
    review_text = review_match.group(1).strip() if review_match else "No review section found"
    diffs_text = diffs_match.group(1).strip() if diffs_match else ""
    
    # Parse diffs and extract new file contents
    file_changes = {}
    
    if diffs_text:
        print(f"DEBUG: Found revision section with {len(diffs_text)} characters")
        
        # Method 1: Look for complete file contents in code blocks with file comments
        file_blocks = re.findall(r'```(?:latex|tex|python|py|txt)?\s*\n# ?(?:File: ?|Filename: ?)?([^\n]+)\n(.*?)\n```', diffs_text, re.DOTALL)
        for filename, content in file_blocks:
            if filename and filename.strip():
                filename = filename.strip()
                if filename.endswith(':'):
                    filename = filename[:-1]  # Remove trailing colon
                file_changes[filename] = content.strip()
                print(f"DEBUG: Extracted {filename} ({len(content)} chars) from code block with file comment")
        
        # Method 2: Look for code blocks without file comments but with language hints
        if not file_changes:
            # Try to match by file extension pattern
            tex_blocks = re.findall(r'```(?:latex|tex)\s*\n(.*?)\n```', diffs_text, re.DOTALL)
            if tex_blocks:
                file_changes['paper.tex'] = tex_blocks[0].strip()
                print(f"DEBUG: Extracted paper.tex ({len(tex_blocks[0])} chars) from tex code block")
            
            py_blocks = re.findall(r'```(?:python|py)\s*\n(.*?)\n```', diffs_text, re.DOTALL)
            if py_blocks:
                file_changes['simulation.py'] = py_blocks[0].strip()
                print(f"DEBUG: Extracted simulation.py ({len(py_blocks[0])} chars) from python code block")
        
        # Method 3: Look for explicit file sections (File: filename)
        file_sections = re.findall(r'(?:File: ?|Filename: ?)([^\n]+)\n```[^\n]*\n(.*?)\n```', diffs_text, re.DOTALL)
        for filename, content in file_sections:
            filename = filename.strip()
            if filename.endswith(':'):
                filename = filename[:-1]  # Remove trailing colon
            if filename not in file_changes:  # Don't override if already found
                file_changes[filename] = content.strip()
                print(f"DEBUG: Extracted {filename} ({len(content)} chars) from file section")
        
        # Method 4: Look for files mentioned by name with content following
        file_mentions = re.findall(r'([a-zA-Z_][a-zA-Z0-9_]*\.[a-zA-Z]+):\s*\n```[^\n]*\n(.*?)\n```', diffs_text, re.DOTALL)
        for filename, content in file_mentions:
            if filename not in file_changes:  # Don't override if already found
                file_changes[filename] = content.strip()
                print(f"DEBUG: Extracted {filename} ({len(content)} chars) from file mention")
        
        print(f"DEBUG: Total files extracted: {list(file_changes.keys())}")
    else:
        print(f"DEBUG: No revision section found in response")
    
    # Always return "REVISE" as the decision since we removed editorial logic
    return review_text, "REVISE", file_changes

def _apply_file_changes(file_changes: dict, project_dir: Path, config=None) -> bool:
    """
    Apply file changes from the revision response with content protection.
    
    Returns:
        bool: True if changes were applied, False if rejected by content protection
    """
    # Initialize content protector
    protector = ContentProtector(project_dir)
    
    # Get content protection settings from config
    enable_protection = getattr(config, 'content_protection',
                                getattr(config, 'enable_content_protection', True)) if config else True
    auto_approve = getattr(config, 'auto_approve_changes',
                           getattr(config, 'auto_approve_safe_changes', False)) if config else False
    
    for filename, content in file_changes.items():
        file_path = project_dir / filename
        
        # Only apply content protection to LaTeX files if enabled
        if filename.endswith('.tex') and enable_protection:
            try:
                # Read existing content for comparison
                original_content = ""
                if file_path.exists():
                    original_content = file_path.read_text(encoding='utf-8', errors='ignore')
                
                # Create backup before making changes
                if original_content:
                    backup_path = protector.create_backup(file_path, f"{filename}_pre_revision_{datetime.now().strftime('%H%M%S')}")
                
                # Validate the revision using content protection
                if isinstance(content, list):
                    content = '\n'.join(content)
                
                approved, analysis = protector.validate_revision(original_content, content, auto_approve)
                
                if not approved:
                    print(f"❌ Revision rejected by content protection for {filename}")
                    print("   Changes not applied. Original content preserved.")
                    return False
                
                # Apply the changes
                file_path.parent.mkdir(parents=True, exist_ok=True)
                file_path.write_text(content, encoding='utf-8')
                
                # Log the successful change
                change_percent = analysis.word_count_change_percent
                print(f"✓ Updated {filename}: {analysis.old_metrics.word_count:,} → {analysis.new_metrics.word_count:,} words ({change_percent:+.1f}%)")
                
                if analysis.warnings:
                    print(f"  Warnings (approved): {'; '.join(analysis.warnings[:2])}")
                
            except Exception as e:
                print(f"❌ Failed to update {filename}: {e}")
                return False
        elif filename.endswith('.tex'):
            # LaTeX file with content protection DISABLED
            try:
                file_path.parent.mkdir(parents=True, exist_ok=True)
                
                if isinstance(content, list):
                    content = '\n'.join(content)
                
                # Show warning about disabled protection
                print(f"⚠ Updating {filename} WITHOUT content protection (DANGEROUS)")
                
                file_path.write_text(content, encoding='utf-8')
                print(f"✓ Updated {filename} (content protection disabled)")
                
            except Exception as e:
                print(f"❌ Failed to update {filename}: {e}")
                return False
        else:
            # For non-LaTeX files, apply changes directly (no content protection needed)
            try:
                file_path.parent.mkdir(parents=True, exist_ok=True)
                
                if isinstance(content, list):
                    content = '\n'.join(content)
                
                file_path.write_text(content, encoding='utf-8')
                print(f"✓ Updated {filename} (no content protection applied)")
                
            except Exception as e:
                print(f"❌ Failed to update {filename}: {e}")
                return False
    
    return True

def _review_prompt(paper_tex: str, sim_summary: str, project_dir: Path = None, user_prompt: Optional[str] = None, enable_quality_enhancements: bool = True) -> List[Dict[str, str]]:
    sys_prompt = (
        "Act as a top-tier journal reviewer (Nature, Science, Cell level) with expertise in LaTeX formatting and scientific programming. "
        "Your review must meet the highest academic standards. Be constructive but demanding. "
        "CRITICAL: If the simulation ran successfully and produced actual results, the paper MUST use these real numbers, not fake/placeholder values. "
        
        "REVIEW APPROACH:\n"
        "You will be provided with both the LaTeX source code AND the rendered PDF version of the paper. "
        "Use the PDF to assess the visual presentation, layout, figure placement, and overall appearance, "
        "while using the LaTeX source to verify technical requirements and code structure.\n\n"
        
        "📊 CRITICAL PDF VISUAL INSPECTION REQUIREMENTS:\n"
        "If a PDF is provided, perform THOROUGH visual analysis of the rendered document:\n"
        "- GRAPH/FIGURE SIZING: Check if graphs, charts, plots are properly sized and not stretching beyond page margins\n"
        "- TEXT POSITIONING: Verify all text elements (axis labels, legends, captions, annotations) are in correct positions\n"
        "- TABLE FORMATTING: Ensure tables fit within page width, text is readable, no content is cut off\n"
        "- FONT CONSISTENCY: Check that font sizes are appropriate and consistent throughout figures\n"
        "- VISUAL CLARITY: Verify graphs are not pixelated, blurry, or distorted\n"
        "- MARGIN COMPLIANCE: Ensure no content extends beyond page margins or overlaps with other elements\n"
        "- FIGURE REFERENCES: Confirm all figures are properly numbered and referenced in text\n"
        "- CAPTION ALIGNMENT: Check that captions are properly aligned and positioned relative to their figures/tables\n\n"
        
        "📋 PAPER STRUCTURE AND LAYOUT ANALYSIS:\n"
        "Perform comprehensive structural review of the paper organization:\n"
        "- SECTION ORDERING: Verify logical flow - Abstract, Introduction, Methods, Results, Discussion, Conclusion\n"
        "- GLOSSARY PLACEMENT: If present, glossary should be positioned appropriately (typically before references or as appendix)\n"
        "- APPENDIX ORGANIZATION: Consider if large tables, detailed algorithms, or pseudocode should be moved to appendix for better readability\n"
        "- PSEUDOCODE PLACEMENT: Evaluate if complex algorithms should be moved to appendix while keeping simplified versions in main text\n"
        "- REFERENCE POSITIONING: Ensure references section is at the very end, properly formatted\n"
        "- SUBSECTION BALANCE: Check that sections are well-balanced in length and content depth\n"
        "- FLOATING ELEMENTS: Verify figures/tables appear near their first mention, not orphaned on separate pages\n"
        "- PAGE BREAKS: Ensure logical page breaks that don't split related content inappropriately\n"
        "- HEADER/FOOTER: Check for consistent and appropriate header/footer formatting\n"
        "- ACKNOWLEDGMENTS: Verify acknowledgments section is properly placed (typically before references)\n\n"
        
        "🎯 CRITICAL LATEX FLOAT POSITIONING REQUIREMENTS:\n"
        "MANDATORY LaTeX float positioning rules to ensure proper table/figure placement:\n"
        "- USE [htbp] POSITIONING: Replace ALL \\begin{figure}[H] and \\begin{table}[H] with [htbp] for better flow\n"
        "  * [H] forces exact placement and often creates poor page breaks and large white spaces\n"
        "  * [htbp] allows LaTeX to optimize placement: h=here, t=top, b=bottom, p=page\n"
        "- PROPER SPACING: Add \\vspace{0.5em} before and after tables/figures if needed for visual separation\n"
        "- CAPTION POSITIONING: Ensure captions are ABOVE tables and BELOW figures (standard convention)\n"
        "- REFERENCE IN TEXT: Every figure/table MUST be referenced in text using \\ref{} before it appears\n"
        "- CENTERED CONTENT: Use \\centering (not \\begin{center}) inside figure/table environments\n"
        "- AVOID FORCED POSITIONING: Never use [!h] or multiple [H] in sequence - let LaTeX optimize\n"
        "- LOGICAL ORDERING: Figures/tables should appear in the same order as referenced in text\n"
        "- SIZE OPTIMIZATION: Ensure tables fit page width using \\small, \\footnotesize, or \\resizebox if needed\n"
        "- BREAK WIDE TABLES: For tables wider than page, consider splitting or rotating (\\rotatebox{90})\n\n"
        
        "MANDATORY REQUIREMENTS - CHECK CAREFULLY:\n"
        "1. SINGLE FILE ONLY: Paper must be ONE LaTeX file with NO \\input{} or \\include{} commands\n"
        "2. EMBEDDED REFERENCES MANDATORY: References MUST be embedded directly in the paper.tex file using EITHER:\n"
        "   - \\begin{filecontents*}{refs.bib}...\\end{filecontents*} at the TOP of the file, OR\n"
        "   - \\begin{thebibliography}...\\end{thebibliography} at the END of the file\n"
        "   - NO separate refs.bib files are allowed - everything must be in paper.tex\n"
        "   - Verify \\bibliography{refs} command uses the embedded filecontents, not external files\n"
        "3. COMPILABLE: File must compile successfully with pdflatex (check for syntax errors, missing packages, etc.)\n"
        "4. NO FILENAMES IN TEXT: The paper text must NOT contain ANY references to uploaded filenames, code files, data files, or directory structures. Examples to avoid:\n"
        "   - 'simulation.py', 'results.txt', 'data.csv', 'model.pkl', etc.\n"
        "   - Directory paths like 'output/', 'src/', 'data/', etc.\n"
        "   - File extensions like '.py', '.txt', '.csv', '.pkl', etc.\n"
        "   - The paper should describe methods and results without revealing the underlying file structure\n"
        "   - Use generic descriptions like 'our implementation', 'experimental results', 'computational analysis'\n"
        "5. AUTHENTIC REFERENCES MANDATORY: ALL references must be real, published works with correct bibliographic details. Verify:\n"
        "   - Minimum 15-20 authentic references from reputable sources\n"
        "   - Author names are real and spelled correctly\n"
        "   - Journal/venue names are authentic\n"
        "   - Publication years are realistic\n"
        "   - DOIs are properly formatted (if provided)\n"
        "   - References are directly relevant to the topic\n"
        "   - NO placeholder, fake, or made-up citations\n"
        "   - All references must be cited in the text using \\cite{} commands\n"
        "6. SELF-CONTAINED VISUALS: ALL tables, figures, and diagrams must be:\n"
        "   - Defined within the LaTeX file using TikZ, tabular, PGFPlots, etc., OR\n"
        "   - Generated by simulation.py and saved as local files with proper \\includegraphics references\n"
        "   - Populated with REAL data from simulation results\n"
        "   - NO fake, placeholder, or estimated numbers\n"
        "7. APPROPRIATE STRUCTURE: The paper structure must align with the paper type and field conventions:\n"
        "   - Verify section organization matches the paper's contribution type\n"
        "   - Check for field-specific sections and evaluation methodologies\n"
        "   - Ensure logical flow appropriate for the research area\n"
        "   - Validate that section names and content align with journal standards\n"
        "8. RESULTS DOCUMENTATION: Check that numerical results are properly documented:\n"
        "   - Simulation.py should save key results to 'results.txt' for reproducibility\n"
        "   - All numerical values in the paper must be traceable to simulation output\n"
        "   - Results file should be well-structured and human-readable\n"
        "   - Verify consistency between cited numbers and simulation output\n"
        "9. FIGURE GENERATION: If paper includes computational figures:\n"
        "   - Figure generation code must be included in simulation.py\n"
        "   - Generated figures must be saved in the local project folder\n"
        "   - LaTeX must reference generated figures with proper \\includegraphics paths\n"
        "   - No external figure dependencies or missing image files\n"
        "   - GRAPH CONSTRAINTS: Verify graphs contain at least 5 data samples and use appropriate sizing\n"
        "   - DATA VISUALIZATION QUALITY: Check for proper axis labels, legends, and meaningful data representation\n"
        "   - FIGURE POSITIONING: Figures and graphs must be positioned appropriately within the paper - they CANNOT be placed between references/bibliography sections\n"
        "10. TABLE/FIGURE CONTEXTUAL POSITIONING: Tables and figures must be positioned in their relevant subsections, not forced to page tops:\n"
        "   - Use [h] for 'here if possible', [ht] for 'here or top of section', [H] for 'exactly here'\n"
        "   - Avoid [t] or [!t] positioning that forces floats to page tops regardless of context\n"
        "   - Tables/figures should appear after their first mention in text within the same subsection\n"
        "   - Keep floats contextually relevant to their surrounding discussion\n"
        "   - No orphaned tables appearing in unrelated sections or far from their text references\n"
        "   - Ensure logical reading flow where visuals enhance their specific subsection content\n"
        "11. FIGURE AND TABLE PLACEMENT REQUIREMENTS (CRITICAL): Check that all figures and tables are placed correctly:\n"
        "   - **All figures and tables must be either inline in the main text where they are first cited, or at the very end of the document but strictly before the references section**\n"
        "   - **No figures or tables should appear after the references or between references**\n"
        "   - If LaTeX floats push figures out of order, they should use [H] positioning (from float package) to force placement where cited\n"
        "   - Verify that the float package is loaded when [H] positioning is used: \\usepackage{float}\n"
        "   - For end-of-document placement, figures/tables should appear immediately before \\bibliography{} or \\begin{thebibliography}\n"
        "   - Flag any figures or tables that appear in or after the reference section as a major violation\n"
        "12. SINGLE CODE FILE: ALL computational code must be consolidated into ONE simulation.py file only - no additional .py files or scripts.\n\n"
        
        "STRUCTURE ALIGNMENT CRITERIA:\n"
        "- Theoretical papers should emphasize theory, proofs, and mathematical analysis\n"
        "- Experimental papers should have clear methodology, controlled experiments, and statistical analysis\n"
        "- Survey papers should provide comprehensive coverage, classification, and comparative analysis\n"
        "- Systems papers should detail architecture, implementation, and performance evaluation\n"
        "- Algorithm papers should include complexity analysis, correctness proofs, and empirical evaluation\n"
        "- Security papers should include threat models, security analysis, and attack/defense scenarios\n"
        "- Medical papers should follow clinical research standards with appropriate validation\n"
        "- Check that evaluation methodology matches the paper type and claims\n\n"
        
        "RESULTS AND FIGURE GENERATION CRITERIA:\n"
        "- Verify that simulation.py produces 'results.txt' with documented numerical outcomes\n"
        "- Check that all figures are either LaTeX-native (TikZ/PGFPlots) or generated by simulation code\n"
        "- Ensure figure files are saved locally and properly referenced in LaTeX\n"
        "- Validate that numerical values in tables/text match simulation outputs exactly\n"
        "- Confirm reproducibility through proper documentation and code organization\n\n"
        
        "REFERENCE QUALITY CRITERIA:\n"
        "- Minimum 15-20 authentic references from reputable sources\n"
        "- Recent publications (prefer 2018-2025) mixed with foundational works\n"
        "- Proper citation usage throughout the paper\n"
        "- References must support claims made in the text\n"
        "- Check for citation format consistency\n\n"
        
        "CONTENT SELF-CONTAINMENT CRITERIA:\n"
        "- All figures created with LaTeX code (TikZ, PGFPlots, etc.) OR generated by simulation.py\n"
        "- All tables use tabular environment with simulation data from results.txt\n"
        "- All diagrams use LaTeX-native drawing tools\n"
        "- Numbers in tables/figures match simulation outputs exactly\n"
        "- NO references to external files except those generated by simulation.py\n"
        "- GRAPH QUALITY STANDARDS: Generated graphs must contain at least 5 data samples with appropriate sizing and clear visualization\n\n"
        
        "CONTENT QUALITY CRITERIA:\n"
        "- Scientific rigor and methodology soundness\n"
        "- Novel contribution and significance to the field\n"
        "- Proper literature review and citation of related work\n"
        "- Clear research question and hypothesis\n"
        "- Appropriate experimental design and statistical analysis\n"
        "- Results interpretation and discussion quality\n"
        "- Limitations and future work acknowledgment\n"
        "- Reproducibility of results and code\n"
        
        "LATEX FORMATTING CRITERIA (CRITICAL FOR PAGE LAYOUT):\n"
        "- Proper document structure and section organization\n"
        "- Content width constraints (MUST use width=\\linewidth, never exceed page margins)\n"
        "- Table formatting (ALL wide tables MUST use adjustbox with width=\\linewidth)\n"
        "- TikZ diagrams MUST be constrained (use adjustbox or scale to fit page width)\n"
        "- Mathematical notation and equation formatting\n"
        "- NO CODE BLOCKS: NEVER use \\begin{lstlisting}, \\begin{verbatim}, \\begin{code}, or any code listing environments\n"
        "- ALGORITHMS ONLY: Use \\begin{algorithm}, \\begin{algorithmic}, or algorithm2e environments for pseudocode/algorithms\n"
        "- Replace any existing code blocks with proper algorithm pseudocode descriptions\n"
        "- NO PROMPT REPETITION: Do not repeat or reference the instructions/prompts given to you directly in the paper content\n"
        "- SINGLE CODE FILE: ALL computational code must be consolidated into simulation.py only\n"
        "- Citation style and bibliography completeness\n"
        "- Package usage and compatibility\n"
        "- Caption quality and cross-referencing\n"
        "- NO CONTENT should extend beyond vertical page borders\n"
        
        "SIMULATION CRITERIA:\n"
        "- Code quality, efficiency, and documentation\n"
        "- Algorithm correctness and implementation\n"
        "- Parameter choices and justification\n"
        "- Results alignment with simulation outputs - VERIFY numbers match exactly\n"
        "- Statistical significance and error analysis\n"
        "- Visualization quality and clarity\n"
        
        "CRITICAL REQUIREMENTS:\n"
        "- NO FAKE NUMBERS: All numerical results must come from actual simulation output\n"
        "- NO FAKE REFERENCES: All citations must be authentic, published works\n"
        "- NO FILENAMES: The paper must not mention specific filenames, file extensions, or directory structures\n"
        "- DOCUMENTED RESULTS: Key findings must be saved in results.txt\n"
        "- GENERATED FIGURES: Computational figures must be created by simulation.py\n"
        "- NO OVERFLOW: All content must fit within page margins\n"
        "- REPRODUCIBILITY: Code should be well-documented and runnable\n"
        "- SIGNIFICANCE: Work must make a meaningful contribution to the field\n"
        "- SELF-CONTAINED: Single file with embedded references and traceable visual content\n"
        "- APPROPRIATE STRUCTURE: Paper organization must match the research type and field standards\n"
        "- FIGURE/TABLE PLACEMENT: All figures and tables must be placed either inline where cited or at document end before references - NEVER after or between references\n"
        
        "Provide specific, actionable feedback with concrete suggestions for improvement. "
        "If the paper violates any of the 12 mandatory requirements, mark it as needing major revision. "
        "Pay special attention to reference authenticity, results documentation, figure generation, filename removal, structural appropriateness, and figure/table placement relative to references."
    )
    
    # Add custom user prompt if provided - it takes priority
    if user_prompt:
        sys_prompt = (
            f"PRIORITY INSTRUCTION FROM USER: {user_prompt}\n\n"
            "The above user instruction takes precedence when evaluating the paper. "
            "However, still maintain the critical technical requirements (single file, embedded references, etc.).\n\n"
            + sys_prompt
        )
    
    # Collect all project files for complete context
    project_files_content = ""
    if project_dir and project_dir.exists():
        project_files_content = _collect_project_files(project_dir)
    
    user = (
        "Here is the current paper (LaTeX):\n\n"
        "----- BEGIN LATEX -----\n" + paper_tex + "\n----- END LATEX -----\n\n"
        "Here are the actual simulation results - the paper MUST use these real numbers:\n\n"
        "----- BEGIN SIMULATION & OUTPUTS -----\n" + sim_summary + "\n----- END SIMULATION & OUTPUTS -----\n\n"
        "Here are ALL project files for complete context:\n\n"
        "----- BEGIN PROJECT FILES -----\n" + project_files_content + "\n----- END PROJECT FILES -----\n\n"
        "Write a constructive review. Check that ALL figures/tables/diagrams use proper size constraints to prevent page overflow. "
        "If the simulation produced valid results, ensure the paper uses ONLY these actual numbers, not fake values. "
        "Review the complete project context including all code, data, and supporting files."
    )
    return [{"role": "system", "content": sys_prompt}, {"role": "user", "content": user}]

def _editor_prompt(review_text: str, iteration_count: int, user_prompt: Optional[str] = None) -> List[Dict[str, str]]:
    sys_prompt = (
        "You are the handling editor of a top-tier journal. Make an informed decision about paper readiness. "
        "Consider both the reviewer's feedback and the iteration history. "
        "Papers should only be accepted when they meet publication standards for impact, rigor, and clarity."
    )
    
    # Add custom user prompt if provided - it takes priority
    if user_prompt:
        sys_prompt = (
            f"PRIORITY INSTRUCTION FROM USER: {user_prompt}\n\n"
            "The above user instruction should guide your revision approach. "
            "Balance user preferences with publication standards.\n\n"
            + sys_prompt
        )
    
    user = (
        f"This is iteration {iteration_count}. Review the feedback below and decide:\n"
        "- Answer 'YES' ONLY if the paper meets top journal standards and is ready for publication\n"
        "- Answer 'NO' if significant issues remain that require revision\n"
        "- Answer 'REJECT' if fundamental flaws make the paper unsuitable\n\n"
        "----- REVIEWER FEEDBACK -----\n" + review_text + "\n----- END FEEDBACK -----\n\n"
        "Decision (YES/NO/REJECT):"
    )
    return [{"role": "system", "content": sys_prompt}, {"role": "user", "content": user}]

def _compile_latex_and_get_errors(paper_path: Path, timeout: int = 120) -> Tuple[bool, str]:
    """Compile LaTeX file and return success status and error log."""
    try:
        # Run pdflatex with nonstopmode
        result = subprocess.run(
            ["pdflatex", "-interaction=nonstopmode", paper_path.name],
            cwd=paper_path.parent,
            capture_output=True,
            text=True,
            timeout=timeout  # Use dynamic timeout parameter
        )
        
        # Check if PDF was generated
        pdf_path = paper_path.with_suffix('.pdf')
        success = pdf_path.exists()
        
        # Get last 20 lines of log file
        log_path = paper_path.with_suffix('.log')
        error_log = ""
        if log_path.exists():
            try:
                with open(log_path, 'r', encoding='utf-8', errors='ignore') as f:
                    lines = f.readlines()
                    error_log = ''.join(lines[-20:])  # Last 20 lines
            except Exception:
                error_log = "Could not read log file"
        
        return success, error_log
        
    except subprocess.TimeoutExpired:
        return False, f"LaTeX compilation timed out after {timeout} seconds"
    except FileNotFoundError:
        return False, "pdflatex command not found. Please install a LaTeX distribution."
    except Exception as e:
        return False, f"LaTeX compilation error: {str(e)}"

def _generate_pdf_for_review(paper_path: Path, timeout: int = 120) -> Tuple[bool, Optional[Path], str]:
    """
    Generate PDF from LaTeX file specifically for AI review.
    
    Args:
        paper_path: Path to the LaTeX file
        timeout: Compilation timeout in seconds
        
    Returns:
        Tuple of (success, pdf_path, error_message)
    """
    try:
        print(f"Generating PDF for AI review from {paper_path.name}...")
        
        # Run pdflatex with nonstopmode
        result = subprocess.run(
            ["pdflatex", "-interaction=nonstopmode", paper_path.name],
            cwd=paper_path.parent,
            capture_output=True,
            text=True,
            timeout=timeout
        )
        
        # Check if PDF was generated
        pdf_path = paper_path.with_suffix('.pdf')
        
        if pdf_path.exists():
            print(f" PDF generated successfully: {pdf_path.name}")
            return True, pdf_path, ""
        else:
            # Get error details from log
            log_path = paper_path.with_suffix('.log')
            error_msg = "PDF generation failed."
            if log_path.exists():
                try:
                    with open(log_path, 'r', encoding='utf-8', errors='ignore') as f:
                        lines = f.readlines()
                        error_msg += f" Log (last 10 lines):\n{''.join(lines[-10:])}"
                except Exception:
                    error_msg += " Could not read log file."
            
            print(f" PDF generation failed: {error_msg}")
            return False, None, error_msg
        
    except subprocess.TimeoutExpired:
        error_msg = f"PDF generation timed out after {timeout} seconds"
        print(f" {error_msg}")
        return False, None, error_msg
    except FileNotFoundError:
        error_msg = "pdflatex command not found. Please install a LaTeX distribution."
        print(f" {error_msg}")
        return False, None, error_msg
    except Exception as e:
        error_msg = f"PDF generation error: {str(e)}"
        print(f" {error_msg}")
        return False, None, error_msg

def _revise_prompt(paper_tex: str, sim_summary: str, review_text: str, latex_errors: str = "", project_dir: Path = None, user_prompt: Optional[str] = None, quality_issues: Optional[List[str]] = None, enable_quality_enhancements: bool = True) -> List[Dict[str, str]]:
    sys_prompt = (
        "You are the paper author making revisions based on peer review. Your goal is to address ALL reviewer concerns "
        "while maintaining scientific integrity and clarity. Produce a COMPLETE revised LaTeX file.\n\n"
        
        "🔒 CRITICAL CONTENT PRESERVATION REQUIREMENTS:\n"
        "- NEVER delete entire sections, subsections, or substantial content blocks\n"
        "- PRESERVE the paper's core content, findings, and methodology\n"
        "- MAINTAIN or INCREASE the paper's word count and substance\n"
        "- When fixing issues, ADD content rather than DELETE existing content\n"
        "- If content needs restructuring, REARRANGE rather than REMOVE\n"
        "- PRESERVE all figures, tables, equations, and references\n"
        "- ONLY delete content if it's clearly redundant, incorrect, or harmful\n"
        "- When addressing reviewer concerns, ADD explanations rather than REMOVE existing content\n"
        "- If sections need improvement, ENHANCE them rather than DELETE them\n"
        "- MAINTAIN the paper's comprehensive nature and academic depth\n\n"
        
        "REVISION APPROACH:\n"
        "You will be provided with the LaTeX source code, the rendered PDF version (if available), and the reviewer feedback. "
        "Use the PDF to understand how the paper currently appears when rendered, including figure placement, table formatting, "
        "and overall visual layout. Address any visual presentation issues mentioned in the review.\n\n"
        
        "📊 CRITICAL PDF VISUAL INSPECTION REQUIREMENTS:\n"
        "If a PDF is provided, perform THOROUGH visual analysis of the rendered document:\n"
        "- GRAPH/FIGURE SIZING: Check if graphs, charts, plots are properly sized and not stretching beyond page margins\n"
        "- TEXT POSITIONING: Verify all text elements (axis labels, legends, captions, annotations) are in correct positions\n"
        "- TABLE FORMATTING: Ensure tables fit within page width, text is readable, no content is cut off\n"
        "- FONT CONSISTENCY: Check that font sizes are appropriate and consistent throughout figures\n"
        "- VISUAL CLARITY: Verify graphs are not pixelated, blurry, or distorted\n"
        "- MARGIN COMPLIANCE: Ensure no content extends beyond page margins or overlaps with other elements\n"
        "- FIGURE REFERENCES: Confirm all figures are properly numbered and referenced in text\n"
        "- CAPTION ALIGNMENT: Check that captions are properly aligned and positioned relative to their figures/tables\n\n"
        
        "📋 PAPER STRUCTURE AND LAYOUT ANALYSIS:\n"
        "Perform comprehensive structural review of the paper organization:\n"
        "- SECTION ORDERING: Verify logical flow - Abstract, Introduction, Methods, Results, Discussion, Conclusion\n"
        "- GLOSSARY PLACEMENT: If present, glossary should be positioned appropriately (typically before references or as appendix)\n"
        "- APPENDIX ORGANIZATION: Consider moving large tables, detailed algorithms, or pseudocode to appendix for better readability\n"
        "- PSEUDOCODE PLACEMENT: Evaluate if complex algorithms should be moved to appendix while keeping simplified versions in main text\n"
        "- REFERENCE POSITIONING: Ensure references section is at the very end, properly formatted\n"
        "- SUBSECTION BALANCE: Check that sections are well-balanced in length and content depth\n"
        "- FLOATING ELEMENTS: Verify figures/tables appear near their first mention, not orphaned on separate pages\n"
        "- PAGE BREAKS: Ensure logical page breaks that don't split related content inappropriately\n"
        "- HEADER/FOOTER: Check for consistent and appropriate header/footer formatting\n"
        "- ACKNOWLEDGMENTS: Verify acknowledgments section is properly placed (typically before references)\n\n"
        
        "🎯 CRITICAL LATEX FLOAT POSITIONING REQUIREMENTS:\n"
        "MANDATORY LaTeX float positioning rules to ensure proper table/figure placement:\n"
        "- USE [htbp] POSITIONING: Replace ALL \\begin{figure}[H] and \\begin{table}[H] with [htbp] for better flow\n"
        "  * [H] forces exact placement and often creates poor page breaks and large white spaces\n"
        "  * [htbp] allows LaTeX to optimize placement: h=here, t=top, b=bottom, p=page\n"
        "- PROPER SPACING: Add \\vspace{0.5em} before and after tables/figures if needed for visual separation\n"
        "- CAPTION POSITIONING: Ensure captions are ABOVE tables and BELOW figures (standard convention)\n"
        "- REFERENCE IN TEXT: Every figure/table MUST be referenced in text using \\ref{} before it appears\n"
        "- CENTERED CONTENT: Use \\centering (not \\begin{center}) inside figure/table environments\n"
        "- AVOID FORCED POSITIONING: Never use [!h] or multiple [H] in sequence - let LaTeX optimize\n"
        "- LOGICAL ORDERING: Figures/tables should appear in the same order as referenced in text\n"
        "- SIZE OPTIMIZATION: Ensure tables fit page width using \\small, \\footnotesize, or \\resizebox if needed\n"
        "- BREAK WIDE TABLES: For tables wider than page, consider splitting or rotating (\\rotatebox{90})\n\n"
        
        "CRITICAL REQUIREMENTS - NO EXCEPTIONS:\n"
        "1. SINGLE FILE ONLY: The paper must be contained in ONE LaTeX file with NO separate bibliography files\n"
        "2. EMBEDDED REFERENCES MANDATORY: All references MUST be embedded directly in the paper.tex file using EITHER:\n"
        "   - \\begin{filecontents*}{refs.bib}...\\end{filecontents*} at the TOP of the file, OR\n"
        "   - \\begin{thebibliography}...\\end{thebibliography} at the END of the file\n"
        "   - NO separate refs.bib files are allowed - everything must be in paper.tex\n"
        "   - Ensure \\bibliography{refs} command uses the embedded filecontents, not external files\n"
        "3. COMPILABLE: The file must compile successfully with pdflatex\n"
        "4. NO FILENAMES IN TEXT: The paper text must NOT contain ANY references to uploaded filenames, code files, data files, or directory structures. Remove or replace:\n"
        "   - Specific filenames like 'simulation.py', 'results.txt', 'data.csv', 'model.pkl', etc.\n"
        "   - Directory paths like 'output/', 'src/', 'data/', etc.\n"
        "   - File extensions like '.py', '.txt', '.csv', '.pkl', etc.\n"
        "   - Replace with generic descriptions like 'our implementation', 'experimental results', 'computational framework'\n"
        "   - The paper should describe methods and results without revealing the underlying file structure\n"
        "5. AUTHENTIC REFERENCES MANDATORY: ALL references must be real, published works. Verify and correct:\n"
        "   - Minimum 15-20 authentic references from reputable sources\n"
        "   - Author names are real researchers in the field\n"
        "   - Journal/venue names are authentic and properly formatted\n"
        "   - Publication years are realistic and consistent\n"
        "   - DOIs are properly formatted (when available)\n"
        "   - References directly support claims in the text\n"
        "   - NO placeholder, fake, or fictional citations\n"
        "   - All references must be cited in the text using \\cite{} commands\n"
        "6. SELF-CONTAINED VISUALS: ALL tables, figures, diagrams must be:\n"
        "   - Created using LaTeX code only (TikZ, tabular, PGFPlots, etc.), OR\n"
        "   - Generated by simulation.py and saved as local files with proper \\includegraphics references\n"
        "   - Populated with actual simulation data, not fake numbers\n"
        "   - Self-rendering within the LaTeX document\n"
        "7. APPROPRIATE STRUCTURE: Ensure the paper structure aligns with the paper type and field conventions:\n"
        "   - Organize sections according to the paper's contribution type\n"
        "   - Include field-specific sections and evaluation methodologies\n"
        "   - Follow established academic writing conventions for the research area\n"
        "   - Ensure logical flow and appropriate section transitions\n"
        "8. RESULTS DOCUMENTATION: Ensure numerical results are properly documented:\n"
        "   - Simulation.py should save key results to 'results.txt' for reproducibility\n"
        "   - All numerical values in the paper must be traceable to simulation output\n"
        "   - Results file should be well-structured and human-readable\n"
        "   - Maintain consistency between cited numbers and simulation output\n"
        "9. FIGURE GENERATION: If paper includes computational figures:\n"
        "   - Figure generation code must be included in simulation.py\n"
        "   - Generated figures must be saved in the local project folder\n"
        "   - LaTeX must reference generated figures with proper \\includegraphics paths\n"
        "   - No external figure dependencies or missing image files\n"
        "   - GRAPH GENERATION STANDARDS: Ensure graphs contain at least 5 data samples and use appropriate sizing (minimum 6x4 inches)\n"
        "   - VISUALIZATION REQUIREMENTS: Include proper axis labels, legends, grid lines, and clear data representation\n"
        "   - FIGURE POSITIONING: Figures and graphs must be positioned appropriately within the paper - they CANNOT be placed between references/bibliography sections\n\n"
        
        "STRUCTURE ALIGNMENT REQUIREMENTS:\n"
        "- Theoretical papers: Focus on mathematical rigor, proofs, and theoretical analysis\n"
        "- Experimental papers: Include detailed methodology, controlled experiments, statistical validation\n"
        "- Survey papers: Provide comprehensive coverage, systematic classification, comparative analysis\n"
        "- Systems papers: Detail system architecture, implementation specifics, performance evaluation\n"
        "- Algorithm papers: Include complexity analysis, correctness proofs, empirical comparison\n"
        "- Security papers: Include threat models, security analysis, attack scenarios, defense mechanisms\n"
        "- Medical/Clinical papers: Follow clinical research standards with appropriate validation protocols\n"
        "- Use section names and organization appropriate for the paper type\n"
        "- Include evaluation methodology that matches the research contribution\n\n"
        
        "REFERENCE AUTHENTICITY REQUIREMENTS:\n"
        "- Replace any suspicious or placeholder references with real publications\n"
        "- Ensure minimum 15-20 authentic references from reputable sources\n"
        "- Use recent publications (2018-2025) mixed with foundational works\n"
        "- Verify all bibliographic details are correct\n"
        "- Ensure proper citation usage throughout the paper\n\n"
        
        "VISUAL CONTENT SELF-CONTAINMENT:\n"
        "- Convert any external figure references to TikZ/PGFPlots code or simulation-generated files\n"
        "- Ensure all tables use tabular environment with real simulation data\n"
        "- Create diagrams using TikZ or other LaTeX-native tools, or reference simulation-generated figures\n"
        "- Populate all numerical content from actual simulation results\n"
        "- Allow dependencies only on simulation-generated image files\n"
        "- GRAPH GENERATION STANDARDS: Ensure all graphs contain at least 5 data samples with proper sizing (minimum 6x4 inches) and clear visualization\n\n"
        
        "TABLE/FIGURE POSITIONING REQUIREMENTS:\n"
        "- CRITICAL: Use contextual positioning specifiers - [h] (here if possible), [ht] (here or top of section), or [H] (force here)\n"
        "- NEVER use [t] or [!t] positioning which forces floats to page tops regardless of context\n"
        "- Place tables/figures in relevant subsections immediately after first text mention\n"
        "- Ensure visual self-containment with comprehensive, descriptive captions\n"
        "- Each float must appear in logical context within its subsection, not forced to arbitrary page positions\n"
        "- Tables/figures should enhance understanding within their specific subsection context\n"
        "- Review all existing \\begin{table} and \\begin{figure} environments to ensure proper positioning\n"
        "- CRITICAL: Table formatting for large tables that exceed page borders:\n"
        "  * Split long tables into multiple parts using \\begin{longtable} or manual splitting\n"
        "  * Repeat table headers/keys in each part: 'Table X (continued)' or 'Table X (part 2 of 3)'\n"
        "  * Add '(continued)' text between table parts to maintain continuity\n"
        "  * Use \\multicolumn spanning for section breaks in large tables\n"
        "  * Ensure each table part is self-explanatory with repeated column headers\n"
        "  * Consider landscape orientation (\\begin{landscape}) for very wide tables\n\n"
        
        "FIGURE AND TABLE PLACEMENT REQUIREMENTS (CRITICAL):\n"
        "- Place **all figures and tables either inline in the main text where they are first cited, or at the very end of the document but strictly before the references section**\n"
        "- **Do not place any figures or tables after the references or between references**\n"
        "- If LaTeX floats push figures out of order, insert them explicitly with [H] (from the float package) to force placement where cited\n"
        "- When using [H] positioning, ensure the float package is loaded: \\usepackage{float}\n"
        "- For figures/tables at document end, place them immediately before \\bibliography{} or \\begin{thebibliography}\n"
        "- Never allow figures or tables to appear in or after the reference section\n\n"
        
        "REVISION PRIORITIES:\n"
        "1. Address all scientific/methodological concerns raised\n"
        "2. Fix LaTeX formatting issues (figures, tables, equations, citations)\n"
        "3. Update content based on actual simulation results\n"
        "4. Replace fake references with authentic ones\n"
        "5. Remove all specific filenames, file extensions, and directory structures from paper text\n"
        "6. Convert external visuals to self-contained LaTeX code or simulation-generated files\n"
        "7. Restructure sections to match paper type and field conventions\n"
        "8. Ensure proper documentation of results in results.txt\n"
        "9. Implement figure generation in simulation.py with local file saving\n"
        "10. CRITICAL: Fix table/figure positioning using [h]/[ht]/[H] instead of [t]/[!t] for contextual placement\n"
        "11. CRITICAL: Ensure all figures and tables are placed either inline where cited or at document end before references\n"
        "12. CRITICAL: Prevent any figures/tables from appearing after or between references\n"
        "13. Improve clarity and presentation quality\n"
        "14. Ensure reproducibility and code quality\n\n"
        
        "FORMATTING REQUIREMENTS (CRITICAL - NO EXCEPTIONS):\n"
        "- ALL content: use width=\\linewidth constraints (never exceed page width)\n"
        "- ALL wide tables: \\begin{adjustbox}{width=\\linewidth}...\\end{adjustbox}\n"
        "- ALL TikZ diagrams: wrap in \\begin{adjustbox}{width=\\linewidth}...\\end{adjustbox}\n"
        "- ALL content must fit within vertical page margins\n"
        "- NO CODE BLOCKS: NEVER use \\begin{lstlisting}, \\begin{verbatim}, \\begin{code}, or any code listing environments\n"
        "- ALGORITHMS ONLY: Use \\begin{algorithm}, \\begin{algorithmic}, or algorithm2e environments for pseudocode/algorithms\n"
        "- Replace any existing code blocks with proper algorithm pseudocode descriptions\n"
        "- NO PROMPT REPETITION: Do not repeat or reference the instructions/prompts given to you directly in the paper content\n"
        "- Proper math environments and symbol usage\n"
        "- Consistent citation style and complete bibliography\n"
        "- Clear section structure and logical flow\n\n"
        
        "CONTENT REQUIREMENTS:\n"
        "- Use ONLY actual simulation results (no fake numbers)\n"
        "- Remove all specific filenames, file extensions, and directory references from paper text\n"
        "- Ensure numerical results are documented in results.txt\n"
        "- Include simulation-generated figures with proper local file references\n"
        "- Ensure claims are supported by authentic references\n"
        "- Address limitations and future work\n"
        "- Improve clarity of methodology and results\n\n"
        
        "BIBLIOGRAPHY INTEGRATION REQUIREMENTS - MANDATORY:\n"
        "Option 1 - filecontents (RECOMMENDED - all in paper.tex):\n"
        "\\begin{filecontents*}{refs.bib}\n"
        "@article{RealAuthor2024,\n"
        "  author = {A. Real and B. Author},\n"
        "  title = {Authentic Published Title},\n"
        "  journal = {Nature Communications},\n"
        "  year = {2024},\n"
        "  volume = {15},\n"
        "  pages = {1234},\n"
        "  doi = {10.1038/s41467-024-xxxxx}\n"
        "}\n"
        "\\end{filecontents*}\n"
        "\\documentclass{...}\n"
        "...\n"
        "\\bibliography{refs}\n"
        "\n"
        "Option 2 - Direct thebibliography (all in paper.tex):\n"
        "\\begin{thebibliography}{99}\n"
        "\\bibitem{RealAuthor2024} A. Real, B. Author, \"Authentic Title,\" Nature Communications, vol. 15, p. 1234, 2024.\n"
        "\\end{thebibliography}\n"
        "\n"
        "CRITICAL: NO separate refs.bib files - all references must be embedded in paper.tex!\n\n"
        
        "SELF-CONTAINED VISUAL EXAMPLES:\n"
        "- Figures: \\begin{tikzpicture}...\\end{tikzpicture} (not \\includegraphics)\n"
        "- Tables: \\begin{adjustbox}{width=\\linewidth}\\begin{tabular}{|c|c|}\\hline Data1 & Data2 \\\\\\hline\\end{tabular}\\end{adjustbox}\n"
        "- Plots: \\begin{tikzpicture}\\begin{axis}\\addplot coordinates {(1,2) (3,4)};\\end{axis}\\end{tikzpicture}\n\n"
        
        "Return ONLY the complete revised LaTeX file with ALL issues addressed, authentic references, self-contained visuals, appropriate structure for the paper type, and proper size constraints applied."
    )
    
    # Add custom user prompt if provided - it takes priority
    if user_prompt:
        sys_prompt = (
            f"PRIORITY INSTRUCTION FROM USER: {user_prompt}\n\n"
            "The above user instruction takes precedence when revising the paper. "
            "However, still maintain the critical technical requirements (single file, embedded references, compilable LaTeX).\n\n"
            + sys_prompt
        )
    
    # Collect all project files for complete context
    project_files_content = ""
    if project_dir and project_dir.exists():
        project_files_content = _collect_project_files(project_dir)
    
    user = (
        "----- CURRENT PAPER (LATEX) -----\n" + paper_tex + "\n"
        "----- SIMULATION CODE & OUTPUTS -----\n" + sim_summary + "\n"
        "----- REVIEW FEEDBACK -----\n" + review_text + "\n"
        "----- ALL PROJECT FILES (FOR CONTEXT) -----\n" + project_files_content + "\n"
    )
    
    # Add quality issues if detected
    if quality_issues:
        user += (
            "\n----- DETECTED QUALITY ISSUES -----\n"
            "The following specific quality issues have been automatically detected and MUST be addressed:\n\n"
        )
        for issue in quality_issues:
            user += f"• {issue}\n"
        user += (
            "\n----- END QUALITY ISSUES -----\n\n"
            "CRITICAL: Your revision MUST specifically address ALL of the above quality issues. "
            "These are not suggestions - they are required fixes that must be implemented.\n"
        )
    
    # Always include LaTeX compilation information
    if latex_errors:
        user += (
            "\n----- LATEX COMPILATION ERRORS (LAST 20 LINES OF .log) -----\n" + 
            latex_errors + 
            "\n----- END LATEX ERRORS -----\n\n"
            "CRITICAL: Fix ALL LaTeX compilation errors. The paper MUST compile successfully with pdflatex.\n"
        )
    else:
        user += (
            "\n----- LATEX COMPILATION STATUS -----\n" +
            "Previous compilation was SUCCESSFUL. No errors detected.\n" +
            "----- END COMPILATION STATUS -----\n\n"
        )
    
    user += (
        "Return ONLY the complete revised LaTeX file. CRITICAL: Apply proper size constraints to ALL figures, tables, and diagrams. "
        "Ensure the paper is self-contained with embedded references and compiles without errors."
    )
    
    return [{"role": "system", "content": sys_prompt}, {"role": "user", "content": user}]

def run_workflow(
    topic: str,
    field: str,
    question: str,
    output_dir: Path,
    model: str = DEFAULT_MODEL,
    request_timeout: Optional[int] = 3600,
    max_retries: int = 3,
    max_iterations: int = 4,
    modify_existing: bool = False,
    strict_singletons: bool = True,
    python_exec: Optional[str] = None,
    quality_threshold: float = 1.0,  # New parameter
    check_references: bool = True,    # New parameter
    validate_figures: bool = True,    # New parameter
    user_prompt: Optional[str] = None,  # New parameter for custom user prompt
    config: Optional[WorkflowConfig] = None,  # Configuration parameter
    enable_ideation: bool = True,     # Enable ideation phase for new papers
    specify_idea: Optional[str] = None,  # Specify a research idea to use directly
    num_ideas: int = 15,             # Number of ideas to generate
    output_diffs: bool = False,       # Optional diff output for each review/revision cycle
    document_type: str = "auto"      # Document type to generate
) -> Path:
    """Enhanced workflow with quality validation, progress tracking, and custom user prompts."""
    
    # Use provided config or load default
    if config is None:
        config = WorkflowConfig()
    
    # Override config values with explicit parameters if provided
    if quality_threshold != 1.0:  # Only override if explicitly set
        config.quality_threshold = quality_threshold
    if max_iterations != 4:
        config.max_iterations = max_iterations
    if check_references is not True:
        config.reference_validation = check_references
    if validate_figures is not True:
        config.figure_validation = validate_figures
    
    logger.info(f"Starting workflow with config: quality_threshold={config.quality_threshold}, max_iterations={config.max_iterations}")
    
    # Initialize document type and prompt generator
    if document_type == "auto":
        detected_type = infer_document_type(topic=topic, field=field, question=question)
        print(f"Auto-detected document type: {detected_type.value}")
    else:
        try:
            detected_type = DocumentType(document_type)
            print(f"Using specified document type: {detected_type.value}")
        except ValueError:
            available_types = [t.value for t in DocumentType]
            raise ValueError(f"Invalid document type '{document_type}'. Available types: {available_types}")
    
    prompt_generator = DocumentPromptGenerator
    logger.info(f"Document type: {detected_type.value}")
    
    project_dir = _prepare_project_dir(output_dir, modify_existing)
    paper_path, sim_path = ensure_single_tex_py(project_dir, strict=strict_singletons)

    # Progress tracking variables
    quality_history = []
    stagnation_count = 0
    best_quality_score = 0.0
    
    # Get custom user prompt if not provided as parameter
    if user_prompt is None:
        print("\n" + "="*60)
        print("CUSTOM PROMPT INPUT")
        print("="*60)
        print("You can provide a custom prompt that will be integrated into all AI interactions.")
        print("This prompt will take priority over standard requirements when conflicts arise.")
        print("Examples:")
        print("  - 'Focus on mathematical rigor and formal proofs'")
        print("  - 'Emphasize practical applications and real-world examples'")
        print("  - 'Use a conversational writing style suitable for broader audiences'")
        print("  - 'Include extensive experimental validation and statistical analysis'")
        print("\nLeave empty to use standard prompts only.")
        print("-" * 60)
        
        user_prompt = timeout_input("Enter your custom prompt (or press Enter to skip):", timeout=30, default="").strip()
        if not user_prompt:
            user_prompt = None
        else:
            print(f"\n Custom prompt set: {user_prompt[:100]}{'...' if len(user_prompt) > 100 else ''}")
    
    # Store user prompt for use in all AI interactions
    if user_prompt:
        print(f"\nUsing custom user prompt throughout workflow")
    
    print(f"\nProject directory: {project_dir}")
    print(f"Paper file: {paper_path}")
    print(f"Simulation file: {sim_path}")

    # Check for separate refs.bib files and warn user
    refs_bib_files = list(project_dir.glob("*.bib"))
    if refs_bib_files:
        print(f" WARNING: Found separate bibliography files: {[f.name for f in refs_bib_files]}")
        print("   All references must be embedded in paper.tex using filecontents or thebibliography")
        print("   Separate .bib files will be ignored during compilation!")

    # Check if this is actually a minimal template (new paper) or has real content
    paper_content = paper_path.read_text(encoding="utf-8").strip()
    is_minimal_template = (paper_content == "\\documentclass{article}\\begin{document}\\end{document}" or len(paper_content) < 200)
    
    # If no real paper content yet (fresh), run ideation and then draft one.
    if is_minimal_template:
        if specify_idea:
            print(f"Using specified research idea: {specify_idea}")
            
            # Use the specified idea directly
            final_topic = specify_idea
            final_question = specify_idea
            
            # Save specified idea to project directory
            ideation_file = project_dir / "ideation_analysis.txt"
            with open(ideation_file, 'w', encoding='utf-8') as f:
                f.write("RESEARCH IDEATION ANALYSIS\n")
                f.write("=" * 50 + "\n\n")
                f.write(f"Original Topic: {topic}\n")
                f.write(f"Original Question: {question}\n")
                f.write(f"Field: {field}\n\n")
                f.write(f"Specified Idea: {specify_idea}\n")
                f.write("Note: Ideation phase was skipped because a specific idea was provided.\n")
            
            print(f"Specified idea saved to: {ideation_file}")
            
        elif enable_ideation:
            print("Starting Ideation Phase for new paper...")
            
            # Generate and analyze multiple research ideas
            ideation_result = _generate_research_ideas(
                topic=topic,
                field=field,
                question=question,
                model=model,
                prompt_generator=prompt_generator,
                doc_type=detected_type,
                num_ideas=num_ideas,
                request_timeout=request_timeout,
                fallback_models=config.fallback_models
            )
            
            # Use the best idea to refine the topic and question
            selected_idea = ideation_result.get("selected_idea")
            if selected_idea:
                refined_topic = selected_idea.get("title", topic)
                refined_question = selected_idea.get("core_concept", question)
                
                print(f"\n Selected Research Idea:")
                print(f"   Title: {refined_topic}")
                print(f"   Concept: {refined_question}")
                print(f"   Scores: O={selected_idea.get('originality', 'N/A')}, "
                      f"I={selected_idea.get('impact', 'N/A')}, "
                      f"F={selected_idea.get('feasibility', 'N/A')}")
                
                # Save ideation results to project directory
                ideation_file = project_dir / "ideation_analysis.txt"
                with open(ideation_file, 'w', encoding='utf-8') as f:
                    f.write("RESEARCH IDEATION ANALYSIS\n")
                    f.write("=" * 50 + "\n\n")
                    f.write(f"Original Topic: {topic}\n")
                    f.write(f"Original Question: {question}\n")
                    f.write(f"Field: {field}\n\n")
                    f.write(f"Selected Idea: {refined_topic}\n")
                    f.write(f"Refined Concept: {refined_question}\n\n")
                    f.write("FULL IDEATION RESPONSE:\n")
                    f.write("-" * 30 + "\n")
                    f.write(ideation_result.get("raw_response", ""))
                
                print(f"Ideation analysis saved to: {ideation_file}")
                
                # Use refined topic and question for paper generation
                final_topic = refined_topic
                final_question = refined_question
            else:
                print("  No ideas selected, using original topic/question")
                final_topic = topic
                final_question = question
        else:
            print("  Ideation phase skipped (--skip-ideation flag used)")
            final_topic = topic
            final_question = question
        
        print(f"\nCreating paper draft for: {final_topic}")
        
        draft = generate_initial_draft(
            final_topic,
            field,
            final_question,
            user_prompt,
            model,
            request_timeout,
            config,
        )
        paper_path.write_text(draft, encoding="utf-8")
        
        if enable_ideation:
            print(" Initial draft created with ideation-selected concept")
        else:
            print(" Initial draft created with original concept")
    else:
        print(f"Using existing paper content ({len(paper_content)} characters)")
        print("  Skipping ideation phase for existing paper")

    # Extract/refresh simulation.py from LaTeX initially
    extract_simulation_from_tex(paper_path, sim_path)

    # Review-Revise loop with quality tracking
    for i in range(1, config.max_iterations + 1):
        print(f"Starting iteration {i} of {max_iterations}")
        
        # ALWAYS run simulation before each review to get current results
        print(f"Running simulation before review {i}...")
        
        sim_summary, _ = run_simulation_step(
            paper_path,
            sim_path,
            project_dir,
            model,
            request_timeout,
            python_exec,
        )
        
        # COMPILE LATEX WITH DYNAMIC TIMEOUT
        print(f"Compiling LaTeX file with pdflatex...")
        current_tex = paper_path.read_text(encoding="utf-8", errors="ignore")
        dynamic_timeout = _calculate_dynamic_timeout(current_tex, config)
        latex_success, latex_errors = _compile_latex_and_get_errors(paper_path, timeout=dynamic_timeout)
        
        if not latex_success:
            print(f" LaTeX compilation failed. Errors will be sent to LLM for fixing.")
            print(f"Error log (last 20 lines):\n{latex_errors}")
        else:
            print(f" LaTeX compilation successful!")
        
        # COMPREHENSIVE QUALITY VALIDATION
        quality_issues = _validate_research_quality(current_tex, sim_summary)
        
        # Additional validations based on config
        if config.reference_validation:
            ref_issues = _validate_references_with_external_apis(current_tex, config)
            quality_issues.extend(ref_issues)
        
        if config.figure_validation:
            fig_issues = _validate_figure_generation(current_tex, sim_path, project_dir)
            quality_issues.extend(fig_issues)
        
        if quality_issues:
            print(f"⚠ Quality issues detected ({len(quality_issues)} total):")
            for idx, issue in enumerate(quality_issues[:10], 1):  # Show first 10 issues
                print(f"   {idx}. {issue}")
            if len(quality_issues) > 10:
                print(f"   ... and {len(quality_issues) - 10} more issues")
            logger.info(f"Quality issues detected: {len(quality_issues)} issues found")
        else:
            print("✓ No quality issues detected")
            logger.info("No quality issues detected")
        
        # CALCULATE QUALITY METRICS AND TRACK PROGRESS
        current_metrics = _extract_quality_metrics(current_tex, sim_summary)
        quality_score = _calculate_quality_score(current_metrics, quality_issues)
        quality_history.append(quality_score)
        
        # Resource management - limit quality history size using config
        if len(quality_history) > config.max_quality_history_size:
            keep_size = config.max_quality_history_size // 2
            quality_history = quality_history[-keep_size:]  # Keep recent entries
            logger.info(f"Quality history trimmed to {keep_size} entries")
            print("Quality history trimmed for memory management")
        
        logger.info(f"Iteration {i} quality score: {quality_score:.2f}")
        print(f"Iteration {i} quality score: {quality_score:.2f}")
        
        # Check for improvement
        if quality_score > best_quality_score:
            best_quality_score = quality_score
            stagnation_count = 0
        else:
            stagnation_count += 1
        
        # Early stopping for stagnation
        if stagnation_count >= 2 and i > 1:
            print(f" Quality stagnation detected ({stagnation_count} iterations without improvement)")
        
        # COMBINED REVIEW AND REVISION IN ONE CALL
        print(f"Running combined review/editorial/revision process...")
        
        # Generate PDF for AI review if LaTeX compilation was successful and PDF review is enabled
        pdf_path = None
        if latex_success and config.enable_pdf_review:
            pdf_success, generated_pdf_path, pdf_error = _generate_pdf_for_review(paper_path, dynamic_timeout)
            if pdf_success and generated_pdf_path:
                pdf_path = generated_pdf_path
                file_size = pdf_path.stat().st_size
                print(f"✓ PDF generated for AI review: {pdf_path.name} ({file_size:,} bytes)")
                logger.info(f"PDF generated for AI review: {pdf_path.name} ({file_size:,} bytes)")
            else:
                print(f"✗ PDF generation failed: {pdf_error}")
                logger.warning(f"PDF generation failed: {pdf_error}")
        elif not config.enable_pdf_review:
            print(f"ℹ PDF review disabled in configuration - sending text-only content to AI model")
            logger.info("PDF review disabled - text-only review mode")
        else:
            print(f"⚠ Skipping PDF generation due to LaTeX compilation failure")
            logger.warning("PDF generation skipped due to LaTeX compilation failure")
        
        review, decision = run_review_revision_step(
            current_tex,
            sim_summary,
            latex_errors,
            project_dir,
            user_prompt,
            i,
            model,
            request_timeout,
            config,
            pdf_path,
            output_diffs,
            paper_path,
            quality_issues,
        )

        print(f"Review completed")
        print(f"Review complete, applying revisions...")

        # SIMPLIFIED STOPPING LOGIC BASED ON QUALITY THRESHOLD
        meets_quality_threshold = quality_score >= config.quality_threshold
        
        if latex_success and meets_quality_threshold:
            print(f"[OK] Quality threshold met at iteration {i} (score: {quality_score:.2f}) and LaTeX compiles successfully")
            final_metrics = _extract_quality_metrics(current_tex, sim_summary)
            print(f"Final paper metrics: {final_metrics}")
            break
        elif stagnation_count >= 2 and not config.no_early_stopping:
            print(f"[STOP] Quality stagnating for {stagnation_count} iterations. Ending revisions.")
            break
        elif stagnation_count >= 2 and config.no_early_stopping:
            print(f"[INFO] Quality stagnating for {stagnation_count} iterations, but early stopping is disabled. Continuing...")
        
        print(f"Iteration {i}: Combined review and revision completed")
    
    # Final quality report
    print(f"\nQuality progression: {[f'{q:.2f}' for q in quality_history]}")
    print(f"Best quality score achieved: {best_quality_score:.2f}")

    return project_dir

def _calculate_quality_score(metrics: Dict[str, Any], issues: List[str]) -> float:
    """Calculate a quality score based on metrics and issues."""
    score = 0.0
    
    # Structural completeness (0-40 points)
    if metrics.get('has_abstract'): score += 5
    if metrics.get('has_related_work'): score += 5
    if metrics.get('has_methodology'): score += 10
    if metrics.get('has_results'): score += 10
    if metrics.get('has_discussion'): score += 5
    if metrics.get('has_conclusion'): score += 5
    
    # Content richness (0-30 points)
    section_score = min(metrics.get('section_count', 0) * 2, 10)
    citation_score = min(metrics.get('citation_count', 0), 10)
    figure_table_score = min((metrics.get('figure_count', 0) + metrics.get('table_count', 0)) * 2, 10)
    score += section_score + citation_score + figure_table_score
    
    # Technical quality (0-20 points)
    if metrics.get('has_simulation'): score += 10
    if metrics.get('simulation_success'): score += 10
    
    # Issue penalty (0-10 points deduction)
    issue_penalty = min(len(issues) * 2, 10)
    score -= issue_penalty
    
    # Normalize to 0-1 scale
    return max(0.0, min(1.0, score / 90.0))

def _extract_paper_metadata(paper_content: str) -> Tuple[str, str, str]:
    """Extract topic, field, and research question from existing paper content with improved classification."""
    # Try to extract from title and abstract
    title_match = re.search(r'\\title\{([^}]+)\}', paper_content)
    abstract_match = re.search(r'\\begin\{abstract\}(.*?)\\end\{abstract\}', paper_content, re.DOTALL)
    
    title = title_match.group(1) if title_match else "Research Paper"
    abstract = abstract_match.group(1).strip() if abstract_match else ""
    
    # Extract key concepts from title and abstract to infer field and topic
    field = "Computer Science"  # Default field
    topic = title.split(':')[0] if ':' in title else title
    question = f"How to improve the methodology and results presented in: {title}"
    
    # Enhanced field classification with more sophisticated patterns
    content_lower = (title + " " + abstract).lower()
    
    # Define field classification patterns
    field_patterns = {
        "Quantum Computing": ['quantum', 'qubit', 'entangle', 'superposition', 'decoherence', 'quantum gate', 'quantum circuit'],
        "Machine Learning": ['neural', 'deep learning', 'machine learning', 'ai', 'artificial intelligence', 'model training', 'dataset', 'algorithm', 'classification', 'regression'],
        "Computational Biology": ['biology', 'protein', 'gene', 'medical', 'bioinformatics', 'genomics', 'molecular', 'clinical', 'biomedical'],
        "Cybersecurity": ['security', 'crypto', 'attack', 'defense', 'vulnerability', 'encryption', 'authentication', 'privacy'],
        "Computer Networks": ['network', 'protocol', 'routing', 'tcp', 'ip', 'wireless', 'internet', 'bandwidth'],
        "Computer Vision": ['image', 'vision', 'visual', 'recognition', 'detection', 'segmentation', 'opencv', 'cnn'],
        "Natural Language Processing": ['nlp', 'language', 'text', 'linguistic', 'translation', 'sentiment', 'chatbot'],
        "High Performance Computing": ['parallel', 'distributed', 'cluster', 'gpu', 'cuda', 'mpi', 'performance optimization'],
        "Software Engineering": ['software', 'programming', 'development', 'testing', 'debugging', 'code quality'],
        "Database Systems": ['database', 'sql', 'nosql', 'query', 'storage', 'indexing', 'transaction'],
        "Human-Computer Interaction": ['hci', 'user interface', 'usability', 'user experience', 'interaction design'],
        "Computer Graphics": ['graphics', 'rendering', 'visualization', '3d', 'animation', 'shader'],
        "Theoretical Computer Science": ['algorithm', 'complexity', 'theory', 'mathematical', 'proof', 'formal'],
        "Data Science": ['data', 'analytics', 'statistics', 'big data', 'data mining', 'visualization']
    }
    
    # Determine paper type based on content structure and keywords
    paper_type = _classify_paper_type(content_lower, paper_content)
    
    # Score each field based on keyword matches
    field_scores = {}
    for field_name, keywords in field_patterns.items():
        score = sum(1 for keyword in keywords if keyword in content_lower)
        if score > 0:
            field_scores[field_name] = score
    
    # Select field with highest score
    if field_scores:
        field = max(field_scores, key=field_scores.get)
    
    return topic, field, question

def _classify_paper_type(content_lower: str, paper_content: str) -> str:
    """Classify paper type based on content analysis."""
    
    # Check section structure for paper type indicators
    sections = re.findall(r'\\section\{([^}]+)\}', paper_content, re.IGNORECASE)
    section_names = [s.lower() for s in sections]
    
    type_indicators = {
        'theoretical': [
            'theorem', 'proof', 'lemma', 'proposition', 'mathematical', 'formal',
            'complexity analysis', 'theoretical framework'
        ],
        'experimental': [
            'experiment', 'evaluation', 'empirical', 'benchmark', 'performance',
            'results', 'methodology', 'dataset'
        ],
        'survey': [
            'survey', 'review', 'taxonomy', 'classification', 'comparative',
            'state of the art', 'literature review'
        ],
        'systems': [
            'system', 'architecture', 'implementation', 'design', 'framework',
            'platform', 'tool', 'prototype'
        ],
        'algorithm': [
            'algorithm', 'algorithmic', 'optimization', 'heuristic', 'approach',
            'method', 'procedure', 'technique'
        ],
        'security': [
            'security', 'attack', 'defense', 'vulnerability', 'threat model',
            'cryptographic', 'privacy'
        ]
    }
    
    # Score each type
    type_scores = {}
    all_content = content_lower + ' ' + ' '.join(section_names)
    
    for paper_type, indicators in type_indicators.items():
        score = sum(1 for indicator in indicators if indicator in all_content)
        if score > 0:
            type_scores[paper_type] = score
    
    if type_scores:
        return max(type_scores, key=type_scores.get)
    
    return 'general'

def _check_existing_paper(output_dir: Path) -> Optional[Tuple[str, str, str]]:
    """Check if there's an existing paper and extract its metadata."""
    if not output_dir.exists():
        return None
    
    # Look for any .tex files
    tex_files = list(output_dir.glob("*.tex"))
    if not tex_files:
        return None
    
    # Read the first .tex file found
    try:
        paper_content = tex_files[0].read_text(encoding="utf-8", errors="ignore")
        if len(paper_content.strip()) > 100:  # Make sure it's not just a minimal template
            return _extract_paper_metadata(paper_content)
    except Exception:
        pass
    
    return None

def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    # First stage: Check if --modify-existing is present
    # We need to do this to determine which arguments to make available
    if argv is None:
        argv = sys.argv[1:]
    
    modify_existing = "--modify-existing" in argv
    
    p = argparse.ArgumentParser(description="Enhanced SciResearch Workflow with Quality Validation")
    
    # Conditionally add topic/field/question arguments only if NOT modifying existing
    if not modify_existing:
        p.add_argument("--topic", required=False, help="Research topic")
        p.add_argument("--field", required=False, help="Research field")
        p.add_argument("--question", required=False, help="Research question")
        p.add_argument("--document-type", choices=["research_paper", "engineering_paper", "finance_research", 
                      "equity_research", "survey_paper", "presentation_slides", "technical_report", 
                      "white_paper", "conference_paper", "journal_article", "auto"], 
                      default="auto", help="Type of document to generate (auto-detect if not specified)")
    
    p.add_argument("--output-dir", default="output", help="Output directory root (contains project subfolder)")
    p.add_argument("--model", default=DEFAULT_MODEL, help="OpenAI model to use (default: gpt-5)")
    p.add_argument("--request-timeout", type=int, default=3600, help="Per-request timeout seconds (0 means no timeout)")
    p.add_argument("--max-retries", type=int, default=3, help="Max OpenAI retries")
    p.add_argument("--max-iterations", type=int, default=4, help="Max review->revise iterations")
    p.add_argument("--no-early-stopping", action="store_true", help="Disable early stopping for quality stagnation (run all max iterations)")
    p.add_argument("--modify-existing", action="store_true", help="If output dir already has paper.tex, modify in place")
    p.add_argument("--strict-singletons", action="store_true", default=True, help="Keep only paper.tex & simulation.py (others archived)")
    p.add_argument("--python-exec", default=None, help="Python interpreter for running simulation.py")
    
    # Configuration file support
    p.add_argument("--config", type=str, default=None, help="Path to configuration JSON file")
    p.add_argument("--save-config", type=str, default=None, help="Save current configuration to JSON file")
    
    # New quality control parameters
    p.add_argument("--quality-threshold", type=float, default=1.0, help="Minimum quality score required for acceptance (0.0-1.0)")
    p.add_argument("--check-references", action="store_true", default=True, help="Enable external reference validation")
    p.add_argument("--validate-figures", action="store_true", default=True, help="Enable figure generation validation")
    p.add_argument("--skip-reference-check", action="store_true", help="Disable external reference validation (faster)")
    p.add_argument("--skip-figure-validation", action="store_true", help="Disable figure generation validation (faster)")
    
    p.add_argument("--enable-pdf-review", action="store_true", default=False, help="Send PDF files to AI models during review/revision")
    p.add_argument("--disable-pdf-review", action="store_true", help="Disable PDF review (text-only review)")
    
    # Ideation parameters
    p.add_argument("--enable-ideation", action="store_true", default=True, help="Enable research ideation phase for new papers")
    p.add_argument("--skip-ideation", action="store_true", help="Skip research ideation phase (use original topic directly)")
    p.add_argument("--specify-idea", type=str, default=None, help="Specify a research idea to use directly (skips ideation phase)")
    p.add_argument("--num-ideas", type=int, default=15, help="Number of research ideas to generate (10-20)")
    
    # Custom prompt parameter
    p.add_argument("--user-prompt", type=str, default=None, help="Custom prompt that takes priority over standard requirements")
    
    # Diff output parameter - now enabled by default
    p.add_argument("--no-output-diffs", action="store_true", help="Disable diff file saving for each review/revision cycle")
    p.add_argument("--output-diffs", action="store_true", default=True, help="Save diff files for each review/revision cycle to track changes (default: enabled)")
    
    # Content protection parameters
    p.add_argument("--disable-content-protection", action="store_true", help="Disable content protection against accidental deletions (DANGEROUS)")
    p.add_argument("--auto-approve-changes", action="store_true", help="Automatically approve content changes that pass safety checks")
    p.add_argument("--content-protection-threshold", type=float, default=0.15, help="Maximum allowed content reduction as fraction (default: 0.15 = 15%%)")
    
    # Test-time compute scaling parameters
    p.add_argument("--test-scaling", action="store_true", help="Run test-time compute scaling analysis instead of normal workflow")
    p.add_argument("--scaling-candidates", type=str, default="3,5,7,10", help="Comma-separated list of candidate counts to test (e.g., '3,5,7,10')")
    p.add_argument("--scaling-timeout", type=int, default=1800, help="Base timeout for scaling tests (seconds)")
    p.add_argument("--scaling-prompt", type=str, default=None, help="Custom prompt for scaling tests")
    
    # Test-time compute scaling for normal workflow
    p.add_argument("--use-test-time-scaling", action="store_true", help="Enable test-time compute scaling during revision cycles")
    p.add_argument("--revision-candidates", type=int, default=3, help="Number of revision candidates to generate when using test-time scaling")
    p.add_argument("--draft-candidates", type=int, default=1, help="Number of initial draft candidates to generate")
    
    args = p.parse_args(argv)
    
    # Handle skip flags
    if args.skip_reference_check:
        args.check_references = False
    if args.skip_figure_validation:
        args.validate_figures = False
    if args.skip_ideation:
        args.enable_ideation = False
    if args.specify_idea:
        args.enable_ideation = False  # Disable ideation if idea is specified
    if args.disable_pdf_review:
        args.enable_pdf_review = False
    if args.no_output_diffs:
        args.output_diffs = False
    if args.disable_content_protection:
        args.enable_content_protection = False
    else:
        args.enable_content_protection = True
    
    # Initialize topic, field, question, document_type attributes if they don't exist (modify-existing mode)
    if modify_existing:
        if not hasattr(args, 'topic'):
            args.topic = None
        if not hasattr(args, 'field'):
            args.field = None
        if not hasattr(args, 'question'):
            args.question = None
        if not hasattr(args, 'document_type'):
            args.document_type = "auto"
    
    # Check if there's an existing paper first
    output_path = Path(args.output_dir)
    existing_metadata = None
    
    if args.modify_existing:
        # First check if the output_dir itself contains a .tex file
        existing_metadata = _check_existing_paper(output_path)
        
        # If not found directly, look for project subdirectories with .tex files
        if not existing_metadata and output_path.exists():
            for subdir in output_path.iterdir():
                if subdir.is_dir():
                    subdir_metadata = _check_existing_paper(subdir)
                    if subdir_metadata:
                        existing_metadata = subdir_metadata
                        # Update output_dir to point to the found project directory
                        args.output_dir = str(subdir)
                        break
    
    # If we found an existing paper, use its metadata; otherwise prompt for missing args
    if existing_metadata:
        topic, field, question = existing_metadata
        if not getattr(args, 'topic', None):
            args.topic = topic
        if not getattr(args, 'field', None):
            args.field = field
        if not getattr(args, 'question', None):
            args.question = question
        print(f"Detected existing paper - Topic: {args.topic}, Field: {args.field}")
    elif not args.modify_existing:
        # Interactive prompts if missing and no existing paper (but NOT when modifying existing)
        if not getattr(args, 'topic', None):
            args.topic = timeout_input("Topic:", timeout=30, default="Large Language Models").strip()
        if not getattr(args, 'field', None):
            args.field = timeout_input("Field:", timeout=30, default="Computer Science").strip()
        if not getattr(args, 'question', None):
            args.question = timeout_input("Research question:", timeout=30, default="Find revolutionary, impactful, and practical methods?").strip()
    # If modify_existing is True but no existing metadata found, we'll use whatever args were provided
    
    return args

def _extract_quality_metrics(paper_content: str, sim_summary: str) -> Dict[str, Any]:
    """Extract quality metrics from paper and simulation."""
    metrics = {}
    
    # Paper structure metrics
    metrics['has_abstract'] = bool(re.search(r'\\begin\{abstract\}', paper_content))
    metrics['section_count'] = len(re.findall(r'\\section\{', paper_content))
    metrics['figure_count'] = len(re.findall(r'\\includegraphics', paper_content))
    metrics['table_count'] = len(re.findall(r'\\begin\{table\}', paper_content))
    metrics['citation_count'] = len(re.findall(r'\\cite\{', paper_content))
    metrics['equation_count'] = len(re.findall(r'\\begin\{equation\}', paper_content))
    
    # Content quality indicators
    metrics['word_count'] = len(paper_content.split())
    metrics['has_related_work'] = bool(re.search(r'related.work|literature.review', paper_content, re.IGNORECASE))
    metrics['has_methodology'] = bool(re.search(r'methodology|method|approach', paper_content, re.IGNORECASE))
    metrics['has_results'] = bool(re.search(r'results|findings|outcomes', paper_content, re.IGNORECASE))
    metrics['has_discussion'] = bool(re.search(r'discussion|analysis', paper_content, re.IGNORECASE))
    metrics['has_conclusion'] = bool(re.search(r'conclusion|summary', paper_content, re.IGNORECASE))
    
    # Simulation quality
    if 'SIMULATION CODE:' in sim_summary:
        metrics['has_simulation'] = True
        metrics['simulation_success'] = 'error' not in sim_summary.lower()
    
    return metrics

def _validate_research_quality(paper_content: str, sim_summary: str) -> List[str]:
    """Validate the quality of the research paper."""
    issues = []
    
    # Check for single file requirement
    if '\\input{' in paper_content or '\\include{' in paper_content:
        issues.append("Paper uses \\input or \\include - must be a single self-contained file")
    
    # Check for embedded references requirement - MANDATORY
    has_filecontents = bool(re.search(r'\\begin\{filecontents\*?\}\{[^}]*\.bib\}', paper_content))
    has_thebibliography = bool(re.search(r'\\begin\{thebibliography\}', paper_content))
    has_external_bib = bool(re.search(r'\\bibliography\{[^}]+\}', paper_content)) and not has_filecontents
    
    if has_external_bib and not has_filecontents:
        issues.append("CRITICAL: Paper references external .bib file - all references must be embedded in paper.tex using filecontents or thebibliography")
    
    if not has_filecontents and not has_thebibliography:
        issues.append("CRITICAL: No embedded bibliography found - all references must be embedded in paper.tex using filecontents or thebibliography")
    
    # Check for minimum number of references
    if has_filecontents:
        bib_entries = len(re.findall(r'@\w+\{', paper_content))
    elif has_thebibliography:
        bib_entries = len(re.findall(r'\\bibitem\{', paper_content))
    else:
        bib_entries = 0
    
    if bib_entries < 15:
        issues.append(f"Insufficient references: only {bib_entries} found (minimum 15 required)")
    
    # Check that all references are cited in text
    if has_filecontents:
        # Extract citation keys from filecontents
        cite_keys = re.findall(r'@\w+\{([^,]+),', paper_content)
        citations_in_text = re.findall(r'\\cite\{([^}]+)\}', paper_content)
        cited_keys = []
        for citation in citations_in_text:
            cited_keys.extend(citation.split(','))
        cited_keys = [key.strip() for key in cited_keys]
        
        uncited_refs = [key for key in cite_keys if key not in cited_keys]
        if uncited_refs:
            issues.append(f"Uncited references found: {', '.join(uncited_refs[:5])}{'...' if len(uncited_refs) > 5 else ''}")
    
    # Verify no separate bibliography files are referenced
    if '\\input{refs}' in paper_content or '\\input{references}' in paper_content:
        issues.append("CRITICAL: Found \\input command for bibliography - all references must be embedded in paper.tex")
    
    # Check for filenames in paper text - NO filenames should appear except when referencing embedded filecontents
    # Extract embedded filecontents filenames to allow legitimate references
    filecontents_files = set()
    filecontents_matches = re.findall(r'\\begin\{filecontents\*?\}\{([^}]+)\}', paper_content)
    for filename in filecontents_matches:
        filecontents_files.add(filename)
    
    filename_patterns = [
        r'\b\w+\.py\b',     # Python files
        r'\b\w+\.csv\b',    # CSV files  
        r'\b\w+\.npy\b',    # NumPy files
        r'\b\w+\.txt\b',    # Text files
        r'\b\w+\.dat\b',    # Data files
        r'\b\w+\.json\b',   # JSON files
        r'\b\w+\.pkl\b',    # Pickle files
        r'\b\w+\.h5\b',     # HDF5 files
        r'\b\w+\.mat\b',    # MATLAB files
        r'\b\w+\.xlsx?\b',  # Excel files
        r'\b__pycache__\b', # Python cache directory
        r'\bdata/\b',       # Data directory
        r'\bcode/\b',       # Code directory
        r'\bresults/\b',    # Results directory
        r'\bsrc/\b',        # Source directory
        r'\boutput/\b',     # Output directory
    ]
    
    found_filenames = []
    for pattern in filename_patterns:
        matches = re.findall(pattern, paper_content, re.IGNORECASE)
        for match in matches:
            # Skip if this filename is from an embedded filecontents block
            if match not in filecontents_files:
                # Also check if it's used in legitimate pgfplots table references like {filename.csv}
                # Look for patterns like table [...] {filename} or \addplot table [...] {filename}
                table_refs = re.findall(rf'table\s*\[[^\]]*\]\s*\{{{re.escape(match)}\}}', paper_content, re.IGNORECASE)
                if not table_refs:
                    found_filenames.append(match)
    
    if found_filenames:
        issues.append(f"CRITICAL: Filenames found in paper text: {', '.join(set(found_filenames[:10]))}{'...' if len(set(found_filenames)) > 10 else ''}")
    
    # Check for authentic references
    suspicious_refs = _check_reference_authenticity(paper_content)
    if suspicious_refs:
        issues.extend(suspicious_refs)
    
    # Check for self-contained visual content
    visual_issues = _check_visual_self_containment(paper_content)
    if visual_issues:
        issues.extend(visual_issues)
    
    # Check for appropriate paper structure
    structure_issues = _check_paper_structure(paper_content)
    if structure_issues:
        issues.extend(structure_issues)
    
    # Check for essential sections
    if not re.search(r'\\begin\{abstract\}', paper_content):
        issues.append("Missing abstract")
    if not re.search(r'\\section\*?\{.*[Ii]ntroduction.*\}', paper_content):
        issues.append("Missing Introduction section")
    if not re.search(r'\\section\*?\{.*[Rr]elated.*[Ww]ork.*\}', paper_content):
        issues.append("Missing Related Work section")
    if not re.search(r'\\section\*?\{.*[Mm]ethodology.*\}', paper_content):
        issues.append("Missing Methodology section")
    if not re.search(r'\\section\*?\{.*[Ee]xperiments.*\}', paper_content):
        issues.append("Missing Experiments section")
    if not re.search(r'\\section\*?\{.*[Rr]esults.*\}', paper_content):
        issues.append("Missing Results section")
    if not re.search(r'\\section\*?\{.*[Dd]iscussion.*\}', paper_content):
        issues.append("Missing Discussion section")
    if not re.search(r'\\section\*?\{.*[Cc]onclusion.*\}', paper_content):
        issues.append("Missing Conclusion section")
    
    # Check for reasonable number of references
    bib_entries = len(re.findall(r'\\bibitem\{|@\w+\{', paper_content))
    if bib_entries < 10:
        issues.append(f"Only {bib_entries} bibliography entries (recommend 15+)")
    
    # Add figure/table validation
    figure_table_issues = _validate_figures_tables(paper_content)
    issues.extend(figure_table_issues)
    
    return issues

def _check_paper_structure(paper_content: str) -> List[str]:
    """Check if paper structure aligns with paper type and field conventions."""
    issues = []
    
    # Extract all section titles
    sections = re.findall(r'\\section\*?\{([^}]+)\}', paper_content)
    section_titles = [s.lower() for s in sections]
    
    # Identify potential paper type based on content and sections
    content_lower = paper_content.lower()
    
    # Check for field-specific requirements
    if any(word in content_lower for word in ['security', 'attack', 'vulnerability', 'threat']):
        # Security paper - should have threat model
        if not any('threat' in title for title in section_titles):
            issues.append("Security paper missing 'Threat Model' or 'Security Model' section")
        if not any(word in ' '.join(section_titles) for word in ['security analysis', 'attack', 'defense']):
            issues.append("Security paper should include security analysis, attack scenarios, or defense mechanisms")
    
    if any(word in content_lower for word in ['clinical', 'medical', 'patient', 'diagnosis']):
        # Medical paper - should have validation protocols
        if not any(word in ' '.join(section_titles) for word in ['validation', 'clinical', 'evaluation']):
            issues.append("Medical paper should include clinical validation or evaluation section")
    
    if any(word in content_lower for word in ['algorithm', 'complexity', 'optimization']):
        # Algorithm paper - should have analysis
        if not any(word in ' '.join(section_titles) for word in ['analysis', 'complexity', 'performance']):
            issues.append("Algorithm paper should include complexity analysis or performance analysis section")
    
    if any(word in content_lower for word in ['survey', 'review', 'taxonomy', 'classification']):
        # Survey paper - should have comparative analysis
        if not any(word in ' '.join(section_titles) for word in ['comparison', 'comparative', 'analysis', 'classification']):
            issues.append("Survey paper should include comparative analysis or classification section")
        if not any(word in ' '.join(section_titles) for word in ['future', 'directions', 'challenges']):
            issues.append("Survey paper should include future work or research directions section")
    
    if any(word in content_lower for word in ['system', 'architecture', 'implementation', 'design']):
        # Systems paper - should have design and evaluation
        if not any(word in ' '.join(section_titles) for word in ['design', 'architecture', 'implementation']):
            issues.append("Systems paper should include system design or architecture section")
        if not any(word in ' '.join(section_titles) for word in ['evaluation', 'performance', 'experiments']):
            issues.append("Systems paper should include performance evaluation section")
    
    if any(word in content_lower for word in ['theory', 'theorem', 'proof', 'mathematical']):
        # Theoretical paper - should have theory and analysis
        if not any(word in ' '.join(section_titles) for word in ['theory', 'analysis', 'proof']):
            issues.append("Theoretical paper should include theory, analysis, or proof sections")
    
    # Check for logical section flow
    if sections:
        # Introduction should be early
        intro_pos = next((i for i, title in enumerate(section_titles) if 'introduction' in title), None)
        if intro_pos is not None and intro_pos > 2:
            issues.append("Introduction section should appear early in the paper")
        
        # Conclusion should be near the end
        conclusion_pos = next((i for i, title in enumerate(section_titles) if 'conclusion' in title), None)
        if conclusion_pos is not None and conclusion_pos < len(sections) - 3:
            issues.append("Conclusion section should appear near the end of the paper")
        
        # Related work should be early (typically sections 1-3)
        related_pos = next((i for i, title in enumerate(section_titles) if 'related' in title), None)
        if related_pos is not None and related_pos > 4:
            issues.append("Related Work section should appear early in the paper (after Introduction)")
    
    return issues

def _check_reference_authenticity(paper_content: str) -> List[str]:
    """Check for potentially fake or suspicious references."""
    issues = []
    
    # Look for common fake reference patterns
    fake_patterns = [
        r'@article\{[^}]*,\s*author\s*=\s*\{[^}]*Example[^}]*\}',  # "Example" in author names
        r'@article\{[^}]*,\s*author\s*=\s*\{[^}]*Test[^}]*\}',     # "Test" in author names
        r'@article\{[^}]*,\s*author\s*=\s*\{[^}]*Placeholder[^}]*\}',  # "Placeholder" in author names
        r'@article\{[^}]*,\s*title\s*=\s*\{[^}]*Example[^}]*\}',   # "Example" in titles
        r'@article\{[^}]*,\s*title\s*=\s*\{[^}]*Sample[^}]*\}',    # "Sample" in titles
        r'@article\{[^}]*,\s*journal\s*=\s*\{[^}]*Journal of Example[^}]*\}',  # Fake journal names
        r'@article\{[^}]*,\s*journal\s*=\s*\{[^}]*Example Journal[^}]*\}',
    ]
    
    for pattern in fake_patterns:
        if re.search(pattern, paper_content, re.IGNORECASE):
            issues.append("Detected potentially fake or placeholder references - replace with authentic citations")
            break
    
    # Check for suspicious author patterns (single letters, too generic)
    author_matches = re.findall(r'author\s*=\s*\{([^}]+)\}', paper_content)
    for author in author_matches:
        if re.match(r'^[A-Z]\. [A-Z]\.$', author.strip()):  # Pattern like "A. B."
            issues.append(f"Suspicious author name '{author}' - use real researcher names")
        if any(word in author.lower() for word in ['example', 'test', 'sample', 'placeholder']):
            issues.append(f"Fake author name detected: '{author}' - replace with real researchers")
    
    # Check for unrealistic publication years
    year_matches = re.findall(r'year\s*=\s*\{(\d{4})\}', paper_content)
    current_year = 2025
    for year in year_matches:
        year_int = int(year)
        if year_int > current_year:
            issues.append(f"Future publication year {year} detected - use realistic years")
        if year_int < 1900:
            issues.append(f"Unrealistic publication year {year} detected")
    
    return issues

def _check_visual_self_containment(paper_content: str) -> List[str]:
    """Check if all visual content is self-contained within LaTeX."""
    issues = []
    
    # Check for external image includes
    external_includes = re.findall(r'\\includegraphics\[[^\]]*\]\{([^}]+)\}', paper_content)
    for include in external_includes:
        # Allow some common extensions but flag external files
        if not include.endswith(('.tex', '.tikz')) and '.' in include:
            issues.append(f"External image file reference detected: '{include}' - convert to TikZ or remove")
    
    # Check for missing TikZ usage when figures are present
    has_figures = bool(re.search(r'\\begin\{figure\}', paper_content))
    has_tikz = bool(re.search(r'\\begin\{tikzpicture\}', paper_content))
    has_pgfplots = bool(re.search(r'\\begin\{axis\}', paper_content))
    
    if has_figures and not (has_tikz or has_pgfplots):
        issues.append("Figures found but no TikZ/PGFPlots content - ensure all visuals are LaTeX-generated")
    
    # Check for tables with placeholder data
    table_content = re.findall(r'\\begin\{tabular\}.*?\\end\{tabular\}', paper_content, re.DOTALL)
    for table in table_content:
        if any(word in table.lower() for word in ['example', 'placeholder', 'xxx', 'tbd', 'todo']):
            issues.append("Table contains placeholder data - populate with real simulation results")
    
    # Check for figure positioning issues - figures should not be between references
    # Find bibliography/references section
    bib_start_patterns = [
        r'\\begin\{thebibliography\}',
        r'\\bibliography\{',
        r'\\section\*?\{.*?[Rr]eferences.*?\}',
        r'\\section\*?\{.*?[Bb]ibliography.*?\}'
    ]
    
    bib_positions = []
    for pattern in bib_start_patterns:
        matches = list(re.finditer(pattern, paper_content, re.IGNORECASE))
        bib_positions.extend([match.start() for match in matches])
    
    if bib_positions:
        bib_start = min(bib_positions)
        # Check for figures after bibliography start
        figures_after_bib = re.finditer(r'\\begin\{figure\}', paper_content[bib_start:])
        if any(figures_after_bib):
            issues.append("CRITICAL: Figures found after bibliography/references section - figures must be positioned before references")
    
    # Check for adequate data samples in plots (minimum 5 samples)
    if has_pgfplots:
        # Look for addplot commands with coordinate data
        plot_coords = re.findall(r'\\addplot.*?coordinates\s*\{([^}]+)\}', paper_content, re.DOTALL)
        for coords in plot_coords:
            # Count coordinate pairs
            coord_pairs = re.findall(r'\([^)]+\)', coords)
            if len(coord_pairs) < 5:
                issues.append(f"CRITICAL: Plot has insufficient data points ({len(coord_pairs)} found, minimum 5 required)")
        
        # Look for table-based plots and check CSV data
        table_plots = re.findall(r'\\addplot.*?table.*?\{([^}]+)\}', paper_content)
        for table_ref in table_plots:
            # Look for corresponding filecontents data
            csv_pattern = rf'\\begin\{{filecontents\*?\}}\{{{re.escape(table_ref)}\}}(.*?)\\end\{{filecontents\*?\}}' 
            csv_matches = re.search(csv_pattern, paper_content, re.DOTALL)
            if csv_matches:
                csv_content = csv_matches.group(1)
                # Count data rows (excluding header)
                data_lines = [line.strip() for line in csv_content.strip().split('\n') if line.strip() and not line.startswith('%')]
                if len(data_lines) < 6:  # Header + 5 data rows minimum
                    issues.append(f"CRITICAL: CSV data in '{table_ref}' has insufficient samples ({len(data_lines)-1} data rows, minimum 5 required)")
    
    return issues

def _validate_references_with_external_apis(paper_content: str, config: Optional[WorkflowConfig] = None) -> List[str]:
    """Validate references using external APIs like CrossRef DOI validation."""
    issues = []
    
    # Extract DOIs from references
    doi_pattern = r'doi\s*=\s*\{([^}]+)\}'
    dois = re.findall(doi_pattern, paper_content, re.IGNORECASE)
    
    valid_dois = 0
    invalid_dois = []
    
    for doi in dois:
        if _validate_doi_with_crossref(doi.strip(), config):
            valid_dois += 1
        else:
            invalid_dois.append(doi)
    
    if invalid_dois:
        issues.append(f"Invalid DOIs found: {', '.join(invalid_dois[:3])}{'...' if len(invalid_dois) > 3 else ''}")
    
    # Extract years and check if reasonable
    year_pattern = r'year\s*=\s*\{(\d{4})\}'
    years = [int(y) for y in re.findall(year_pattern, paper_content)]
    current_year = datetime.now().year
    
    future_years = [y for y in years if y > current_year]
    old_years = [y for y in years if y < 1950]
    
    if future_years:
        issues.append(f"References with future years detected: {future_years}")
    if old_years:
        issues.append(f"References with suspiciously old years: {old_years}")
    
    # Check author name patterns for suspicious entries
    author_pattern = r'author\s*=\s*\{([^}]+)\}'
    authors = re.findall(author_pattern, paper_content, re.IGNORECASE)
    
    suspicious_authors = []
    for author in authors:
        if any(word in author.lower() for word in ['example', 'placeholder', 'test', 'fake', 'dummy', 'sample']):
            suspicious_authors.append(author[:50])
    
    if suspicious_authors:
        issues.append(f"Suspicious author names found: {', '.join(suspicious_authors[:2])}{'...' if len(suspicious_authors) > 2 else ''}")
    
    return issues

import functools

# Rate limiting and caching for DOI validation
_doi_validation_cache = {}
_last_doi_check_time = 0

@functools.lru_cache(maxsize=1000)
@functools.lru_cache(maxsize=1000)
def _validate_doi_with_crossref_cached(doi: str) -> bool:
    """Cached DOI validation to avoid redundant API calls."""
    return _validate_doi_with_crossref_uncached(doi)

def _validate_doi_with_crossref(doi: str, config: Optional[WorkflowConfig] = None) -> bool:
    """Validate a DOI using CrossRef API with rate limiting and caching."""
    global _last_doi_check_time
    
    # Rate limiting - use config delay or default
    rate_limit_delay = getattr(config, 'doi_rate_limit_delay', 0.1) if config else 0.1
    current_time = time.time()
    if current_time - _last_doi_check_time < rate_limit_delay:
        time.sleep(rate_limit_delay)
    _last_doi_check_time = time.time()
    
    return _validate_doi_with_crossref_cached(doi)

def _validate_doi_with_crossref_uncached(doi: str) -> bool:
    """Validate a DOI using CrossRef API."""
    try:
        # Clean up DOI
        doi = doi.replace('\\', '').strip()
        if not doi.startswith('10.'):
            return False
        
        # Query CrossRef API
        url = f"https://api.crossref.org/works/{urllib.parse.quote(doi)}"
        request = urllib.request.Request(url)
        request.add_header('User-Agent', 'SciResearch-Workflow/1.0')
        
        with urllib.request.urlopen(request, timeout=5) as response:
            data = json.loads(response.read().decode())
            return 'status' in data and data['status'] == 'ok'
    except Exception:
        # If API call fails, assume DOI might be valid (don't block on network issues)
        return True

def _extract_simulation_code_with_validation(paper_path: Path, sim_path: Path) -> Tuple[bool, str]:
    """Extract simulation code with validation that it's complete."""
    try:
        original_content = sim_path.read_text() if sim_path.exists() else ""
        extract_simulation_from_tex(paper_path, sim_path)
        
        if not sim_path.exists():
            return False, "No simulation.py file was created"
        
        new_content = sim_path.read_text()
        
        # Validate that extracted code looks complete
        validation_issues = []
        
        if len(new_content.strip()) < 50:
            validation_issues.append("Extracted code is too short")
        
        if 'import' not in new_content and 'from' not in new_content:
            validation_issues.append("No import statements found")
        
        if 'def ' not in new_content and 'class ' not in new_content:
            validation_issues.append("No functions or classes found")
        
        # Check for common simulation patterns
        simulation_indicators = ['numpy', 'scipy', 'matplotlib', 'plot', 'calculate', 'simulate', 'random', 'range']
        if not any(indicator in new_content.lower() for indicator in simulation_indicators):
            validation_issues.append("Code doesn't appear to contain simulation logic")
        
        if validation_issues:
            return False, f"Simulation code validation failed: {'; '.join(validation_issues)}"
        
        return True, "Simulation code extracted and validated successfully"
    except Exception as e:
        return False, f"Error extracting simulation code: {str(e)}"

def _calculate_dynamic_timeout(paper_content: str, config: Optional[WorkflowConfig] = None) -> int:
    """Calculate dynamic timeout based on document complexity."""
    base_timeout = getattr(config, 'latex_timeout_base', 120) if config else 120  # Use config or default
    
    # Count complexity indicators
    tikz_count = len(re.findall(r'\\begin\{tikzpicture\}', paper_content))
    table_count = len(re.findall(r'\\begin\{tabular\}', paper_content))
    figure_count = len(re.findall(r'\\begin\{figure\}', paper_content))
    equation_count = len(re.findall(r'\\begin\{equation\}', paper_content))
    page_count = max(1, len(paper_content) // 3000)  # Rough estimate of pages
    
    # Add time for each complexity factor
    additional_time = (
        tikz_count * 30 +      # 30s per TikZ diagram
        table_count * 10 +     # 10s per table
        figure_count * 15 +    # 15s per figure
        equation_count * 5 +   # 5s per equation
        page_count * 10        # 10s per estimated page
    )
    
    total_timeout = min(base_timeout + additional_time, 600)  # Cap at 10 minutes
    print(f"Dynamic timeout calculated: {total_timeout}s (TikZ:{tikz_count}, Tables:{table_count}, Figures:{figure_count}, Pages:{page_count})")
    
    return total_timeout

def _validate_figure_generation(paper_content: str, sim_path: Path, project_dir: Path) -> List[str]:
    """Validate that simulation generates all figures referenced in the paper."""
    issues = []
    
    # Extract figure references from paper
    figure_refs = re.findall(r'\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}', paper_content)
    expected_figures = [fig for fig in figure_refs if not fig.endswith('.tex')]
    
    if not expected_figures:
        return issues  # No figures to validate
    
    # Check if simulation.py exists and contains figure generation code
    if not sim_path.exists():
        if expected_figures:
            issues.append(f"Paper references {len(expected_figures)} figures but simulation.py doesn't exist")
        return issues
    
    try:
        sim_content = sim_path.read_text()
        
        # Check for matplotlib/plotting imports
        has_plotting = any(lib in sim_content.lower() for lib in ['matplotlib', 'pyplot', 'seaborn', 'plotly'])
        if expected_figures and not has_plotting:
            issues.append("Paper references figures but simulation.py has no plotting libraries")
        
        # Check for savefig or similar save commands
        has_save_commands = any(cmd in sim_content.lower() for cmd in ['savefig', '.save(', 'plt.show', 'fig.write'])
        if expected_figures and not has_save_commands:
            issues.append("Paper references figures but simulation.py has no figure saving commands")
        
        # Check if referenced figure files exist in project directory
        missing_figures = []
        for fig in expected_figures:
            fig_path = project_dir / fig
            if not fig_path.exists():
                # Try common extensions
                found = False
                for ext in ['.png', '.pdf', '.svg', '.jpg']:
                    if (project_dir / (fig + ext)).exists():
                        found = True
                        break
                    if fig.endswith(ext) and (project_dir / fig).exists():
                        found = True
                        break
                if not found:
                    missing_figures.append(fig)
        
        if missing_figures:
            issues.append(f"Referenced figures not found in project directory: {', '.join(missing_figures[:3])}{'...' if len(missing_figures) > 3 else ''}")
    
    except Exception as e:
        issues.append(f"Error validating figure generation: {str(e)}")
    
    return issues

def _validate_figures_tables(paper_content: str) -> List[str]:
    """Validate figure and table formatting to prevent overflow and ensure self-containment."""
    issues = []
    
    # Check for external image dependencies
    external_images = re.findall(r'\\includegraphics\[[^\]]*\]\{([^}]+)\}', paper_content)
    external_files = [img for img in external_images if '.' in img and not img.endswith(('.tex', '.tikz'))]
    if external_files:
        issues.append(f"Found {len(external_files)} external image references - convert to TikZ/PGFPlots for self-containment")
    
    # Check for proper figure width constraints
    bad_figures = re.findall(r'\\includegraphics(?!\[[^\]]*width\s*=\s*\\linewidth)', paper_content)
    if bad_figures:
        issues.append(f"Found {len(bad_figures)} figures without width=\\linewidth constraint")
    
    # Check for oversized figures (width > \linewidth)
    oversized_figures = re.findall(r'\\includegraphics\[[^\]]*width\s*=\s*[^\\][^\]]*\]', paper_content)
    if oversized_figures:
        issues.append("Found figures with custom width that may exceed page margins")
    
    # Check for table captions
    table_count = len(re.findall(r'\\begin\{table\}', paper_content))
    caption_count = len(re.findall(r'\\caption\{', paper_content))
    if table_count > caption_count:
        issues.append(f"Some tables missing captions ({table_count} tables, {caption_count} captions)")
    
    # Check for tables without adjustbox wrapping
    unwrapped_tables = []
    table_blocks = re.finditer(r'\\begin\{table\}(.*?)\\end\{table\}', paper_content, re.DOTALL)
    for match in table_blocks:
        table_content = match.group(1)
        if '\\begin{tabular}' in table_content and 'adjustbox' not in table_content and 'resizebox' not in table_content:
            unwrapped_tables.append(match.group(0)[:50] + "...")
    
    if unwrapped_tables:
        issues.append(f"Found {len(unwrapped_tables)} tables without adjustbox width constraint")
    
    # Check for tikzpicture without size constraints
    tikz_pictures = re.findall(r'\\begin\{tikzpicture\}.*?\\end\{tikzpicture\}', paper_content, re.DOTALL)
    oversized_tikz = []
    for tikz in tikz_pictures:
        if 'adjustbox' not in tikz and 'width=' not in tikz and 'scale=' not in tikz:
            oversized_tikz.append(tikz[:50] + "...")
    
    if oversized_tikz:
        issues.append(f"Found {len(oversized_tikz)} tikzpicture diagrams without size constraints")
    
    # Check for placeholder or fake data in tables
    table_data = re.findall(r'\\begin\{tabular\}.*?\\end\{tabular\}', paper_content, re.DOTALL)
    for table in table_data:
        if any(placeholder in table.lower() for placeholder in ['xxx', 'placeholder', 'example', 'tbd', 'todo', 'n/a']):
            issues.append("Tables contain placeholder data - populate with real simulation results")
            break
    
    # CHECK FOR POOR TABLE/FIGURE POSITIONING
    # Find tables and figures with problematic positioning specifiers
    bad_table_positioning = re.findall(r'\\begin\{table\}\[([^\]]*[!]?[t][^\]]*)\]', paper_content)
    bad_figure_positioning = re.findall(r'\\begin\{figure\}\[([^\]]*[!]?[t][^\]]*)\]', paper_content)
    
    problematic_positions = []
    if bad_table_positioning:
        for pos in bad_table_positioning:
            if 't' in pos and ('h' not in pos or '!' in pos):
                problematic_positions.append(f"table[{pos}]")
    
    if bad_figure_positioning:
        for pos in bad_figure_positioning:
            if 't' in pos and ('h' not in pos or '!' in pos):
                problematic_positions.append(f"figure[{pos}]")
    
    if problematic_positions:
        issues.append(f"Found poor positioning specifiers forcing floats to page tops: {', '.join(problematic_positions[:3])}{'...' if len(problematic_positions) > 3 else ''} - use [h], [ht], or [H] for contextual placement")
    
    # Check for tables/figures without any positioning specifier (defaults to [tbp])
    unspecified_tables = len(re.findall(r'\\begin\{table\}(?!\[)', paper_content))
    unspecified_figures = len(re.findall(r'\\begin\{figure\}(?!\[)', paper_content))
    
    if unspecified_tables > 0 or unspecified_figures > 0:
        issues.append(f"Found {unspecified_tables + unspecified_figures} floats without explicit positioning specifiers - specify [h], [ht], or [H] for better contextual placement")

    return issues

def _validate_bibliography(paper_content: str) -> List[str]:
    """Check bibliography quality."""
    issues = []
    if not re.search(r'\\bibliography\{|\\begin\{thebibliography\}', paper_content):
        issues.append("No bibliography section found")
    
    # Check for reasonable number of references
    bib_entries = len(re.findall(r'\\bibitem\{|@\w+\{', paper_content))
    if bib_entries < 10:
        issues.append(f"Only {bib_entries} bibliography entries (recommend 10+)")
    
    return issues

if __name__ == "__main__":
    print("Starting Enhanced SciResearch Workflow...")
    try:
        ns = parse_args()
        
        # Load configuration
        config = None
        if ns.config:
            config = WorkflowConfig.from_file(Path(ns.config))
            print(f"Configuration loaded from: {ns.config}")
        else:
            config = WorkflowConfig()
            print("Using default configuration")
        
        # Save configuration if requested
        if ns.save_config:
            config.save_to_file(Path(ns.save_config))
            print(f" Configuration saved to: {ns.save_config}")
            sys.exit(0)
        
        print(f"Working with: {ns.output_dir}")
        print(f"Using model: {ns.model}")
        print(f"Max iterations: {ns.max_iterations}")
        print(f"Quality threshold: {ns.quality_threshold}")
        print(f"Early stopping: {'disabled' if ns.no_early_stopping else 'enabled'}")
        print(f"Reference validation: {'enabled' if ns.check_references else 'disabled'}")
        print(f" Figure validation: {'enabled' if ns.validate_figures else 'disabled'}")
        print(f"PDF review: {'enabled' if ns.enable_pdf_review else 'disabled'}")
        if ns.specify_idea:
            print(f"Research ideation: disabled (using specified idea: '{ns.specify_idea}')")
        else:
            print(f"Research ideation: {'enabled' if ns.enable_ideation else 'disabled'}")
        print(f"Diff output tracking: {'enabled' if ns.output_diffs else 'disabled'}")
        print(f"Content protection: {'enabled' if ns.enable_content_protection else 'DISABLED (DANGEROUS)'}")
        if hasattr(ns, 'auto_approve_changes') and ns.auto_approve_changes:
            print(f"Auto-approve changes: enabled")
        print(f"Content protection threshold: {ns.content_protection_threshold:.1%}")
        
        # Set configuration options
        config.enable_pdf_review = ns.enable_pdf_review
        config.content_protection = ns.enable_content_protection
        config.auto_approve_changes = getattr(ns, 'auto_approve_changes', False)
        config.content_protection_threshold = ns.content_protection_threshold
        config.no_early_stopping = ns.no_early_stopping
        
        # Handle test scaling mode
        if ns.test_scaling:
            print("\nStarting Test-Time Compute Scaling Analysis...")
            # Convert comma-separated string to list of integers
            candidate_counts_list = [int(x.strip()) for x in ns.scaling_candidates.split(',')]
            result = test_time_compute_scaling(
                model=ns.model,
                test_prompt=ns.scaling_prompt,
                candidate_counts=candidate_counts_list,
                timeout_base=ns.scaling_timeout
            )
            if result:
                print("\n Test-Time Compute Scaling Analysis Complete!")
            else:
                print("\n Test-Time Compute Scaling Analysis Failed!")
            sys.exit(0)
        
        # Apply test-time compute scaling configuration
        if hasattr(ns, 'use_test_time_scaling') and ns.use_test_time_scaling:
            config.use_test_time_scaling = True
            config.revision_candidates = ns.revision_candidates
            config.initial_draft_candidates = ns.draft_candidates
            print(f"Test-time compute scaling enabled: {config.revision_candidates} revision candidates")
        
        result_dir = run_workflow(
            topic=ns.topic,
            field=ns.field,
            question=ns.question,
            output_dir=Path(ns.output_dir),
            model=ns.model,
            request_timeout=(None if ns.request_timeout == 0 else ns.request_timeout),
            max_retries=ns.max_retries,
            max_iterations=ns.max_iterations,
            modify_existing=ns.modify_existing,
            strict_singletons=ns.strict_singletons,
            python_exec=ns.python_exec,
            quality_threshold=ns.quality_threshold,
            check_references=ns.check_references,
            validate_figures=ns.validate_figures,
            user_prompt=ns.user_prompt,
            config=config,
            enable_ideation=ns.enable_ideation,
            specify_idea=ns.specify_idea,
            num_ideas=ns.num_ideas,
            output_diffs=ns.output_diffs,
            document_type=ns.document_type
        )
        print(f" Workflow completed! Results in: {result_dir}")
    except Exception as e:
        print(f" Workflow failed: {e}")
        import traceback
        traceback.print_exc()
