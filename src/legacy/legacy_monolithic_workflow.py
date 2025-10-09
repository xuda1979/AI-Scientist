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

from dataclasses import dataclass, asdict
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

# Workflow step modules
from workflow_steps.initial_draft import generate_initial_draft
from workflow_steps.simulation import run_simulation_step
from workflow_steps.review_revision import run_review_revision_step, run_optimized_review_revision_step

# Acceptance readiness scoring module
try:
    from acceptance_readiness import calculate_acceptance_readiness  # new scoring module
except Exception:  # pragma: no cover
    calculate_acceptance_readiness = None  # type: ignore

DEFAULT_MODEL = os.environ.get("SCI_MODEL", "ollama:llama3.1:8b")

@dataclass
class WorkflowConfig:
    """Configuration class for workflow parameters."""
    quality_threshold: float = 0.8
    max_iterations: int = 10
    simulation_timeout: int = 300
    latex_timeout_base: int = 120
    api_retry_attempts: int = 3
    figure_validation: bool = True
    reference_validation: bool = True
    max_quality_history_size: int = 20
    enable_pdf_review: bool = False  # Enable PDF file review during revisions (disabled by default)
    doi_rate_limit_delay: float = 0.1
    max_doi_cache_size: int = 1000
    default_model: str = DEFAULT_MODEL
    fallback_models: Optional[List[str]] = None
    output_diffs: bool = False  # Optional diff output for each review/revision cycle
    
    # Test-time compute scaling parameters
    use_test_time_scaling: bool = False
    revision_candidates: int = 3
    initial_draft_candidates: int = 1
    
    # Combined approach - single API call for review/editorial/revision with diffs
    use_combined_approach: bool = True  # Always True - this is the only approach
    
    def __post_init__(self):
        if self.fallback_models is None:
            if str(self.default_model).startswith('ollama:'):
                self.fallback_models = []
            elif "gpt-5" in str(self.default_model):
                self.fallback_models = ["gpt-4o", "gpt-4"]
            else:
                self.fallback_models = ["gpt-3.5-turbo"]
    
    @classmethod
    def from_file(cls, config_path: Path) -> 'WorkflowConfig':
        """Load configuration from JSON file."""
        if not config_path.exists():
            return cls()  # Return default config
        
        try:
            with open(config_path) as f:
                data = json.load(f)
            return cls(**data)
        except (json.JSONDecodeError, TypeError) as e:
            print(f"[WARNING] Config file error: {e}. Using defaults.")
            return cls()
    
    def to_file(self, config_path: Path) -> None:
        """Save configuration to JSON file."""
        config_path.parent.mkdir(parents=True, exist_ok=True)
        with open(config_path, 'w') as f:
            json.dump(asdict(self), f, indent=2)
        print(f"[SUCCESS] Configuration saved to {config_path}")

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
        logger.info(f"API call successful for {model} ({prompt_type})")
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
            # Handle different API parameters for different model versions
            if model.startswith('gpt-5') or model.startswith('o1'):
                # GPT-5 and o1 models use max_completion_tokens and only support default temperature
                print(f"Sending request with default temperature (1.0), timeout={request_timeout}s (attempt {attempt + 1}/{max_retries})...")
                resp = client.chat.completions.create(
                    model=model, 
                    messages=processed_messages, 
                    timeout=request_timeout,
                    max_completion_tokens=4000
                    # Note: temperature not specified for GPT-5 (uses default of 1.0)
                )
            else:
                # GPT-4 and earlier models use max_tokens and support custom temperature
                print(f"Sending request with temperature={temp}, timeout={request_timeout}s (attempt {attempt + 1}/{max_retries})...")
                resp = client.chat.completions.create(
                    model=model, 
                    messages=processed_messages, 
                    temperature=temp, 
                    timeout=request_timeout,
                    max_tokens=4000
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
        'gpt-4o', 'gpt-4o-mini', 'gpt-5', 'gpt-5-pro'  # Add more vision-capable models as they become available
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
    # Normalize messages input (accept str or list[str])
    if isinstance(messages, str):
        messages = [{"role": "user", "content": messages}]
    elif isinstance(messages, list) and messages and isinstance(messages[0], str):
        messages = [{"role": "user", "content": m} for m in messages]

    # Detect provider based on model name
    if model.startswith(('gemini', 'models/gemini')):
        return _google_chat(messages, model, request_timeout, prompt_type, fallback_models, pdf_path)
    elif model.startswith('ollama:'):
        return _ollama_chat(messages, model, request_timeout, prompt_type, fallback_models, pdf_path)
    else:
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
    if os.getenv("OFFLINE_MODE") == "1":
        print("OFFLINE_MODE enabled - returning stub response without contacting API")
        return _offline_response(prompt_type)

    if not GOOGLE_AI_AVAILABLE:
        raise APIError("Google AI SDK not available. Please install with: pip install google-generativeai")
    
    # Set proxy specifically for Google AI API (not needed for OpenAI)
    original_proxy = os.environ.get("HTTPS_PROXY")
    gemini_proxy = os.environ.get("GEMINI_HTTPS_PROXY")
    if gemini_proxy:
        os.environ["HTTPS_PROXY"] = gemini_proxy
        print(f"Set proxy for Gemini API: {gemini_proxy}")
    
    if False and pdf_path and pdf_path.exists():  # PDF upload disabled
        print(f"INFO: Including PDF in Gemini request: {pdf_path.name}")
    
    try:
        # Configure API key - use hardcoded key as in reference
        # Configure API key from environment (no hardcoded keys)
        api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise APIError("Google AI API key not set. Set GOOGLE_API_KEY or GEMINI_API_KEY.")
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
        
        print("INFO: Google AI API call successful.")
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
    
    if modify_existing:
        # Check for any .tex file in the output directory
        tex_files = list(output_dir.glob("*.tex"))
        if tex_files:
            print(f"INFO: Found existing .tex files: {[f.name for f in tex_files]}")
            return output_dir
        else:
            print(f"WARNING: --modify-existing specified but no .tex files found in {output_dir}")
            print(f"INFO: Will create minimal template and proceed with modifications")
            return output_dir  # Return the output_dir itself, not a timestamped subdirectory
    
    # Create timestamped subdirectory for new projects
    project_dir = output_dir / _nowstamp()
    project_dir.mkdir(parents=True, exist_ok=True)
    print(f"INFO: Created new project directory: {project_dir}")
    return project_dir

def _generate_research_ideas(
    topic: str, 
    field: str, 
    question: str, 
    model: str,
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
    
    ideation_prompt = f"""
You are a brilliant research strategist tasked with generating innovative research ideas.

TOPIC: {topic}
FIELD: {field}
RESEARCH QUESTION: {question}

Your task is to generate {num_ideas} distinct, high-quality research ideas that address this topic/question in the field of {field}. For each idea, provide:

1. **Title**: A concise, descriptive title
2. **Core Concept**: 2-3 sentences describing the main research direction
3. **Originality Score**: 1-10 (10 = highly novel, never done before)
4. **Impact Score**: 1-10 (10 = revolutionary potential, broad applications)
5. **Feasibility Score**: 1-10 (10 = very feasible with current technology/methods)
6. **Pros**: 2-3 key advantages of this approach
7. **Cons**: 2-3 potential challenges or limitations

Please ensure ideas span different approaches: theoretical, experimental, algorithmic, systems-based, survey/analysis, etc.

Format your response as:

## Research Idea #1
**Title**: [Title]
**Core Concept**: [Description]
**Originality**: [1-10] - [Brief justification]
**Impact**: [1-10] - [Brief justification]  
**Feasibility**: [1-10] - [Brief justification]
**Pros**: 
- [Pro 1]
- [Pro 2]
**Cons**:
- [Con 1] 
- [Con 2]

## Research Idea #2
[Continue same format...]

After listing all {num_ideas} ideas, provide:

## RANKING ANALYSIS
Rank the top 5 ideas by overall potential (considering originality  impact  feasibility), and explain your ranking criteria.

## RECOMMENDATION
Select the single best idea and explain why it's optimal for development into a research paper.
"""

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
    """Parse the ideation response to extract structured idea data."""
    ideas: List[Dict[str, Any]] = []
    try:
        import re
        idea_sections = re.split(r'## Research Idea #\d+', response)
        for section in idea_sections[1:]:
            idea: Dict[str, Any] = {}
            title_match = re.search(r'\*\*Title\*\*:\s*(.+)', section)
            if title_match:
                idea['title'] = title_match.group(1).strip()
            concept_match = re.search(r'\*\*Core Concept\*\*:\s*(.+?)(?=\*\*|$)', section, re.DOTALL)
            if concept_match:
                idea['core_concept'] = concept_match.group(1).strip()
            for score_type in ['Originality', 'Impact', 'Feasibility']:
                pattern = fr'\*\*{score_type}\*\*:\s*(\d+)'
                match = re.search(pattern, section)
                if match:
                    idea[score_type.lower()] = int(match.group(1))
            pros_match = re.search(r'Pros?\*?:(.+?)(?=Cons?:|$)', section, re.DOTALL | re.IGNORECASE)
            if pros_match:
                idea['pros'] = [p.strip('- ').strip() for p in pros_match.group(1).strip().splitlines() if p.strip()]
            cons_match = re.search(r'Cons?\*?:(.+?)(?=\n\n|$)', section, re.DOTALL | re.IGNORECASE)
            if cons_match:
                idea['cons'] = [c.strip('- ').strip() for c in cons_match.group(1).strip().splitlines() if c.strip()]
            if idea:
                ideas.append(idea)
    except Exception as e:
        print(f"WARN: Failed to parse ideation response: {e}")
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
            # Create varied initial draft prompts to encourage diversity
            base_prompt = _initial_draft_prompt(topic, field, question, user_prompt)
            
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
            selected_idx = int(selection_match.group(1)) - 1
            if 0 <= selected_idx < len(valid_candidates):
                orig_idx, best_candidate = valid_candidates[selected_idx]
                print(f"  LLM selected candidate {selected_idx + 1} (original index {orig_idx + 1})")
                reasoning_match = re.search(r'REASONING:\s*(.*?)(?=SELECTED:|$)', best_candidate_response, re.DOTALL | re.IGNORECASE)
                if reasoning_match:
                    reasoning = reasoning_match.group(1).strip()
                    print(f"   Selection reasoning: {reasoning[:200]}...")
                if output_diffs:
                    print("\nFINAL SELECTED CANDIDATE DIFF:")
                    _save_candidate_diff(current_tex, best_candidate, selected_idx + 1, "SELECTED")
                total_time = sum(t for t in generation_times if t)
                print(f"    Total compute time: {total_time:.1f}s")
                return best_candidate
            else:
                print(f"   Invalid selection index {selected_idx + 1}, using first candidate")
                return valid_candidates[0][1]
        else:
            print("  No explicit selection found in LLM response; defaulting to first candidate")
            return valid_candidates[0][1]
    except Exception as e:
        print(f"WARN: Failed to parse selection response: {e}; using first candidate")
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
    latex_indicators = ['\\begin{filecontents}', '\\begin{document}', '\\end{document}', '\\section', '\\subsection', '\\cite{', 
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
    math_indicators = ['\\begin{equation}', '\\begin{align}', '$', '\\(', 'theorem', 'proof', 'lemma']
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
            base_prompt = _initial_draft_prompt(topic, field, question, user_prompt)
            if i > 0:
                variation_instructions = [
                    "\nEmphasize theoretical rigor and mathematical formulations in your approach.",
                    "\nFocus on practical applications and experimental validation methods.",
                    "\nPrioritize comprehensive literature review and related work analysis.",
                    "\nConcentrate on novel algorithmic contributions and implementation details.",
                    "\nHighlight the broader impact and interdisciplinary connections."
                ]
                variation = variation_instructions[(i - 1) % len(variation_instructions)]
                varied_prompt = base_prompt.copy()
                varied_prompt[0]["content"] += variation
            else:
                varied_prompt = base_prompt
            candidate = _universal_chat(
                varied_prompt,
                model=model,
                request_timeout=request_timeout,
                prompt_type="initial_draft",
                fallback_models=[]
            )
        except Exception as e:
            print(f"WARNING: Draft candidate {i+1} generation failed: {e}")
            candidate = ""
        end_time = time.time()
        generation_times.append(end_time - start_time)
        candidates.append(candidate)
    
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

def _initial_draft_prompt(topic: str, field: str, question: str, user_prompt: Optional[str] = None) -> List[Dict[str, str]]:
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
        "9. APPROPRIATE STRUCTURE: The paper structure and sections must align with the paper's contribution type and field conventions:\n"
        "   - Theoretical papers: Abstract, Introduction, Related Work, Theory/Methods, Analysis, Discussion, Conclusion\n"
        "   - Experimental papers: Abstract, Introduction, Related Work, Methodology, Experiments, Results, Discussion, Conclusion\n"
        "   - Survey papers: Abstract, Introduction, Background, Classification/Taxonomy, Comparative Analysis, Future Directions, Conclusion\n"
        "   - Systems papers: Abstract, Introduction, Related Work, System Design, Implementation, Evaluation, Discussion, Conclusion\n"
        "   - Algorithm papers: Abstract, Introduction, Related Work, Problem Definition, Algorithm Description, Analysis, Experiments, Conclusion\n"
        "10. RESULTS DOCUMENTATION: The numerical results from running simulation.py MUST be saved in 'results.txt' in the project folder for reproducibility and verification.\n"
        "11. FIGURE GENERATION: If the paper includes figures generated by code, ALL figure generation code must be included in simulation.py and figures must be saved in the local project folder.\n"
        "12. SINGLE CODE FILE: ALL computational code must be consolidated into ONE simulation.py file - no additional .py files, scripts, or code fragments.\n\n"
        
        "STRUCTURE ALIGNMENT REQUIREMENTS:\n"
        "- Identify the paper type based on the research question and field\n"
        "- Use appropriate section names and organization for that paper type\n"
        "- Include field-specific sections (e.g., 'Threat Model' for security papers, 'Clinical Validation' for medical papers)\n"
        "- Follow established conventions for the target journal/conference\n"
        "- Ensure logical flow appropriate for the paper's contribution type\n"
        "- Include appropriate evaluation methodology for the paper type\n\n"
        
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
        "- References must support claims made in the text\n"
        "- Check for citation format consistency\n\n"
        
        "CONTENT SELF-CONTAINMENT:\n"
        "- Figures: Use TikZ, PGFPlots, or pure LaTeX constructs for diagrams; \\includegraphics for simulation-generated figures\n"
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
        "- For figures/tables at document end, place them immediately before \\bibliography{} or \\begin{thebibliography}\n\n"
        
        "STRUCTURE REQUIREMENTS:\n"
        "- Start with \\begin{filecontents*}{refs.bib}...\\end{filecontents*} containing ALL bibliography entries\n"
        "- Follow with standard LaTeX document structure\n"
        "- Organize sections according to paper's contribution type and field conventions\n"
        "- End with \\bibliography{refs} to use the embedded references\n\n"
        
        "FORMATTING REQUIREMENTS:\n"
        "- All content must use width=\\linewidth constraints\n"
        "- Wrap wide tables using adjustbox width=\\linewidth\n"
        "- ALL TikZ diagrams MUST be constrained (use adjustbox or scale to fit page width)\n"
        "- ALL content must fit within vertical page margins\n"
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
        "- Tables: \\begin{adjustbox}{width=\\linewidth}...\\end{adjustbox} (for wide tables)\n"
        "- Plots: \\begin{tikzpicture}\\begin{axis}\\addplot coordinates {(1,2) (3,4)};\\end{axis}\\end{tikzpicture} (for PGFPlots)\n\n"
        
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
    
    user = (
        f"Create a comprehensive research paper on the topic: {topic}\n"
        f"Field: {field}\n"
        f"Research Question: {question}\n\n"
        "Provide a complete LaTeX paper that addresses this research question with proper structure, methodology, and authentic references."
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
    quality_threshold: float = 0.8,
    check_references: bool = True,
    validate_figures: bool = True,
    user_prompt: Optional[str] = None,
    config: Optional[WorkflowConfig] = None,
    enable_ideation: bool = True,
    num_ideas: int = 15,
    output_diffs: bool = False,
    use_acceptance_readiness: bool = True,
    continuous: bool = False,
) -> Path:
    """Enhanced workflow with quality validation, progress tracking, and custom user prompts.
    
    SAFEGUARDS IMPLEMENTED:
    - Function parameter 'max_iterations' is explicitly invalidated after config assignment
    - Only 'config.max_iterations' should be used throughout the function
    - Multiple validation checks ensure iteration count consistency
    - Explicit logging of expected vs actual iteration counts
    - Runtime assertions prevent invalid configuration values
    """
    
    # Use provided config or load default
    if config is None:
        config = WorkflowConfig()
    
    # Override config values with explicit parameters if provided
    if quality_threshold != 0.8:  # Only override if explicitly set
        config.quality_threshold = quality_threshold
    if max_iterations != 4:
        config.max_iterations = max_iterations
    if check_references is not True:
        config.reference_validation = check_references
    if validate_figures is not True:
        config.figure_validation = validate_figures
    
    # SAFEGUARD: Clear function parameter to prevent scope confusion
    # From this point forward, ONLY use config.max_iterations
    # pylint: disable-next=redefined-argument-from-local
    max_iterations = None  # Explicitly invalidate to prevent future mistakes
    # DO NOT USE 'max_iterations' VARIABLE ANYWHERE BELOW THIS LINE
    # ALWAYS USE 'config.max_iterations' INSTEAD
    
    # VALIDATION: Ensure configuration is properly set
    assert config.max_iterations > 0, f"Invalid max_iterations: {config.max_iterations}"
    assert config.quality_threshold > 0, f"Invalid quality_threshold: {config.quality_threshold}"
    
    logger.info(f"Starting workflow with config: quality_threshold={config.quality_threshold}, max_iterations={config.max_iterations}")
    logger.info(f"VALIDATED: Will run exactly {config.max_iterations} iteration(s) maximum")
    
    project_dir = _prepare_project_dir(output_dir, modify_existing)
    paper_path, sim_path = ensure_single_tex_py(project_dir, strict=strict_singletons, preserve_original_filename=modify_existing)

    # Show which file is being used
    print(f"Working with paper file: {paper_path.name}")

    # Progress tracking variables
    quality_history = []
    stagnation_count = 0
    best_quality_score = 0.0
    
    # Optimization tracking variables
    last_tex_hash = None
    last_latex_success = False
    
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
        print("  - 'Concentrate on fundamental enhancements and optimizations'")
        print("\nLeave empty to use standard prompts only.")
        print("-" * 60)
        
        user_prompt = "" if not (sys.stdin and sys.stdin.isatty()) else input("Enter your custom prompt (or press Enter to skip): ").strip()
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
        print(f"   All references must be embedded in {paper_path.name} using filecontents or thebibliography")
        print("   Separate .bib files will be ignored during compilation!")

    # Check if this is actually a minimal template (new paper) or has real content
    paper_content = paper_path.read_text(encoding="utf-8").strip()
    is_minimal_template = (paper_content == "\\documentclass{article}\\begin{document}\\end{document}" or len(paper_content) < 200)
    
    # If no real paper content yet (fresh), run ideation and then draft one.
    if is_minimal_template:
        if enable_ideation:
            print("Starting Ideation + Initial Draft Phase for new paper...")
            
            # OPTIMIZED: Combined ideation and draft generation in one call
            combined_ideation_draft_prompt = f"""
You are a brilliant research strategist and academic writer. Please generate innovative research ideas and create an initial draft paper.

TOPIC: {final_topic if 'final_topic' in locals() else topic}
FIELD: {field}
RESEARCH QUESTION: {question}
USER REQUIREMENTS: {user_prompt or "Standard academic quality"}

TASK 1 - IDEATION: Generate {num_ideas} research ideas covering different approaches (theoretical, experimental, algorithmic, systems-based, etc.). For each idea provide:
- Title & Core Concept
- Originality/Impact/Feasibility scores (1-10)
- Key pros and cons

TASK 2 - SELECTION: Select the best idea based on overall potential.

TASK 3 - DRAFT GENERATION: Create a complete LaTeX research paper draft based on the selected idea, including:
- Proper LaTeX structure with documentclass, abstract, sections
- Introduction with motivation and contributions
- Related work section
- Methodology/approach section
- Experimental setup or theoretical analysis
- Results section (with placeholder for simulation outputs)
- Conclusion and future work
- Bibliography with relevant citations
- Embedded Python simulation code in appropriate sections

Format your response as:
## IDEATION ANALYSIS
[Brief analysis of top ideas]

## SELECTED RESEARCH DIRECTION
**Title**: [Selected idea title]
**Rationale**: [Why this idea was chosen]

## COMPLETE LATEX PAPER
```latex
[Full LaTeX paper here]
```

Focus on creating a substantial, publication-ready draft that integrates the best research direction.
"""
            
            combined_response = _universal_chat(
                [{"role": "user", "content": combined_ideation_draft_prompt}],
                model=model,
                request_timeout=request_timeout,
                prompt_type="combined_ideation_draft",
                fallback_models=config.fallback_models,
            )
            
            # Parse the combined response
            import re
            draft_match = re.search(r'##\s*COMPLETE\s+LATEX\s+PAPER\s*\n```(?:latex|tex)?\s*\n(.*?)\n```', combined_response, re.DOTALL | re.IGNORECASE)
            if draft_match:
                draft = draft_match.group(1).strip()
                
                # Extract selected idea info for logging
                idea_match = re.search(r'##\s*SELECTED\s+RESEARCH\s+DIRECTION\s*\n\*\*Title\*\*:\s*([^\n]+)', combined_response, re.IGNORECASE)
                selected_title = idea_match.group(1).strip() if idea_match else "Generated Research Idea"
                
                print(f"✓ Combined ideation and draft generation completed")
                print(f"  Selected Research Direction: {selected_title}")
                
                # Save ideation results to project directory
                ideation_file = project_dir / "ideation_analysis.txt"
                with open(ideation_file, 'w', encoding='utf-8') as f:
                    f.write("COMBINED IDEATION AND DRAFT ANALYSIS\n")
                    f.write("=" * 50 + "\n\n")
                    f.write(f"Original Topic: {topic}\n")
                    f.write(f"Original Question: {question}\n")
                    f.write(f"Field: {field}\n\n")
                    f.write(f"Selected Research Direction: {selected_title}\n\n")
                    f.write("FULL RESPONSE:\n")
                    f.write("-" * 30 + "\n")
                    f.write(combined_response)
                
                print(f"Combined analysis saved to: {ideation_file}")
            else:
                print("⚠ Could not parse combined response, falling back to separate calls")
                # Fallback to original separate approach
                ideation_result = _generate_research_ideas(
                    topic=topic,
                    field=field,
                    question=question,
                    model=model,
                    num_ideas=num_ideas,
                    request_timeout=request_timeout,
                    fallback_models=config.fallback_models
                )
                
                selected_idea = ideation_result.get("selected_idea")
                if selected_idea:
                    final_topic = selected_idea.get("title", topic)
                    final_question = selected_idea.get("core_concept", question)
                else:
                    final_topic = topic
                    final_question = question
                
                draft = generate_initial_draft(
                    final_topic,
                    field,
                    final_question,
                    user_prompt,
                    model,
                    request_timeout,
                    config,
                )
        else:
            print("  Ideation phase skipped (--skip-ideation flag used)")
            draft = generate_initial_draft(
                topic,
                field,
                question,
                user_prompt,
                model,
                request_timeout,
                config,
            )
        
        paper_path.write_text(draft, encoding="utf-8")
        print(" Paper draft created successfully")
    else:
        print(f"Using existing paper content ({len(paper_content)} characters)")
        print("  Skipping ideation phase for existing paper")

    # Extract/refresh simulation.py from LaTeX initially
    extract_simulation_from_tex(paper_path, sim_path)
    last_sim_content_hash = None
    last_sim_summary = ""

    # Review-Revise loop with quality tracking
    # SAFEGUARD: Explicit validation before starting iterations
    expected_iterations = config.max_iterations
    assert expected_iterations > 0, f"Invalid iteration count: {expected_iterations}"
    
    logger.info(f"LOOP START: Beginning {expected_iterations} iteration(s)")
    print(f"VALIDATED: Will run exactly {expected_iterations} iteration(s)")
    
    for i in range(1, expected_iterations + 1):
        logger.info(f"ITERATION {i}/{expected_iterations}: Starting")
        print(f"Starting iteration {i} of {expected_iterations}")
        
        # SAFEGUARD: Double-check we're not exceeding expected count
        if i > expected_iterations:
            error_msg = f"CRITICAL ERROR: Iteration {i} exceeds max_iterations {expected_iterations}"
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        # OPTIMIZED: Only run simulation if code changed or first iteration
        sim_content = sim_path.read_text(encoding="utf-8", errors="ignore") if sim_path.exists() else ""
        import hashlib
        current_sim_hash = hashlib.md5(sim_content.encode()).hexdigest()
        
        if current_sim_hash != last_sim_content_hash:
            print(f"Running simulation (code changed)...")
            logger.info(f"ITERATION {i}: Running simulation")
            
            sim_summary, _ = run_simulation_step(
                paper_path,
                sim_path,
                project_dir,
                model,
                request_timeout,
                python_exec,
            )
            last_sim_content_hash = current_sim_hash
            last_sim_summary = sim_summary
        else:
            print(f"Reusing simulation results (code unchanged)...")
            sim_summary = last_sim_summary
        
        # OPTIMIZED: Only compile LaTeX if content changed or compilation previously failed
        current_tex = paper_path.read_text(encoding="utf-8", errors="ignore")
        current_tex_hash = hashlib.md5(current_tex.encode()).hexdigest()
        
        if (current_tex_hash != last_tex_hash or not last_latex_success):
            print(f"Compiling LaTeX file with pdflatex...")
            dynamic_timeout = _calculate_dynamic_timeout(current_tex, config)
            latex_success, latex_errors = _compile_latex_and_get_errors(paper_path, timeout=dynamic_timeout)
            
            last_tex_hash = current_tex_hash
            last_latex_success = latex_success
            
            if not latex_success:
                print(f" LaTeX compilation failed. Errors will be sent to LLM for fixing.")
                print(f"Error log (last 20 lines):\n{latex_errors}")
            else:
                print(f" LaTeX compilation successful!")
        else:
            print(f"Reusing LaTeX compilation result (content unchanged)...")
            latex_success = last_latex_success
            latex_errors = ""
        
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
            print(f"Quality issues detected: {', '.join(quality_issues[:5])}{'...' if len(quality_issues) > 5 else ''}")
        
        # CALCULATE QUALITY / READINESS METRICS AND TRACK PROGRESS
        current_metrics = _extract_quality_metrics(current_tex, sim_summary)
        readiness_breakdown = None
        if calculate_acceptance_readiness is not None:
            try:
                readiness_breakdown = calculate_acceptance_readiness(current_tex, sim_summary, quality_issues)
                quality_score = readiness_breakdown.penalty_adjusted_score / 100.0  # Convert to 0-1 scale
            except Exception as e:
                print(f"WARNING: Acceptance readiness calculation failed: {e}")
                quality_score = _calculate_quality_score(current_metrics, quality_issues)
        else:
            quality_score = _calculate_quality_score(current_metrics, quality_issues)
        quality_history.append(quality_score)

        # PROGRESS SUMMARY
        print("--- ITERATION SUMMARY ---")
        print(f"Iteration {i} score: {quality_score:.3f} (best so far: {best_quality_score:.3f})")
        print(f"Issues this iteration: {len(quality_issues)}")
        if readiness_breakdown:
            print(f"  Structure: {readiness_breakdown.structure_score:.2f}")
            print(f"  References: {readiness_breakdown.references_score:.2f}")
            print(f"  Methodology: {readiness_breakdown.methodology_score:.2f}")
            print(f"  Evaluation: {readiness_breakdown.evaluation_score:.2f}")
            print(f"  Rigor: {readiness_breakdown.rigor_score:.2f}")
            print(f"  Reproducibility: {readiness_breakdown.reproducibility_score:.2f}")
            print(f"  Clarity: {readiness_breakdown.clarity_score:.2f}")
            print(f"  Formatting: {readiness_breakdown.formatting_score:.2f}")
            print(f"  Base score: {readiness_breakdown.base_score:.1f}/100")
            print(f"  Final score: {readiness_breakdown.penalty_adjusted_score:.1f}/100")
            print(f"  Acceptance probability: {readiness_breakdown.acceptance_probability:.1%}")
            if readiness_breakdown.major_issues:
                print(f"  Major issues: {', '.join(readiness_breakdown.major_issues[:3])}{'...' if len(readiness_breakdown.major_issues) > 3 else ''}")
            if readiness_breakdown.minor_issues:
                print(f"  Minor issues: {', '.join(readiness_breakdown.minor_issues[:3])}{'...' if len(readiness_breakdown.minor_issues) > 3 else ''}")
        print("-------------------------")
        
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
                print(f"PDF generated for AI review: {pdf_path.name}")
            else:
                print(f" PDF generation failed: {pdf_error}")
        elif not config.enable_pdf_review:
            print(f" PDF review disabled in configuration")
        else:
            print(f" Skipping PDF generation due to LaTeX compilation failure")
        
        # OPTIMIZED: Single comprehensive prompt combining all context
        comprehensive_prompt = f"""
You are an expert academic reviewer and editor. Please provide a comprehensive review and revision of this research paper.

**CURRENT PAPER (Iteration {i}/{expected_iterations}):**
{current_tex}

**SIMULATION RESULTS:**
{sim_summary}

**LATEX COMPILATION STATUS:**
{latex_errors if latex_errors else "✓ Compilation successful"}

**QUALITY ISSUES DETECTED:**
{chr(10).join(f"- {issue}" for issue in quality_issues) if quality_issues else "✓ No major quality issues detected"}

**USER REQUIREMENTS:**
{user_prompt or "Standard academic quality requirements"}

**COMPREHENSIVE TASK:**
1. **REVIEW:** Provide detailed academic review covering:
   - Content quality and rigor
   - Methodology soundness  
   - Results interpretation
   - Writing clarity and flow
   - Technical accuracy

2. **REVISE:** Provide complete revised paper that:
   - Addresses all identified issues
   - Fixes LaTeX compilation errors
   - Improves academic rigor and clarity
   - Maintains proper formatting
   - Integrates simulation results effectively

Please format your response as:
## REVIEW
[Detailed review here]

## REVISED_PAPER
```latex
[Complete revised LaTeX paper here]
```

Focus on significant improvements that advance toward publication quality.
"""

        review, decision = run_optimized_review_revision_step(
            comprehensive_prompt,
            project_dir,
            user_prompt,
            i,
            model,
            request_timeout,
            config,
            pdf_path,
            output_diffs,
            paper_path,
        )

        print(f"Review completed")
        print(f"Review complete, applying revisions...")

        # SIMPLIFIED STOPPING LOGIC - DISABLED QUALITY THRESHOLD FOR FULL ITERATIONS
        meets_quality_threshold = quality_score >= config.quality_threshold
        
        # Only log stagnation; do NOT stop automatically
        if stagnation_count >= 3:  # previously caused early stop
            print(f"[INFO] Quality stagnation ({stagnation_count} iterations without improvement); continuing as requested (no early stop).")
        
        if latex_success and meets_quality_threshold:
            print(f"[INFO] Quality threshold met at iteration {i} (score: {quality_score:.2f}) and LaTeX compiles successfully - continuing...")
        
        print(f"Iteration {i}: Combined review and revision completed")
    
    # Final quality report
    print(f"\nQuality progression: {[f'{q:.2f}' for q in quality_history]}")
    print(f"Best quality score achieved: {best_quality_score:.2f}")
    
    # FINAL SAFEGUARD: Verify we completed the expected number of iterations
    completed_iterations = len(quality_history)
    logger.info(f"WORKFLOW COMPLETE: {completed_iterations} iteration(s) completed, expected max {expected_iterations}")
    if completed_iterations > expected_iterations:
        logger.warning(f"WARNING: Completed {completed_iterations} iterations but max was {expected_iterations}")
    
    print(f"FINAL VALIDATION: Completed {completed_iterations} of max {expected_iterations} iteration(s)")

    return project_dir


def _calculate_dynamic_timeout(tex_content: str, config: Any) -> int:
    """Calculate dynamic timeout based on document complexity."""
    base_timeout = 30
    # Add time based on content length
    timeout = base_timeout + len(tex_content) // 10000
    return min(timeout, 300)  # Cap at 5 minutes


def _compile_latex_and_get_errors(paper_path: Path, timeout: int = 30) -> Tuple[bool, str]:
    """Compile LaTeX and return success status and error output."""
    import subprocess
    try:
        # Use the filename only, run from the directory containing the file
        result = subprocess.run(
            ["pdflatex", "-interaction=nonstopmode", paper_path.name],
            cwd=paper_path.parent,
            capture_output=True,
            text=True,
            timeout=timeout
        )
        return result.returncode == 0, result.stdout + result.stderr
    except Exception as e:
        return False, str(e)


def _validate_research_quality(tex_content: str, sim_summary: str) -> List[str]:
    """Validate research quality and return list of issues."""
    issues = []
    if len(tex_content) < 1000:
        issues.append("Paper too short")
    if "\\begin{abstract}" not in tex_content:
        issues.append("Missing abstract")
    if "\\section" not in tex_content:
        issues.append("Missing sections")
    return issues


def _validate_references_with_external_apis(tex_content: str, config: Any) -> List[str]:
    """Validate references (placeholder implementation)."""
    issues = []
    if "\\cite{" not in tex_content and "\\bibitem{" not in tex_content:
        issues.append("No citations found")
    return issues


def _validate_figure_generation(tex_content: str, sim_path: Path, project_dir: Path) -> List[str]:
    """Validate figure generation (placeholder implementation)."""
    issues = []
    if "\\includegraphics" not in tex_content and "\\begin{figure}" not in tex_content:
        issues.append("No figures found")
    return issues


def _extract_quality_metrics(tex_content: str, sim_summary: str) -> Dict[str, Any]:
    """Extract quality metrics from paper and simulation."""
    return {
        'has_abstract': '\\begin{abstract}' in tex_content,
        'has_related_work': 'related work' in tex_content.lower(),
        'has_methodology': 'method' in tex_content.lower(),
        'has_results': 'results' in tex_content.lower(),
        'has_discussion': 'discussion' in tex_content.lower(),
        'has_conclusion': 'conclusion' in tex_content.lower(),
        'section_count': tex_content.count('\\section{'),
        'figure_count': tex_content.count('\\includegraphics'),
        'table_count': tex_content.count('\\begin{table}'),
        'citation_count': tex_content.count('\\cite{'),
        'equation_count': tex_content.count('\\begin{equation}'),
        'word_count': len(tex_content.split()),
        'has_simulation': 'simulation code:' in sim_summary,
        'simulation_success': 'error' not in sim_summary.lower()
    }


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
    score += min(metrics.get('section_count', 0) * 2, 10)
    score += min(metrics.get('citation_count', 0) * 0.5, 10)
    score += min((metrics.get('figure_count', 0) + metrics.get('table_count', 0)) * 2, 10)
    
    # Technical quality (0-20 points)
    if metrics.get('has_simulation'): score += 10
    if metrics.get('simulation_success'): score += 10
    
    # Deduct for issues (0-10 penalty)
    issue_penalty = min(len(issues) * 2, 10)
    score -= issue_penalty
    
    # Normalize to 0-1
    return max(0.0, min(1.0, score / 90.0))


def _generate_pdf_for_review(paper_path: Path, timeout: int) -> Tuple[bool, Optional[Path], str]:
    """Generate PDF for review purposes."""
    try:
        success, errors = _compile_latex_and_get_errors(paper_path, timeout)
        if success:
            pdf_path = paper_path.with_suffix('.pdf')
            return True, pdf_path if pdf_path.exists() else None, ""
        else:
            return False, None, errors
    except Exception as e:
        return False, None, str(e)


def _create_simulation_fix_prompt(error_msg: str, current_code: str) -> str:
    """Create a prompt to fix simulation code errors."""
    return f"""
SIMULATION CODE DEBUGGING

The simulation code encountered an error:
{error_msg}

Current code:
{current_code}

Please provide the corrected simulation code that:
1. Fixes the specific error mentioned above
2. Maintains the same functionality
3. Includes proper error handling
4. Saves results to 'results.txt'
5. Is complete and runnable

Return ONLY the corrected Python code, nothing else.
"""


def _combined_review_edit_revise_prompt(tex_content, sim_summary, latex_errors, project_dir, user_prompt, iteration):
    """
    Create a combined review, edit, and revise prompt for paper improvement
    """
    prompt = f"""
Review and revise this research paper (iteration {iteration}):

Paper content:
{tex_content}

Simulation summary:
{sim_summary}

LaTeX compilation errors:
{latex_errors}

User requirements: {user_prompt}

Please:
1. Review the paper for quality and completeness
2. Identify specific areas for improvement
3. Provide a revised version of the paper
4. Ensure all LaTeX compilation errors are fixed

Focus on academic rigor, clarity, and proper formatting.
"""
    return prompt



def _parse_combined_response(response, project_dir):
    """
    Parse the combined review/edit/revise response from the AI
    """
    text = response if isinstance(response, str) else str(response)
    decision = 'revised'
    file_changes = {}
    try:
        import re as _re
        m = _re.search(r"##\s*REVISION[\s\S]*?`(?:tex|latex)?\s*(.*?)`", text, _re.IGNORECASE)
        if m:
            file_changes['paper.tex'] = m.group(1).strip()
            review = _re.split(r"##\s*REVISION", text, 1, flags=_re.IGNORECASE)[0].strip()
            return review, decision, file_changes
    except Exception:
        pass
    mid = len(text) // 2
    review = text[:mid].strip()
    revised = text[mid:].strip()
    if revised:
        file_changes['paper.tex'] = revised
    return review, decision, file_changes
def _apply_file_changes(changes, project_dir):
    """
    Apply file changes to the project directory
    """
    # Simple implementation - just write content to files
    for file_path, content in changes.items():
        full_path = Path(project_dir) / file_path
        full_path.write_text(content, encoding='utf-8')
    return True


def _revise_prompt(tex_content, review_feedback, project_dir, user_prompt):
    """
    Create a revision prompt based on review feedback
    """
    return f"""
Please revise this research paper based on the review feedback:

Current paper:
{tex_content}

Review feedback:
{review_feedback}

User requirements: {user_prompt}

Please provide a revised version that addresses all the feedback points.
"""


def _extract_simulation_code_with_validation(paper_path: Path, sim_path: Path) -> Tuple[bool, str]:
    """Extract simulation code from LaTeX content with validation."""
    import re
    
    try:
        # Read the paper content
        tex_content = paper_path.read_text(encoding="utf-8", errors="ignore")
        
        # Look for Python code blocks in LaTeX
        patterns = [
            r'\\begin\{lstlisting\}(.*?)\\end\{lstlisting\}',
            r'\\begin\{verbatim\}(.*?)\\end\{verbatim\}',
            r'\\begin\{python\}(.*?)\\end\{python\}',
            r'```python(.*?)```'
        ]
        
        code_blocks = []
        for pattern in patterns:
            matches = re.findall(pattern, tex_content, re.DOTALL)
            code_blocks.extend(matches)
        
        if code_blocks:
            # Clean up the code
            code = '\n'.join(code_blocks)
            code = re.sub(r'\\begin\{.*?\}', '', code)
            code = re.sub(r'\\end\{.*?\}', '', code)
            code = code.strip()
        else:
            # Generate a default simulation if no code found
            code = '''
import numpy as np
import matplotlib.pyplot as plt

# Default simulation for the paper
def main():
    print("Running simulation...")
    
    # Generate some sample data
    x = np.linspace(0, 10, 100)
    y = np.sin(x) + 0.1 * np.random.randn(100)
    
    # Create a simple plot
    plt.figure(figsize=(8, 6))
    plt.plot(x, y, 'b-', label='Data')
    plt.xlabel('X values')
    plt.ylabel('Y values')
    plt.title('Sample Results')
    plt.legend()
    plt.grid(True)
    plt.savefig('results_plot.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Save results
    with open('results.txt', 'w') as f:
        f.write(f"Sample Results\\n")
        f.write(f"Mean: {np.mean(y):.3f}\\n")
        f.write(f"Std: {np.std(y):.3f}\\n")
        f.write(f"Min: {np.min(y):.3f}\\n")
        f.write(f"Max: {np.max(y):.3f}\\n")
    
    print("Simulation completed. Results saved to results.txt")

if __name__ == "__main__":
    main()
'''
        
        # Write the code to simulation file
        sim_path.write_text(code, encoding="utf-8")
        return True, f"Simulation code extracted and saved to {sim_path}"
        
    except Exception as e:
        return False, f"Failed to extract simulation code: {str(e)}"


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Research workflow with simulation and iterative review/revision")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="AI model to use")
    parser.add_argument("--max-iterations", type=int, default=5, help="Maximum number of iterations")
    parser.add_argument("--quality-threshold", type=float, default=0.8, help="Quality threshold for early stopping")
    parser.add_argument("--output-dir", type=Path, default=Path("output"), help="Output directory")
    parser.add_argument("--modify-existing", action="store_true", help="Modify existing paper in output directory")
    parser.add_argument("--strict-singletons", action="store_true", help="Enforce single .tex and .py files")
    parser.add_argument("--request-timeout", type=int, default=3600, help="Request timeout in seconds")
    parser.add_argument("--python-exec", default="python", help="Python executable to use")
    parser.add_argument("--topic", default="machine learning", help="Research topic")
    parser.add_argument("--field", default="computer science", help="Research field")
    parser.add_argument("--question", default="How can we improve model performance?", help="Research question")
    parser.add_argument("--skip-ideation", action="store_true", help="Skip ideation phase")
    parser.add_argument("--num-ideas", type=int, default=15, help="Number of ideas to generate")
    parser.add_argument("--user-prompt", help="Custom user prompt")
    parser.add_argument("--check-references", action="store_true", help="Enable reference validation")
    parser.add_argument("--validate-figures", action="store_true", help="Enable figure validation")
    parser.add_argument("--output-diffs", action="store_true", help="Output diffs between iterations")
    
    args = parser.parse_args()
    
    try:
        project_dir = run_workflow(
            model=args.model,
            max_iterations=args.max_iterations,
            quality_threshold=args.quality_threshold,
            output_dir=args.output_dir,
            modify_existing=args.modify_existing,
            strict_singletons=args.strict_singletons,
            request_timeout=args.request_timeout,
            python_exec=args.python_exec,
            topic=args.topic,
            field=args.field,
            question=args.question,
            enable_ideation=not args.skip_ideation,
            num_ideas=args.num_ideas,
            user_prompt=args.user_prompt,
            check_references=args.check_references,
            validate_figures=args.validate_figures,
            output_diffs=args.output_diffs
        )
        print(f"\nWorkflow completed successfully!")
        print(f"Project directory: {project_dir}")
    except Exception as e:
        print(f"ERROR: Workflow failed: {e}")
        import traceback
        traceback.print_exc()
        exit(1)



