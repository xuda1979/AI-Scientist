#!/usr/bin/env python3
"""
Extended workflow:
 - Enforce single paper.tex and simulation.py per project
 - Extract simulation code from LaTeX; run it; pass results to LLM during review/revision
 - Sanitize LaTeX to prevent overflow; compile-check; auto-fix on failure
"""
from __future__ import annotations
import argparse
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
from utils.sim_runner import ensure_single_tex_py, extract_simulation_from_tex, run_simulation_with_smart_fixing, summarize_simulation_outputs

DEFAULT_MODEL = os.environ.get("SCI_MODEL", "gpt-5")

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
    doi_rate_limit_delay: float = 0.1
    max_doi_cache_size: int = 1000
    default_model: str = DEFAULT_MODEL
    fallback_models: List[str] = None
    output_diffs: bool = False  # Optional diff output for each review/revision cycle
    
    # Test-time compute scaling parameters
    use_test_time_scaling: bool = False
    revision_candidates: int = 3
    initial_draft_candidates: int = 1
    
    # Combined approach - single API call for review/editorial/revision with diffs
    use_combined_approach: bool = True  # Always True - this is the only approach
    
    def __post_init__(self):
        if self.fallback_models is None:
            if "gpt-5" in self.default_model:
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
            print(f"⚠️ Config file error: {e}. Using defaults.")
            return cls()
    
    def to_file(self, config_path: Path) -> None:
        """Save configuration to JSON file."""
        config_path.parent.mkdir(parents=True, exist_ok=True)
        with open(config_path, 'w') as f:
            json.dump(asdict(self), f, indent=2)
        print(f"✅ Configuration saved to {config_path}")

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
    
    # Console handler (for warnings and errors only)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.WARNING)
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

def _openai_chat(messages: List[Dict[str, str]], model: str, request_timeout: Optional[int] = None, prompt_type: str = "general", fallback_models: Optional[List[str]] = None) -> str:
    """
    Enhanced chat wrapper with error classification, fallback models, and intelligent retry.
    Includes retry logic for timeout errors and configurable temperature based on prompt type.
    """
    logger.info(f"Making API call to {model} for {prompt_type}")
    print(f"🤖 Making API call to {model} for {prompt_type}...")
    
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
        result = _try_openai_model(messages, model, temp, request_timeout, prompt_type)
        logger.info(f"API call successful for {model} ({prompt_type})")
        return result
    except Exception as primary_error:
        error_type, wait_time = _classify_error(primary_error)
        logger.warning(f"Primary model {model} failed", extra={'error': str(primary_error), 'error_type': error_type})
        
        # Don't retry for certain error types
        if wait_time is None:
            print(f"❌ Non-retryable error with {model}: {primary_error}")
            raise APIError(f"Primary model {model} failed with non-retryable error: {primary_error}")
        
        # Try fallback models if available
        if fallback_models:
            print(f"⚠️ Primary model {model} failed, trying fallback models...")
            for fallback_model in fallback_models:
                try:
                    print(f"🔄 Attempting fallback model: {fallback_model}")
                    return _try_openai_model(messages, fallback_model, temp, request_timeout, prompt_type)
                except Exception as fallback_error:
                    print(f"⚠️ Fallback model {fallback_model} also failed: {fallback_error}")
                    continue
        
        # If all models failed, raise the original error
        raise APIError(f"All models failed. Primary error: {primary_error}")

def _try_openai_model(messages: List[Dict[str, str]], model: str, temp: float, request_timeout: int, prompt_type: str, max_retries: int = 3) -> str:
    """Try a specific OpenAI model with intelligent retry logic"""
    
    for attempt in range(max_retries):
        try:
            # Newer SDK
            from openai import OpenAI
            client = OpenAI()
            # Use configured temperature based on prompt type
            print(f"📡 Sending request with temperature={temp}, timeout={request_timeout}s (attempt {attempt + 1}/{max_retries})...")
            
            resp = client.chat.completions.create(
                model=model, 
                messages=messages, 
                temperature=temp, 
                timeout=request_timeout
            )
            print("✅ API call successful!")
            return resp.choices[0].message.content
            
        except KeyboardInterrupt:
            print("❌ User interrupted the process")
            raise
        except Exception as e:
            error_type, wait_time = _classify_error(e)
            print(f"⚠️ API Error (attempt {attempt + 1}): {e} (Type: {error_type})")
            
            # Don't retry for certain error types
            if wait_time is None:
                raise e
            
            # Only retry for retryable errors and if we have attempts left
            if attempt < max_retries - 1 and wait_time is not None:
                print(f"🔄 Retrying in {wait_time} seconds...")
                time.sleep(wait_time)
                continue
            else:
                raise e

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

def _save_iteration_diff(old_content: str, new_content: str, output_dir: Path, iteration: int, filename: str = "paper.tex"):
    """
    Generate and display a unified diff between old and new file content to terminal only.
    
    Args:
        old_content: Content before revision
        new_content: Content after revision  
        output_dir: Directory (unused - no files saved)
        iteration: Current iteration number
        filename: Name of the file being diffed (default: paper.tex)
    """
    try:
        # Generate unified diff
        diff_lines = list(difflib.unified_diff(
            old_content.splitlines(keepends=True),
            new_content.splitlines(keepends=True),
            fromfile=f"{filename} (before iteration {iteration})",
            tofile=f"{filename} (after iteration {iteration})",
            lineterm=""
        ))
        
        if diff_lines:
            print(f"\n{'='*80}")
            print(f"📝 DIFF FOR ITERATION {iteration} - {filename}")
            print(f"{'='*80}")
            # Print diff to terminal
            for line in diff_lines:
                print(line, end='')
            print(f"{'='*80}\n")
        else:
            print(f"📝 No changes detected in iteration {iteration}")
            
    except Exception as e:
        print(f"⚠️ Failed to generate diff for iteration {iteration}: {e}")

def _universal_chat(messages: List[Dict[str, str]], model: str, request_timeout: Optional[int] = None, prompt_type: str = "general", fallback_models: Optional[List[str]] = None) -> str:
    """
    Universal chat function that automatically detects whether to use OpenAI or Google AI
    based on the model name and routes the request accordingly.
    """
    # Detect provider based on model name
    if model.startswith(('gemini', 'models/gemini')):
        # Google AI model
        return _google_chat(messages, model, request_timeout, prompt_type, fallback_models)
    else:
        # OpenAI model
        return _openai_chat(messages, model, request_timeout, prompt_type, fallback_models)

def _google_chat(messages: List[Dict[str, str]], model: str, request_timeout: Optional[int] = None, prompt_type: str = "general", fallback_models: Optional[List[str]] = None) -> str:
    """
    Google AI chat wrapper with similar interface to OpenAI chat.
    Based on working reference implementation.
    Sets HTTPS_PROXY specifically for Gemini API calls.
    """
    if not GOOGLE_AI_AVAILABLE:
        raise APIError("Google AI SDK not available. Please install with: pip install google-generativeai")
    
    # Set proxy specifically for Google AI API (not needed for OpenAI)
    original_proxy = os.environ.get("HTTPS_PROXY")
    os.environ["HTTPS_PROXY"] = "http://127.0.0.1:7078"
    print(f"🌐 Set proxy for Gemini API: {os.environ['HTTPS_PROXY']}")
    
    try:
        # Configure API key - use hardcoded key as in reference
        api_key = "AIzaSyCXhoRyRmp_6Rpbp9eZjjwEvE11KrKIJII"
        genai.configure(api_key=api_key)
        
        logger.info(f"Making Google AI API call to {model} for {prompt_type}")
        print(f"🤖 Making Google AI API call to {model} for {prompt_type}...")
        
        # Set timeout
        if request_timeout is None:
            request_timeout = 1800  # 30 minutes default
        
        # Convert OpenAI messages to a single prompt
        # Combine all messages into one prompt for Gemini
        combined_prompt = ""
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            
            if role == "system":
                combined_prompt += f"System: {content}\n\n"
            elif role == "user":
                combined_prompt += f"User: {content}\n\n"
            elif role == "assistant":
                combined_prompt += f"Assistant: {content}\n\n"
        
        # Remove trailing newlines
        combined_prompt = combined_prompt.strip()
        
        # Create model instance using the reference approach
        genai_model = genai.GenerativeModel(model)
        
        print(f"📡 Sending Google AI request with timeout={request_timeout}s...")
        
        # Generate content directly as in the reference
        response = genai_model.generate_content(combined_prompt)
        
        print("✅ Google AI API call successful!")
        return response.text
        
    except Exception as e:
        error_msg = str(e)
        print(f"❌ Google AI API Error: {error_msg}")
        
        # Provide helpful proxy setup guidance
        if "503" in error_msg or "connect" in error_msg.lower() or "timeout" in error_msg.lower():
            print("💡 Connection issue detected. Proxy is set for Gemini API.")
            print("   Check your network connection to Google AI API")
        
        # Try fallback models if available
        if fallback_models:
            print(f"⚠️ Primary model {model} failed, trying fallback models...")
            for fallback_model in fallback_models:
                try:
                    print(f"🔄 Attempting fallback model: {fallback_model}")
                    if fallback_model.startswith(('gemini', 'models/gemini')):
                        return _google_chat(messages, fallback_model, request_timeout, prompt_type, None)
                    else:
                        # For OpenAI fallback, restore original proxy
                        if original_proxy is not None:
                            os.environ["HTTPS_PROXY"] = original_proxy
                        elif "HTTPS_PROXY" in os.environ:
                            del os.environ["HTTPS_PROXY"]
                        print("🔄 Removed proxy for OpenAI fallback")
                        return _openai_chat(messages, fallback_model, request_timeout, prompt_type, None)
                except Exception as fallback_error:
                    print(f"⚠️ Fallback model {fallback_model} also failed: {fallback_error}")
                    continue
        
        raise APIError(f"Google AI model {model} failed: {error_msg}")
    
    finally:
        # Restore original proxy setting after Google AI call
        if original_proxy is not None:
            os.environ["HTTPS_PROXY"] = original_proxy
        elif "HTTPS_PROXY" in os.environ:
            del os.environ["HTTPS_PROXY"]
        print("🔄 Restored original proxy settings")

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
    print(f"🧠 Generating {num_ideas} research ideas for '{topic}' in {field}...")
    
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
Rank the top 5 ideas by overall potential (considering originality × impact × feasibility), and explain your ranking criteria.

## RECOMMENDATION
Select the single best idea and explain why it's optimal for development into a research paper.
"""

    try:
        print("  📝 Sending ideation request to AI...")
        messages = [{"role": "user", "content": ideation_prompt}]
        
        response = _universal_chat(
            messages=messages,
            model=model,
            request_timeout=request_timeout,
            prompt_type="ideation",
            fallback_models=fallback_models or []
        )
        
        print("  ✅ Ideas generated successfully!")
        
        # Parse the response to extract structured data
        ideas = _parse_ideation_response(response)
        
        print(f"  📊 Parsed {len(ideas)} ideas from response")
        
        # Display summary
        print("\n🎯 Research Ideas Summary:")
        print("━" * 60)
        
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
        print(f"  ❌ Ideation failed: {e}")
        print("  🔄 Proceeding with original topic/question...")
        
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
        print(f"  ⚠️  Parsing error: {e}")
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
    
    print(f"🧪 Starting test-time compute scaling with candidate generation for {model}")
    print(f"📊 Testing candidate counts: {candidate_counts}")
    
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
        math_indicators = ['O(', 'θ(', 'Ω(', 'log', 'n²', 'n^2', 'exponential', 
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
        print(f"\n🔄 Testing {candidate_count} candidates...")
        
        # Prepare test messages
        test_messages = [
            {"role": "system", "content": "You are an expert computer scientist and algorithm designer. Provide detailed, technically accurate solutions with clear explanations."},
            {"role": "user", "content": test_prompt}
        ]
        
        # Generate multiple candidates
        candidates = []
        generation_times = []
        
        print(f"  🎯 Generating {candidate_count} candidate responses...")
        
        for i in range(candidate_count):
            print(f"    📝 Generating candidate {i + 1}/{candidate_count}...")
            
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
                
                print(f"      ✅ Candidate {i + 1} generated: {generation_time:.2f}s, {len(response)} chars")
                
            except Exception as e:
                print(f"      ❌ Candidate {i + 1} failed: {e}")
                candidates.append("")
                generation_times.append(None)
        
        # Evaluate all candidates and select the best
        valid_candidates = [c for c in candidates if c.strip()]
        
        if valid_candidates:
            print(f"  🏆 Selecting best candidate from {len(valid_candidates)} valid responses...")
            
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
            
            print(f"    🎯 Best candidate quality: {best_metrics['overall_quality']:.3f}")
            print(f"    📊 Average quality: {statistics.mean(all_metrics):.3f}")
            print(f"    ⬆️  Quality improvement: {best_metrics['overall_quality'] - statistics.mean(all_metrics):.3f}")
            
        else:
            results["candidate_results"][candidate_count] = {
                "error": "All candidates failed",
                "total_candidates": candidate_count,
                "successful_candidates": 0
            }
    
    # Analyze quality scaling with compute
    print(f"\n📈 Analyzing quality scaling with test-time compute...")
    
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
    print(f"\n📋 Test-Time Compute Scaling Summary for {model}")
    print(f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    
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
    
    print(f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    
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
    candidate_count: int = 3
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
    
    print(f"  🎯 Generating {candidate_count} revision candidates...")
    
    candidates = []
    generation_times = []
    
    # Generate multiple revision candidates with slight variations
    for i in range(candidate_count):
        print(f"    📝 Generating revision candidate {i + 1}/{candidate_count}...")
        
        start_time = time.time()
        
        try:
            # Create varied revision prompts to encourage diversity
            base_prompt = _revise_prompt(current_tex, sim_summary, review_text, latex_errors, project_dir, user_prompt)
            
            # Add variation instructions to encourage different approaches
            if i > 0:
                variation_instructions = [
                    "\nFocus particularly on improving the technical depth and rigor of the analysis.",
                    "\nEmphasize clarity of presentation and logical flow between sections.",
                    "\nPrioritize addressing methodological concerns and experimental validation.",
                    "\nConcentrate on strengthening the theoretical foundations and mathematical formulations.",
                    "\nFocus on improving the practical implications and real-world applications."
                ]
                
                variation = variation_instructions[(i - 1) % len(variation_instructions)]
                
                # Add variation to the system prompt
                varied_prompt = base_prompt.copy()
                varied_prompt[0]["content"] += variation
            else:
                varied_prompt = base_prompt
            
            # Generate revision candidate
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
            
            print(f"      ✅ Candidate {i + 1} generated: {generation_time:.2f}s, {len(candidate)} chars")
            
        except Exception as e:
            print(f"      ❌ Candidate {i + 1} failed: {e}")
            candidates.append("")
            generation_times.append(None)
    
    # Filter out empty candidates
    valid_candidates = [(i, c) for i, c in enumerate(candidates) if c.strip()]
    
    if not valid_candidates:
        print("    ⚠️  All revision candidates failed, using empty response")
        return ""
    
    print(f"  🏆 Selecting best candidate from {len(valid_candidates)} valid responses...")
    
    # Evaluate each candidate using quality metrics
    candidate_scores = []
    
    for idx, (orig_idx, candidate) in enumerate(valid_candidates):
        try:
            # Calculate quality metrics for this candidate
            metrics = _evaluate_revision_quality(candidate, review_text, latex_errors)
            score = metrics['overall_quality']
            candidate_scores.append((orig_idx, candidate, score, metrics))
            
            print(f"    📊 Candidate {orig_idx + 1}: Quality score {score:.3f}")
            
        except Exception as e:
            print(f"    ❌ Failed to evaluate candidate {orig_idx + 1}: {e}")
            candidate_scores.append((orig_idx, candidate, 0.0, {}))
    
    # Select the best candidate
    if candidate_scores:
        best_idx, best_candidate, best_score, best_metrics = max(candidate_scores, key=lambda x: x[2])
        avg_score = sum(score for _, _, score, _ in candidate_scores) / len(candidate_scores)
        improvement = best_score - avg_score
        
        print(f"  🎯 Selected candidate {best_idx + 1} with quality score {best_score:.3f}")
        print(f"  📈 Quality improvement over average: +{improvement:.3f}")
        print(f"  ⏱️  Total compute time: {sum(t for t in generation_times if t):.1f}s")
        
        return best_candidate
    else:
        print("    ⚠️  No valid candidates scored, returning first valid candidate")
        return valid_candidates[0][1] if valid_candidates else ""

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
    
    print(f"  🎯 Generating {candidate_count} initial draft candidates...")
    
    candidates = []
    generation_times = []
    
    # Generate multiple initial draft candidates with slight variations
    for i in range(candidate_count):
        print(f"    📝 Generating draft candidate {i + 1}/{candidate_count}...")
        
        start_time = time.time()
        
        try:
            # Create varied initial draft prompts to encourage diversity
            base_prompt = _initial_draft_prompt(topic, field, question, user_prompt)
            
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
            
            print(f"      ✅ Candidate {i + 1} generated: {generation_time:.2f}s, {len(candidate)} chars")
            
        except Exception as e:
            print(f"      ❌ Candidate {i + 1} failed: {e}")
            candidates.append("")
            generation_times.append(None)
    
    # Filter out empty candidates
    valid_candidates = [(i, c) for i, c in enumerate(candidates) if c.strip()]
    
    if not valid_candidates:
        print("    ⚠️  All draft candidates failed, using empty response")
        return ""
    
    print(f"  🏆 Selecting best candidate from {len(valid_candidates)} valid responses...")
    
    # Evaluate each candidate using quality metrics
    candidate_scores = []
    
    for idx, (orig_idx, candidate) in enumerate(valid_candidates):
        try:
            # Calculate quality metrics for this candidate
            metrics = _evaluate_initial_draft_quality(candidate, topic, field, question)
            score = metrics['overall_quality']
            candidate_scores.append((orig_idx, candidate, score, metrics))
            
            print(f"    📊 Candidate {orig_idx + 1}: Quality score {score:.3f}")
            
        except Exception as e:
            print(f"    ❌ Failed to evaluate candidate {orig_idx + 1}: {e}")
            candidate_scores.append((orig_idx, candidate, 0.0, {}))
    
    # Select the best candidate
    if candidate_scores:
        best_idx, best_candidate, best_score, best_metrics = max(candidate_scores, key=lambda x: x[2])
        avg_score = sum(score for _, _, score, _ in candidate_scores) / len(candidate_scores)
        improvement = best_score - avg_score
        
        print(f"  🎯 Selected candidate {best_idx + 1} with quality score {best_score:.3f}")
        print(f"  📈 Quality improvement over average: +{improvement:.3f}")
        print(f"  ⏱️  Total compute time: {sum(t for t in generation_times if t):.1f}s")
        
        return best_candidate
    else:
        print("    ⚠️  No valid candidates scored, returning first valid candidate")
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
        "- Tables/figures should enhance understanding within their specific subsection context\n\n"
        
        "STRUCTURE REQUIREMENTS:\n"
        "- Start with \\begin{filecontents*}{refs.bib}...\\end{filecontents*} containing ALL bibliography entries\n"
        "- Follow with standard LaTeX document structure\n"
        "- Organize sections according to paper type and field conventions\n"
        "- End with \\bibliography{refs} to use the embedded references\n\n"
        
        "FORMATTING REQUIREMENTS:\n"
        "- All content must use width=\\linewidth constraints\n"
        "- Wrap wide tables using adjustbox width=\\linewidth\n"
        "- NO CODE BLOCKS: Do not include Python code or any programming code in the paper\n"
        "- ALGORITHMS ALLOWED: Use algorithm2e or algorithmic environments for pseudocode when necessary\n"
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

def _combined_review_edit_revise_prompt(paper_tex: str, sim_summary: str, latex_errors: str = "", project_dir: Path = None, user_prompt: Optional[str] = None, iteration_count: int = 1) -> List[Dict[str, str]]:
    """Combined prompt for review, editorial decision, and revision with diff output."""
    sys_prompt = (
        "You are a combined AI system acting as: (1) Top-tier journal reviewer, (2) Handling editor, and (3) Paper author. "
        "Your task is to review the paper, make an editorial decision, and if needed, provide complete file diffs for all revisions.\n\n"
        
        "WORKFLOW STEPS:\n"
        "1. REVIEW: Conduct a thorough peer review meeting top journal standards\n"
        "2. EDITORIAL DECISION: Decide if the paper is ready for publication (YES/NO/REJECT)\n"
        "3. REVISION: If not ready, provide complete file diffs for ALL files that need changes\n\n"
        
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
        
        "EDITORIAL DECISION CRITERIA:\n"
        "- YES: Paper meets top journal standards, compiles successfully, quality threshold met\n"
        "- NO: Significant issues require revision but paper is fundamentally sound\n"
        "- REJECT: Fundamental flaws make paper unsuitable for publication\n\n"
        
        "REVISION OUTPUT FORMAT (when decision is NO):\n"
        "Provide complete file diffs in this exact format:\n\n"
        "```diff\n"
        "--- a/filename.ext\n"
        "+++ b/filename.ext\n"
        "@@ -line_start,line_count +line_start,line_count @@\n"
        " unchanged line\n"
        "-removed line\n"
        "+added line\n"
        " unchanged line\n"
        "```\n\n"
        
        "CRITICAL REVISION REQUIREMENTS:\n"
        "- Address ALL review concerns completely\n"
        "- Fix LaTeX compilation errors if any\n"
        "- Use only authentic references (no fake citations)\n"
        "- Ensure single file structure with embedded references\n"
        "- Apply proper size constraints to all visuals\n"
        "- Remove filename references from paper text\n"
        "- Use only real simulation data\n"
        "- Maintain paper structure appropriate for field\n"
        "- Include all necessary files in diffs (paper.tex, simulation.py, etc.)\n"
        "- ESSENTIAL: Fix table/figure positioning using contextual placement:\n"
        "  * Use [h] (here if possible), [ht] (here or top of section), or [H] (force here) for contextual positioning\n"
        "  * AVOID [t] and [!t] positioning which forces floats to page tops regardless of context\n"
        "  * Place tables/figures in relevant subsections after first text mention\n"
        "  * Ensure visual self-containment with descriptive captions\n"
        "  * Verify each float appears in logical context, not forced to random page positions\n\n"
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
        f"This is iteration {iteration_count}. Please complete the 3-step workflow:\n\n"
        "STEP 1: REVIEW\n"
        "Conduct a thorough peer review of the paper using top journal standards.\n\n"
        "STEP 2: EDITORIAL DECISION\n"
        "Based on your review, make an editorial decision: YES/NO/REJECT\n\n"
        "STEP 3: REVISION (if decision is NO)\n"
        "If decision is NO, provide complete file diffs for all necessary changes.\n\n"
        "----- CURRENT PAPER (LATEX) -----\n" + paper_tex + "\n"
        "----- SIMULATION CODE & OUTPUTS -----\n" + sim_summary + "\n"
        "----- ALL PROJECT FILES (FOR CONTEXT) -----\n" + project_files_content + "\n"
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
        "## EDITORIAL DECISION\n"
        "[YES/NO/REJECT with brief justification]\n\n"
        "## REVISION DIFFS (if decision is NO)\n"
        "[Complete file diffs for all changes needed]\n"
    )
    
    return [{"role": "system", "content": sys_prompt}, {"role": "user", "content": user}]

def _parse_combined_response(response: str, project_dir: Path) -> tuple[str, str, dict]:
    """
    Parse the combined review/editorial/revision response.
    
    Returns:
        (review_text, decision, file_changes)
        where file_changes is a dict {filename: new_content}
    """
    import re
    
    # Extract sections using regex
    review_match = re.search(r'## REVIEW\s*\n(.*?)(?=## EDITORIAL DECISION)', response, re.DOTALL | re.IGNORECASE)
    decision_match = re.search(r'## EDITORIAL DECISION\s*\n(.*?)(?=## REVISION DIFFS|$)', response, re.DOTALL | re.IGNORECASE)
    diffs_match = re.search(r'## REVISION DIFFS.*?\n(.*)', response, re.DOTALL | re.IGNORECASE)
    
    review_text = review_match.group(1).strip() if review_match else "No review section found"
    decision_text = decision_match.group(1).strip() if decision_match else "NO"
    diffs_text = diffs_match.group(1).strip() if diffs_match else ""
    
    # Extract decision (YES/NO/REJECT)
    decision_lines = decision_text.split('\n')
    decision = "NO"  # default
    for line in decision_lines:
        line_upper = line.strip().upper()
        if line_upper.startswith(('YES', 'NO', 'REJECT')):
            decision = line_upper.split()[0]
            break
    
    # Parse diffs and apply them to get new file contents
    file_changes = {}
    
    if decision == "NO" and diffs_text:
        # Parse diff format and extract file changes
        diff_blocks = re.findall(r'```diff\s*\n(.*?)\n```', diffs_text, re.DOTALL)
        
        for diff_block in diff_blocks:
            lines = diff_block.split('\n')
            current_file = None
            
            for line in lines:
                # Extract filename from diff header
                if line.startswith('--- a/') or line.startswith('+++ b/'):
                    filename = line.split('/', 1)[1] if '/' in line else line.split(' ', 1)[1]
                    if line.startswith('+++ b/'):
                        current_file = filename
                        continue
                
                # Process diff content
                if current_file and not line.startswith(('---', '+++', '@@')):
                    if current_file not in file_changes:
                        # Start with original file content if it exists
                        original_path = project_dir / current_file
                        if original_path.exists():
                            try:
                                with open(original_path, 'r', encoding='utf-8') as f:
                                    file_changes[current_file] = f.read().split('\n')
                            except:
                                file_changes[current_file] = []
                        else:
                            file_changes[current_file] = []
        
        # Alternative: Extract complete file contents if diff parsing fails
        if not file_changes and diffs_text:
            # Look for complete file contents in code blocks
            file_blocks = re.findall(r'```(?:python|tex|txt|py)?\s*\n# ?(?:File: ?)?([^\n]+)\n(.*?)\n```', diffs_text, re.DOTALL)
            for filename, content in file_blocks:
                filename = filename.strip()
                file_changes[filename] = content
            
            # Also look for explicit file sections
            file_sections = re.findall(r'(?:File: ?|Filename: ?)([^\n]+)\n```[^\n]*\n(.*?)\n```', diffs_text, re.DOTALL)
            for filename, content in file_sections:
                filename = filename.strip()
                file_changes[filename] = content
    
    return review_text, decision, file_changes

def _apply_file_changes(file_changes: dict, project_dir: Path) -> None:
    """Apply file changes from the revision response."""
    for filename, content in file_changes.items():
        file_path = project_dir / filename
        try:
            # Ensure directory exists
            file_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Write the new content
            if isinstance(content, list):
                content = '\n'.join(content)
            
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            
            print(f"✅ Updated {filename}")
            
        except Exception as e:
            print(f"❌ Failed to update {filename}: {e}")

def _review_prompt(paper_tex: str, sim_summary: str, project_dir: Path = None, user_prompt: Optional[str] = None) -> List[Dict[str, str]]:
    sys_prompt = (
        "Act as a top-tier journal reviewer (Nature, Science, Cell level) with expertise in LaTeX formatting and scientific programming. "
        "Your review must meet the highest academic standards. Be constructive but demanding. "
        "CRITICAL: If the simulation ran successfully and produced actual results, the paper MUST use these real numbers, not fake/placeholder values. "
        
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
        "11. SINGLE CODE FILE: ALL computational code must be consolidated into ONE simulation.py file only - no additional .py files or scripts.\n\n"
        
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
        "- NO CODE BLOCKS: Do not include Python code or any programming code in the paper\n"
        "- ALGORITHMS ALLOWED: Use algorithm2e or algorithmic environments for pseudocode when necessary\n"
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
        
        "Provide specific, actionable feedback with concrete suggestions for improvement. "
        "If the paper violates any of the 10 mandatory requirements, mark it as needing major revision. "
        "Pay special attention to reference authenticity, results documentation, figure generation, filename removal, and structural appropriateness."
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
            "The above user instruction should guide your editorial decision. "
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

def _revise_prompt(paper_tex: str, sim_summary: str, review_text: str, latex_errors: str = "", project_dir: Path = None, user_prompt: Optional[str] = None) -> List[Dict[str, str]]:
    sys_prompt = (
        "You are the paper author making revisions based on peer review. Your goal is to address ALL reviewer concerns "
        "while maintaining scientific integrity and clarity. Produce a COMPLETE revised LaTeX file.\n\n"
        
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
        "- Review all existing \\begin{table} and \\begin{figure} environments to ensure proper positioning\n\n"
        
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
        "11. Improve clarity and presentation quality\n"
        "12. Ensure reproducibility and code quality\n\n"
        
        "FORMATTING REQUIREMENTS (CRITICAL - NO EXCEPTIONS):\n"
        "- ALL content: use width=\\linewidth constraints (never exceed page width)\n"
        "- ALL wide tables: \\begin{adjustbox}{width=\\linewidth}...\\end{adjustbox}\n"
        "- ALL TikZ diagrams: wrap in \\begin{adjustbox}{width=\\linewidth}...\\end{adjustbox}\n"
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
    quality_threshold: float = 0.8,  # New parameter
    check_references: bool = True,    # New parameter
    validate_figures: bool = True,    # New parameter
    user_prompt: Optional[str] = None,  # New parameter for custom user prompt
    config: Optional[WorkflowConfig] = None,  # Configuration parameter
    enable_ideation: bool = True,     # Enable ideation phase for new papers
    num_ideas: int = 15,             # Number of ideas to generate
    output_diffs: bool = False       # Optional diff output for each review/revision cycle
) -> Path:
    """Enhanced workflow with quality validation, progress tracking, and custom user prompts."""
    
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
    
    logger.info(f"Starting workflow with config: quality_threshold={config.quality_threshold}, max_iterations={config.max_iterations}")
    
    project_dir = _prepare_project_dir(output_dir, modify_existing)
    paper_path, sim_path = ensure_single_tex_py(project_dir, strict=strict_singletons)

    # Progress tracking variables
    quality_history = []
    stagnation_count = 0
    best_quality_score = 0.0
    
    # Get custom user prompt if not provided as parameter
    if user_prompt is None:
        print("\n" + "="*60)
        print("🎯 CUSTOM PROMPT INPUT")
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
        
        user_prompt = input("Enter your custom prompt (or press Enter to skip): ").strip()
        if not user_prompt:
            user_prompt = None
        else:
            print(f"\n✅ Custom prompt set: {user_prompt[:100]}{'...' if len(user_prompt) > 100 else ''}")
    
    # Store user prompt for use in all AI interactions
    if user_prompt:
        print(f"\n🎯 Using custom user prompt throughout workflow")
    
    print(f"\n📁 Project directory: {project_dir}")
    print(f"📄 Paper file: {paper_path}")
    print(f"🐍 Simulation file: {sim_path}")

    # Check for separate refs.bib files and warn user
    refs_bib_files = list(project_dir.glob("*.bib"))
    if refs_bib_files:
        print(f"⚠️ WARNING: Found separate bibliography files: {[f.name for f in refs_bib_files]}")
        print("   All references must be embedded in paper.tex using filecontents or thebibliography")
        print("   Separate .bib files will be ignored during compilation!")

    # Check if this is actually a minimal template (new paper) or has real content
    paper_content = paper_path.read_text(encoding="utf-8").strip()
    is_minimal_template = (paper_content == "\\documentclass{article}\\begin{document}\\end{document}" or len(paper_content) < 200)
    
    # If no real paper content yet (fresh), run ideation and then draft one.
    if is_minimal_template:
        if enable_ideation:
            print("🧠 Starting Ideation Phase for new paper...")
            
            # Generate and analyze multiple research ideas
            ideation_result = _generate_research_ideas(
                topic=topic,
                field=field,
                question=question,
                model=model,
                num_ideas=num_ideas,
                request_timeout=request_timeout,
                fallback_models=config.fallback_models
            )
            
            # Use the best idea to refine the topic and question
            selected_idea = ideation_result.get("selected_idea")
            if selected_idea:
                refined_topic = selected_idea.get("title", topic)
                refined_question = selected_idea.get("core_concept", question)
                
                print(f"\n⭐ Selected Research Idea:")
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
                
                print(f"💾 Ideation analysis saved to: {ideation_file}")
                
                # Use refined topic and question for paper generation
                final_topic = refined_topic
                final_question = refined_question
            else:
                print("⚠️  No ideas selected, using original topic/question")
                final_topic = topic
                final_question = question
        else:
            print("ℹ️  Ideation phase skipped (--skip-ideation flag used)")
            final_topic = topic
            final_question = question
        
        print(f"\n📝 Creating paper draft for: {final_topic}")
        
        # Use test-time compute scaling for initial draft if enabled
        if hasattr(config, 'use_test_time_scaling') and config.use_test_time_scaling and hasattr(config, 'initial_draft_candidates') and config.initial_draft_candidates > 1:
            print(f"🧪 Using test-time compute scaling with {config.initial_draft_candidates} draft candidates...")
            draft = _generate_best_initial_draft_candidate(
                final_topic, field, final_question, user_prompt, model, request_timeout, config, config.initial_draft_candidates
            )
        else:
            draft = _universal_chat(_initial_draft_prompt(final_topic, field, final_question, user_prompt), model=model, request_timeout=request_timeout, prompt_type="initial_draft", fallback_models=config.fallback_models)
        paper_path.write_text(draft, encoding="utf-8")
        
        if enable_ideation:
            print("✅ Initial draft created with ideation-selected concept")
        else:
            print("✅ Initial draft created with original concept")
    else:
        print(f"Using existing paper content ({len(paper_content)} characters)")
        print("ℹ️  Skipping ideation phase for existing paper")

    # Extract/refresh simulation.py from LaTeX initially
    extract_simulation_from_tex(paper_path, sim_path)

    # Review-Revise loop with quality tracking
    for i in range(1, config.max_iterations + 1):
        print(f"Starting iteration {i} of {max_iterations}")
        
        # ALWAYS run simulation before each review to get current results
        print(f"Running simulation before review {i}...")
        
        # Enhanced simulation extraction with validation
        extract_success, extract_message = _extract_simulation_code_with_validation(paper_path, sim_path)
        if not extract_success:
            print(f"⚠️ Simulation extraction issues: {extract_message}")
        
        simulation_fixer = _create_simulation_fixer(model, request_timeout)
        sim_out = run_simulation_with_smart_fixing(
            sim_path, 
            python_exec=python_exec, 
            cwd=project_dir,
            llm_fixer=simulation_fixer,
            max_fix_attempts=2
        )
        # Include both simulation code and outputs for LLM review
        simulation_code = sim_path.read_text(encoding="utf-8", errors="ignore")
        sim_summary = summarize_simulation_outputs(sim_out, simulation_code)
        
        # COMPILE LATEX WITH DYNAMIC TIMEOUT
        print(f"Compiling LaTeX file with pdflatex...")
        current_tex = paper_path.read_text(encoding="utf-8", errors="ignore")
        dynamic_timeout = _calculate_dynamic_timeout(current_tex, config)
        latex_success, latex_errors = _compile_latex_and_get_errors(paper_path, timeout=dynamic_timeout)
        
        if not latex_success:
            print(f"⚠️ LaTeX compilation failed. Errors will be sent to LLM for fixing.")
            print(f"Error log (last 20 lines):\n{latex_errors}")
        else:
            print(f"✅ LaTeX compilation successful!")
        
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
        
        # CALCULATE QUALITY METRICS AND TRACK PROGRESS
        current_metrics = _extract_quality_metrics(current_tex, sim_summary)
        quality_score = _calculate_quality_score(current_metrics, quality_issues)
        quality_history.append(quality_score)
        
        # Resource management - limit quality history size using config
        if len(quality_history) > config.max_quality_history_size:
            keep_size = config.max_quality_history_size // 2
            quality_history = quality_history[-keep_size:]  # Keep recent entries
            logger.info(f"Quality history trimmed to {keep_size} entries")
            print("🔧 Quality history trimmed for memory management")
        
        logger.info(f"Iteration {i} quality score: {quality_score:.2f}")
        print(f"📊 Iteration {i} quality score: {quality_score:.2f}")
        
        # Check for improvement
        if quality_score > best_quality_score:
            best_quality_score = quality_score
            stagnation_count = 0
        else:
            stagnation_count += 1
        
        # Early stopping for stagnation
        if stagnation_count >= 2 and i > 1:
            print(f"⚠️ Quality stagnation detected ({stagnation_count} iterations without improvement)")
        
        # COMBINED REVIEW, EDITORIAL DECISION, AND REVISION IN ONE CALL
        print(f"🔄 Running combined review/editorial/revision process...")
        combined_response = _universal_chat(
            _combined_review_edit_revise_prompt(current_tex, sim_summary, latex_errors, project_dir, user_prompt, i), 
            model=model, request_timeout=request_timeout, prompt_type="combined_review_edit_revise", fallback_models=config.fallback_models
        )
        
        # Parse the combined response
        review, decision, file_changes = _parse_combined_response(combined_response, project_dir)
        
        print(f"📝 Review completed")
        print(f"📋 Editorial decision: {decision}")
        
        # Store original content for diff generation if enabled
        original_content = current_tex if output_diffs else None
        
        # Apply file changes if any were provided
        if file_changes:
            print(f"📁 Applying file changes to {len(file_changes)} files...")
            _apply_file_changes(file_changes, project_dir)
            print(f"✅ File changes applied successfully")
            
            # Generate and save diff if enabled
            if output_diffs and original_content:
                new_content = paper_path.read_text(encoding="utf-8", errors="ignore")
                _save_iteration_diff(original_content, new_content, project_dir, i, "paper.tex")
                
        elif decision.strip().upper().startswith("NO"):
            print(f"⚠️ Decision was NO but no file changes were provided. Using fallback revision...")
            # Fallback to traditional revision if no diffs were provided
            revised = _universal_chat(_revise_prompt(current_tex, sim_summary, review, latex_errors, project_dir, user_prompt), model=model, request_timeout=request_timeout, prompt_type="revise", fallback_models=config.fallback_models)
            paper_path.write_text(revised, encoding="utf-8")
            
            # Generate and save diff if enabled (for fallback revision)
            if output_diffs and original_content:
                _save_iteration_diff(original_content, revised, project_dir, i, "paper.tex")
        
        # ENHANCED DECISION LOGIC WITH QUALITY THRESHOLD
        decision_upper = decision.strip().upper()
        meets_quality_threshold = quality_score >= config.quality_threshold
        
        if decision_upper.startswith("YES") and latex_success and meets_quality_threshold:
            print(f"[OK] Editor accepted at iteration {i}, LaTeX compiles, and quality threshold met (score: {quality_score:.2f})")
            final_metrics = _extract_quality_metrics(current_tex, sim_summary)
            print(f"Final paper metrics: {final_metrics}")
            break
        elif decision_upper.startswith("YES") and latex_success and not meets_quality_threshold:
            print(f"[CONDITIONAL] Editor accepted but quality below threshold ({quality_score:.2f} < {config.quality_threshold}). Continuing revisions...")
        elif decision_upper.startswith("REJECT") and stagnation_count >= 2:
            print(f"[STOP] Editor rejected and quality stagnating. Ending revisions.")
            break
        elif decision_upper.startswith("REJECT"):
            print(f"[REJECT] Editor rejected the paper at iteration {i}.")
            print("Paper has fundamental issues but continuing with revisions to improve it...")
        
        print(f"Iteration {i}: Combined review/editorial/revision completed")
    
    # Final quality report
    print(f"\n📈 Quality progression: {[f'{q:.2f}' for q in quality_history]}")
    print(f"🏆 Best quality score achieved: {best_quality_score:.2f}")

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
    
    p.add_argument("--output-dir", default="output", help="Output directory root (contains project subfolder)")
    p.add_argument("--model", default=DEFAULT_MODEL, help="OpenAI model to use (default: gpt-5)")
    p.add_argument("--request-timeout", type=int, default=3600, help="Per-request timeout seconds (0 means no timeout)")
    p.add_argument("--max-retries", type=int, default=3, help="Max OpenAI retries")
    p.add_argument("--max-iterations", type=int, default=4, help="Max review->revise iterations")
    p.add_argument("--modify-existing", action="store_true", help="If output dir already has paper.tex, modify in place")
    p.add_argument("--strict-singletons", action="store_true", default=True, help="Keep only paper.tex & simulation.py (others archived)")
    p.add_argument("--python-exec", default=None, help="Python interpreter for running simulation.py")
    
    # Configuration file support
    p.add_argument("--config", type=str, default=None, help="Path to configuration JSON file")
    p.add_argument("--save-config", type=str, default=None, help="Save current configuration to JSON file")
    
    # New quality control parameters
    p.add_argument("--quality-threshold", type=float, default=0.8, help="Minimum quality score required for acceptance (0.0-1.0)")
    p.add_argument("--check-references", action="store_true", default=True, help="Enable external reference validation")
    p.add_argument("--validate-figures", action="store_true", default=True, help="Enable figure generation validation")
    p.add_argument("--skip-reference-check", action="store_true", help="Disable external reference validation (faster)")
    p.add_argument("--skip-figure-validation", action="store_true", help="Disable figure generation validation (faster)")
    
    # Ideation parameters
    p.add_argument("--enable-ideation", action="store_true", default=True, help="Enable research ideation phase for new papers")
    p.add_argument("--skip-ideation", action="store_true", help="Skip research ideation phase (use original topic directly)")
    p.add_argument("--num-ideas", type=int, default=15, help="Number of research ideas to generate (10-20)")
    
    # Custom prompt parameter
    p.add_argument("--user-prompt", type=str, default=None, help="Custom prompt that takes priority over standard requirements")
    
    # Diff output parameter
    p.add_argument("--output-diffs", action="store_true", help="Save diff files for each review/revision cycle to track changes")
    
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
    
    # Initialize topic, field, question attributes if they don't exist (modify-existing mode)
    if modify_existing:
        if not hasattr(args, 'topic'):
            args.topic = None
        if not hasattr(args, 'field'):
            args.field = None
        if not hasattr(args, 'question'):
            args.question = None
    
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
            args.topic = input("Topic: ").strip()
        if not getattr(args, 'field', None):
            args.field = input("Field: ").strip()
        if not getattr(args, 'question', None):
            args.question = input("Research question: ").strip()
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
    rate_limit_delay = config.doi_rate_limit_delay if config else 0.1
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
    base_timeout = config.latex_timeout_base if config else 120  # Use config or default
    
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
    print(f"📊 Dynamic timeout calculated: {total_timeout}s (TikZ:{tikz_count}, Tables:{table_count}, Figures:{figure_count}, Pages:{page_count})")
    
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
    print("🚀 Starting Enhanced SciResearch Workflow...")
    try:
        ns = parse_args()
        
        # Load configuration
        config = None
        if ns.config:
            config = WorkflowConfig.from_file(Path(ns.config))
            print(f"📄 Configuration loaded from: {ns.config}")
        else:
            config = WorkflowConfig()
            print("📄 Using default configuration")
        
        # Save configuration if requested
        if ns.save_config:
            config.to_file(Path(ns.save_config))
            print(f"✅ Configuration saved to: {ns.save_config}")
            sys.exit(0)
        
        print(f"📁 Working with: {ns.output_dir}")
        print(f"🤖 Using model: {ns.model}")
        print(f"🔄 Max iterations: {ns.max_iterations}")
        print(f"📊 Quality threshold: {ns.quality_threshold}")
        print(f"🔍 Reference validation: {'enabled' if ns.check_references else 'disabled'}")
        print(f"🖼️ Figure validation: {'enabled' if ns.validate_figures else 'disabled'}")
        print(f"🧠 Research ideation: {'enabled' if ns.enable_ideation else 'disabled'}")
        
        # Handle test scaling mode
        if ns.test_scaling:
            print("\n🔬 Starting Test-Time Compute Scaling Analysis...")
            # Convert comma-separated string to list of integers
            candidate_counts_list = [int(x.strip()) for x in ns.scaling_candidates.split(',')]
            result = test_time_compute_scaling(
                model=ns.model,
                test_prompt=ns.scaling_prompt,
                candidate_counts=candidate_counts_list,
                timeout_base=ns.scaling_timeout
            )
            if result:
                print("\n✅ Test-Time Compute Scaling Analysis Complete!")
            else:
                print("\n❌ Test-Time Compute Scaling Analysis Failed!")
            sys.exit(0)
        
        # Apply test-time compute scaling configuration
        if hasattr(ns, 'use_test_time_scaling') and ns.use_test_time_scaling:
            config.use_test_time_scaling = True
            config.revision_candidates = ns.revision_candidates
            config.initial_draft_candidates = ns.draft_candidates
            print(f"🧪 Test-time compute scaling enabled: {config.revision_candidates} revision candidates")
        
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
            num_ideas=ns.num_ideas,
            output_diffs=ns.output_diffs
        )
        print(f"✅ Workflow completed! Results in: {result_dir}")
    except Exception as e:
        print(f"❌ Workflow failed: {e}")
        import traceback
        traceback.print_exc()
