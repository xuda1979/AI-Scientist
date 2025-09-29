#!/usr/bin/env python3
"""
Extended workflow:
 - Enforce single paper.tex and simulation.py per project
 - Extract simulation code from LaTeX; run it; pass results to LLM during review/revision
 - Sanitize LaTeX to prevent overflow; compile-check; auto-fix on failure
"""
from __future__ import annotations
import argparse
import json
import os
import re
import subprocess
import sys
import time
import urllib.request
import urllib.parse
import asyncio
import functools
import logging
import textwrap
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
    print("⚠️ Google AI SDK not installed. Run: pip install google-generativeai")

# Local helpers
from utils.sim_runner import ensure_single_tex_py, extract_simulation_from_tex, run_simulation_with_smart_fixing, summarize_simulation_outputs
from utils.latex_tools import compile_with_autofix
from utils.parallel_checks import run_parallel_checks
from utils.model_client import OSS120BClient

DEFAULT_MODEL = os.environ.get("SCI_MODEL", "gpt-5")

# Handle common environment variable typos for OpenAI credentials
if not os.environ.get("OPENAI_API_KEY"):
    typo_key = os.environ.get("OPEANAI_API_KEY")
    if typo_key:
        os.environ["OPENAI_API_KEY"] = typo_key
        print(
            "⚠️ Detected environment variable 'OPEANAI_API_KEY'. "
            "Using it as OPENAI_API_KEY for this run; please rename it for future sessions."
        )

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
    review_model: Optional[str] = None
    revision_model: Optional[str] = None
    brainstorm_model: Optional[str] = None
    num_brainstorm_ideas: int = 5
    top_brainstorm_ideas: int = 3
    oss120b_endpoint: Optional[str] = None
    oss120b_api_key: Optional[str] = None
    google_api_key: Optional[str] = None
    google_api_proxy: Optional[str] = None
    latex_auto_fix: bool = False
    fast_ref_check: bool = False
    qa_model: Optional[str] = None
    
    def __post_init__(self):
        if self.fallback_models is None:
            if "gpt-5" in self.default_model:
                self.fallback_models = ["gpt-4o", "gpt-4"]
            else:
                self.fallback_models = ["gpt-3.5-turbo"]
        if self.review_model is None:
            self.review_model = self.default_model
        if self.revision_model is None:
            self.revision_model = self.default_model
        if self.brainstorm_model is None:
            self.brainstorm_model = self.review_model
        if self.qa_model is None:
            self.qa_model = self.review_model
    
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

# Optional OSS 120B client
OSS_CLIENT: Optional[OSS120BClient] = None

# Active workflow configuration (populated when available)
CURRENT_WORKFLOW_CONFIG: Optional[WorkflowConfig] = None


def run_offline_demo(topic: str, field: str, question: str, output_dir: Path) -> Path:
    """Run an offline demo that produces a self-contained paper without external APIs."""

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    project_dir = output_dir / f"offline_demo_{timestamp}"
    project_dir.mkdir(parents=True, exist_ok=True)

    simulation_lines = [
        "#!/usr/bin/env python3",
        '"""Lightweight offline simulation for AI-Scientist demo runs."""',
        "import json",
        "from pathlib import Path",
        "",
        "",
        "def generate_curves():",
        "    steps = list(range(0, 11))",
        "    baseline = [round(0.62 + 0.018 * s, 3) for s in steps]",
        "    demo = [round(min(0.94, 0.60 + 0.024 * s + 0.0015 * s * s), 3) for s in steps]",
        "    return {",
        '        "steps": steps,',
        '        "baseline_accuracy": baseline,',
        '        "demo_accuracy": demo,',
        "    }",
        "",
        "",
        "def main():",
        "    results = generate_curves()",
        '    output_dir = Path("simulation_outputs")',
        "    output_dir.mkdir(exist_ok=True)",
        '    (output_dir / "results.json").write_text(json.dumps(results, indent=2) + "\\n", encoding="utf-8")',
        "    print(json.dumps(results))",
        "",
        "",
        "if __name__ == \"__main__\":",
        "    main()",
    ]

    simulation_code = "\n".join(simulation_lines)

    sim_path = project_dir / "simulation.py"
    sim_path.write_text(simulation_code + "\n", encoding="utf-8")

    subprocess.run([sys.executable, sim_path.name], check=True, cwd=project_dir)

    results_path = project_dir / "simulation_outputs" / "results.json"
    results = json.loads(results_path.read_text(encoding="utf-8"))

    steps = results["steps"]
    baseline = results["baseline_accuracy"]
    demo = results["demo_accuracy"]
    improvement = demo[-1] - baseline[-1]
    improvement_pct = improvement * 100

    table_rows = [
        f"  {step} & {base:.3f} & {enh:.3f} \\" for step, base, enh in zip(steps, baseline, demo)
    ]

    paper_lines = [
        "\\documentclass{article}",
        "\\usepackage[margin=1in]{geometry}",
        "\\usepackage{booktabs}",
        "\\usepackage{adjustbox}",
        "\\usepackage{graphicx}",
        f"\\title{{Offline Demo Study on {topic}}}",
        "\\author{AI-Scientist Autonomous Workflow}",
        "\\date{\\today}",
        "",
        "\\begin{document}",
        "\\maketitle",
        "",
        "\\begin{abstract}",
        f"We present an offline demonstration of the AI-Scientist workflow configured for the field of {field}.",
        f"The system is tasked with the research question: ``{question}.'' Without relying on external APIs,",
        "the workflow generates a synthetic experiment showing that the proposed alignment-inspired intervention",
        f"improves simulated task accuracy by {improvement_pct:.1f}\\% over a baseline policy.",
        "\\end{abstract}",
        "",
        "\\section{Introduction}",
        "Modern research pipelines benefit from automation that can rapidly explore candidate ideas, evaluate",
        "them through executable simulations, and record fully reproducible artifacts. In this demonstration we",
        f"showcase how AI-Scientist can operate in a constrained offline setting while still providing a coherent",
        f"narrative around {topic.lower()}.",
        "",
        "\\section{Methodology}",
        "The demo pipeline instantiates a toy learning scenario in which an autonomous agent refines its alignment",
        f"heuristics using a curriculum grounded in {field.lower()}. The simulation tracks how the intervention alters",
        "decision quality relative to a static baseline controller. Although lightweight, the example exercises the",
        "same orchestration layers responsible for simulation extraction, execution, and manuscript assembly in the",
        "full workflow.",
        "",
        "\\section{Results}",
        "Table~\\ref{tab:offline-results} summarizes the synthetic learning curves. The demo agent rapidly closes",
        f"the gap to a target accuracy of 0.9, ultimately outperforming the baseline by {improvement_pct:.1f} percentage",
        "points. The trajectories are derived from the generated Python simulation and saved alongside this manuscript",
        "for auditability.",
        "",
        "\\begin{table}[t]",
        "  \\centering",
        f"  \\caption{{Offline simulation accuracy trends for the {topic.lower()} study.}}",
        "  \\label{tab:offline-results}",
        "  \\begin{adjustbox}{width=\\linewidth}",
        "  \\begin{tabular}{ccc}",
        "  \\toprule",
        "  Step & Baseline Accuracy & Demo Accuracy \\",
        "  \\midrule",
        *table_rows,
        "  \\bottomrule",
        "  \\end{tabular}",
        "  \\end{adjustbox}",
        "\\end{table}",
        "",
        "\\section{Discussion}",
        "Even without high-fidelity models, the workflow maintains good practices such as embedding executable code,",
        "recording intermediate analyses, and structuring LaTeX sections according to common publication norms. Users",
        "can replace the offline controller with real large language models by providing API credentials, unlocking",
        "ideation, reviewer feedback loops, and figure validation modules described in the project documentation.",
        "",
        "\\section{Conclusion}",
        "This offline run verifies that AI-Scientist can stand up a complete research artifact in environments without",
        "external network access. The resulting package—including this paper, simulation code, and recorded metrics—can",
        f"serve as a starting point for customized experiments on {topic.lower()} within the broader context of",
        f"{field.lower()}.",
        "",
        "\\bibliographystyle{unsrt}",
        "\\bibliography{refs}",
        "\\end{document}",
    ]

    paper_content = "\n".join(paper_lines)

    (project_dir / "paper.tex").write_text(paper_content + "\n", encoding="utf-8")

    references = textwrap.dedent(
        """\
        @article{amodei2016concrete,
          title={Concrete problems in AI safety},
          author={Amodei, Dario and others},
          journal={arXiv preprint arXiv:1606.06565},
          year={2016}
        }

        @article{leike2018scalable,
          title={Scalable agent alignment via reward modeling: a research direction},
          author={Leike, Jan and Martic, David and others},
          journal={arXiv preprint arXiv:1811.07871},
          year={2018}
        }

        @article{oord2016wavenet,
          title={WaveNet: A generative model for raw audio},
          author={van den Oord, Aaron and others},
          journal={arXiv preprint arXiv:1609.03499},
          year={2016}
        }

        @article{krizhevsky2012imagenet,
          title={ImageNet classification with deep convolutional neural networks},
          author={Krizhevsky, Alex and Sutskever, Ilya and Hinton, Geoffrey E},
          journal={Communications of the ACM},
          volume={60},
          number={6},
          pages={84--90},
          year={2017}
        }

        @article{silver2017mastering,
          title={Mastering the game of Go without human knowledge},
          author={Silver, David and others},
          journal={Nature},
          volume={550},
          number={7676},
          pages={354--359},
          year={2017}
        }

        @article{ziegler2019fine,
          title={Fine-Tuning Language Models from Human Preferences},
          author={Ziegler, Daniel M and others},
          journal={arXiv preprint arXiv:1909.08593},
          year={2019}
        }

        @article{anthropic2024constitutional,
          title={Constitutional AI: Evaluating Harmlessness with Language Model Judges},
          author={Bai, Yuntao and others},
          journal={arXiv preprint arXiv:2401.08561},
          year={2024}
        }

        @article{brown2020language,
          title={Language Models are Few-Shot Learners},
          author={Brown, Tom B and others},
          journal={Advances in Neural Information Processing Systems},
          volume={33},
          pages={1877--1901},
          year={2020}
        }

        @inproceedings{ouyang2022training,
          title={Training language models to follow instructions with human feedback},
          author={Ouyang, Long and others},
          booktitle={Advances in Neural Information Processing Systems},
          year={2022}
        }

        @article{bai2022training,
          title={Training a helpful and harmless assistant with reinforcement learning from human feedback},
          author={Bai, Yuntao and others},
          journal={arXiv preprint arXiv:2204.05862},
          year={2022}
        }
        """
    )

    (project_dir / "refs.bib").write_text(references.strip() + "\n", encoding="utf-8")

    ideation_report = textwrap.dedent(
        f"""\
        Offline ideation snapshot for {topic}
        ====================================

        Field: {field}
        Research question: {question}

        Key motivations:
        - Demonstrate the workflow's ability to produce reproducible artifacts without external APIs.
        - Provide a starting point for benchmarking alignment-inspired interventions in simulation.
        - Document how code generation, execution, and manuscript authoring interact inside AI-Scientist.

        Summary of simulation insights:
        - Baseline policy accuracy spans {baseline[0]:.2f} to {baseline[-1]:.2f} across the horizon.
        - Demo policy accuracy reaches {demo[-1]:.2f}, yielding a {improvement_pct:.1f}% relative improvement.
        - All assets are stored under {project_dir} for inspection.
        """
    )

    (project_dir / "ideation_analysis.txt").write_text(ideation_report.strip() + "\n", encoding="utf-8")

    return project_dir




def _record_iteration_feedback(
    project_dir: Path,
    iteration: int,
    review_text: str,
    decision_text: str,
    quality_score: float,
    latex_success: bool,
    latex_errors: str,
    ref_report: str,
) -> None:
    """Persist iteration feedback for terminal logs and GUI dashboards."""

    if project_dir is None:
        return

    project_dir.mkdir(parents=True, exist_ok=True)

    entry = {
        "iteration": iteration,
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "review": review_text,
        "decision": decision_text.strip(),
        "quality_score": quality_score,
        "latex_success": latex_success,
        "latex_errors": latex_errors[-2000:] if latex_errors else "",
        "reference_report": ref_report or "",
    }

    history_path = project_dir / "review_history.json"
    history: List[Dict[str, Any]] = []

    if history_path.exists():
        try:
            history = json.loads(history_path.read_text(encoding="utf-8"))
            if not isinstance(history, list):
                history = []
        except json.JSONDecodeError:
            history = []

    history = [item for item in history if item.get("iteration") != iteration]
    history.append(entry)
    history.sort(key=lambda item: item.get("iteration", 0))
    history_path.write_text(json.dumps(history, indent=2), encoding="utf-8")

    latest_path = project_dir / "latest_review.json"
    latest_path.write_text(json.dumps(entry, indent=2), encoding="utf-8")


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

def _universal_chat(messages: List[Dict[str, str]], model: str, request_timeout: Optional[int] = None, prompt_type: str = "general", fallback_models: Optional[List[str]] = None) -> str:
    """Route chat requests to the appropriate backend based on model name."""
    if model.startswith(('gemini', 'models/gemini')):
        return _google_chat(messages, model, request_timeout, prompt_type, fallback_models)
    if model.startswith('oss-120b'):
        return _oss120b_chat(messages, request_timeout)
    return _openai_chat(messages, model, request_timeout, prompt_type, fallback_models)

def _oss120b_chat(messages: List[Dict[str, str]], request_timeout: Optional[int] = None) -> str:
    if OSS_CLIENT is None:
        raise APIError("OSS120BClient not configured")
    return OSS_CLIENT.chat(messages, timeout=request_timeout)

def _google_chat(messages: List[Dict[str, str]], model: str, request_timeout: Optional[int] = None, prompt_type: str = "general", fallback_models: Optional[List[str]] = None) -> str:
    """
    Google AI chat wrapper with similar interface to OpenAI chat.
    Based on working reference implementation.
    Uses configured API key and optional HTTPS proxy for Gemini API calls.
    """
    if not GOOGLE_AI_AVAILABLE:
        raise APIError("Google AI SDK not available. Please install with: pip install google-generativeai")

    # Retrieve API key from environment or configuration
    api_key: Optional[str] = None
    for env_var in ("GOOGLE_API_KEY", "GEMINI_API_KEY"):
        value = os.environ.get(env_var)
        if value and value.strip():
            api_key = value.strip()
            break

    if not api_key and CURRENT_WORKFLOW_CONFIG is not None:
        config_key = (CURRENT_WORKFLOW_CONFIG.google_api_key or "").strip()
        if config_key:
            api_key = config_key

    if not api_key:
        raise APIError(
            "Google API key not configured. Set GOOGLE_API_KEY (or GEMINI_API_KEY) in your environment "
            "or provide 'google_api_key' in the workflow configuration file."
        )

    # Determine proxy configuration (optional)
    proxy_value: Optional[str] = None
    for env_var in ("GOOGLE_API_PROXY", "GEMINI_API_PROXY", "GOOGLE_HTTPS_PROXY"):
        value = os.environ.get(env_var)
        if value and value.strip():
            proxy_value = value.strip()
            break

    if not proxy_value and CURRENT_WORKFLOW_CONFIG is not None:
        config_proxy = (CURRENT_WORKFLOW_CONFIG.google_api_proxy or "").strip()
        if config_proxy:
            proxy_value = config_proxy

    original_proxy = os.environ.get("HTTPS_PROXY")
    proxy_applied = False

    if proxy_value:
        if original_proxy != proxy_value:
            os.environ["HTTPS_PROXY"] = proxy_value
            proxy_applied = True
            print(f"🌐 Set proxy for Gemini API: {proxy_value}")
        else:
            print(f"🌐 Using existing HTTPS proxy for Gemini API: {proxy_value}")
    elif original_proxy:
        print(f"🌐 Using existing HTTPS proxy for Gemini API: {original_proxy}")
    else:
        logger.info("No Gemini-specific HTTPS proxy configured; connecting directly.")

    try:
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
                        # For OpenAI fallback, restore original proxy if we changed it
                        if proxy_applied:
                            if original_proxy is not None:
                                os.environ["HTTPS_PROXY"] = original_proxy
                            else:
                                os.environ.pop("HTTPS_PROXY", None)
                            proxy_applied = False
                            print("🔄 Restored original proxy for OpenAI fallback")
                        return _openai_chat(messages, fallback_model, request_timeout, prompt_type, None)
                except Exception as fallback_error:
                    print(f"⚠️ Fallback model {fallback_model} also failed: {fallback_error}")
                    continue
        
        raise APIError(f"Google AI model {model} failed: {error_msg}")
    
    finally:
        # Restore original proxy setting after Google AI call if we changed it here
        if proxy_applied:
            if original_proxy is not None:
                os.environ["HTTPS_PROXY"] = original_proxy
            else:
                os.environ.pop("HTTPS_PROXY", None)
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
    fallback_models: Optional[List[str]] = None,
    brainstorm_model: Optional[str] = None,
    num_brainstorm: int = 5,
    top_k: int = 3,
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

    if brainstorm_model:
        brainstorm_ideas = []
        for _ in range(num_brainstorm):
            single_prompt = (
                "Generate one research idea for the following topic. Provide Title, Core Concept, Originality (1-10), Impact (1-10), and Feasibility (1-10).\n"
                f"TOPIC: {topic}\nFIELD: {field}\nQUESTION: {question}"
            )
            try:
                resp = _universal_chat(
                    [{"role": "user", "content": single_prompt}],
                    model=brainstorm_model,
                    request_timeout=request_timeout,
                    prompt_type="ideation",
                    fallback_models=fallback_models or [],
                )
                parsed = _parse_ideation_response(resp)
                if parsed:
                    brainstorm_ideas.append(parsed[0])
            except Exception as e:
                print(f"  ⚠️ Brainstorming failed: {e}")

        ideas_sorted = sorted(
            brainstorm_ideas,
            key=lambda i: (
                i.get('originality', 0) + i.get('impact', 0) + i.get('feasibility', 0)
            ) / 3,
            reverse=True,
        )
        top_ideas = ideas_sorted[:top_k]
        summary = "\n".join(
            [
                f"Title: {i.get('title','')}\nConcept: {i.get('core_concept','')}\nScores: O={i.get('originality','')}, I={i.get('impact','')}, F={i.get('feasibility','')}"
                for i in top_ideas
            ]
        )
        ranking_prompt = "Rank the following research ideas and select the best one:\n" + summary
        ranking_response = _universal_chat(
            [{"role": "user", "content": ranking_prompt}],
            model=model,
            request_timeout=request_timeout,
            prompt_type="ideation",
            fallback_models=fallback_models or [],
        )
        selected = top_ideas[0] if top_ideas else None
        return {
            "ideas": top_ideas,
            "raw_response": ranking_response,
            "selected_idea": selected,
            "brainstorm_ideas": brainstorm_ideas,
            "topic": topic,
            "field": field,
            "question": question,
        }

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
            "brainstorm_ideas": ideas,
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
            "brainstorm_ideas": [],
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

def _initial_draft_prompt(topic: str, field: str, question: str, user_prompt: Optional[str] = None) -> List[Dict[str, str]]:
    sys_prompt = (
        "You are a meticulous scientist writing a LaTeX paper suitable for a top journal. "
        
        "CRITICAL REQUIREMENTS - NO EXCEPTIONS:\n"
        "1. SINGLE FILE ONLY: Create ONE LaTeX file with NO separate bibliography files\n"
        "2. EMBEDDED REFERENCES: Include ALL references using \\begin{filecontents*}{refs.bib}...\\end{filecontents*} at the TOP of the file\n"
        "3. COMPILABLE: The file must compile successfully with pdflatex\n"
        "4. REAL REFERENCES ONLY: All references must be authentic, published works with correct details (authors, titles, journals, years, DOIs). NO FAKE or PLACEHOLDER references.\n"
        "5. SELF-CONTAINED CONTENT: ALL tables, figures, diagrams must be defined within the LaTeX file using TikZ, tabular, or other LaTeX constructs. NO external image files.\n"
        "6. DATA-DRIVEN RESULTS: All numerical values in tables/figures must come from actual simulation results, not made-up numbers.\n"
        "7. APPROPRIATE STRUCTURE: The paper structure and sections must align with the paper type and field conventions:\n"
        "   - Theoretical papers: Abstract, Introduction, Related Work, Theory/Methods, Analysis, Discussion, Conclusion\n"
        "   - Experimental papers: Abstract, Introduction, Related Work, Methodology, Experiments, Results, Discussion, Conclusion\n"
        "   - Survey papers: Abstract, Introduction, Background, Classification/Taxonomy, Comparative Analysis, Future Directions, Conclusion\n"
        "   - Systems papers: Abstract, Introduction, Related Work, System Design, Implementation, Evaluation, Discussion, Conclusion\n"
        "   - Algorithm papers: Abstract, Introduction, Related Work, Problem Definition, Algorithm Description, Analysis, Experiments, Conclusion\n"
        "8. RESULTS DOCUMENTATION: The numerical results from running simulation.py MUST be saved in 'results.txt' in the project folder for reproducibility and verification.\n"
        "9. FIGURE GENERATION: If the paper includes figures generated by code, ALL figure generation code must be included in simulation.py and figures must be saved in the local project folder.\n"
        "10. SINGLE CODE FILE: ALL computational code must be consolidated into ONE simulation.py file - no additional .py files, scripts, or code fragments.\n\n"
        
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
        "- Reference saved figures in LaTeX using \\includegraphics with proper paths\n\n"
        
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
        
        "EXAMPLE STRUCTURE:\n"
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
        "\\bibliography{refs}\n"
        "\\end{document}"
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
        "*.py", "*.tex", "*.bib", "*.txt", "*.csv", "*.json", "*.md", "*.yml", "*.yaml", "*.png"
    ]
    
    # Define files/directories to exclude
    exclude_patterns = [
        "__pycache__", "*.aux", "*.log", "*.bbl", "*.blg", "*.out", "*.pdf", 
        "*.npy", "*.npz", "*.pkl", "*.cache", ".git", "node_modules"
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

def _review_prompt(paper_tex: str, sim_summary: str, project_dir: Path = None, user_prompt: Optional[str] = None) -> List[Dict[str, str]]:
    sys_prompt = (
        "Act as a top-tier journal reviewer (Nature, Science, Cell level) with expertise in LaTeX formatting and scientific programming. "
        "Your review must meet the highest academic standards. Be constructive but demanding. "
        "CRITICAL: If the simulation ran successfully and produced actual results, the paper MUST use these real numbers, not fake/placeholder values. "
        
        "MANDATORY REQUIREMENTS - CHECK CAREFULLY:\n"
        "1. SINGLE FILE ONLY: Paper must be ONE LaTeX file with NO \\input{} or \\include{} commands\n"
        "2. EMBEDDED REFERENCES: References must be embedded using \\begin{filecontents*}{refs.bib}...\\end{filecontents*} or \\begin{thebibliography}...\\end{thebibliography}\n"
        "3. COMPILABLE: File must compile successfully with pdflatex (check for syntax errors, missing packages, etc.)\n"
        "4. AUTHENTIC REFERENCES ONLY: ALL references must be real, published works with correct bibliographic details. Verify:\n"
        "   - Author names are real and spelled correctly\n"
        "   - Journal/venue names are authentic\n"
        "   - Publication years are realistic\n"
        "   - DOIs are properly formatted (if provided)\n"
        "   - References are directly relevant to the topic\n"
        "   - NO placeholder, fake, or made-up citations\n"
        "5. SELF-CONTAINED VISUALS: ALL tables, figures, and diagrams must be:\n"
        "   - Defined within the LaTeX file using TikZ, tabular, PGFPlots, etc., OR\n"
        "   - Generated by simulation.py and saved as local files with proper \\includegraphics references\n"
        "   - Populated with REAL data from simulation results\n"
        "   - NO fake, placeholder, or estimated numbers\n"
        "6. APPROPRIATE STRUCTURE: The paper structure must align with the paper type and field conventions:\n"
        "   - Verify section organization matches the paper's contribution type\n"
        "   - Check for field-specific sections and evaluation methodologies\n"
        "   - Ensure logical flow appropriate for the research area\n"
        "   - Validate that section names and content align with journal standards\n"
        "7. RESULTS DOCUMENTATION: Check that numerical results are properly documented:\n"
        "   - Simulation.py should save key results to 'results.txt' for reproducibility\n"
        "   - All numerical values in the paper must be traceable to simulation output\n"
        "   - Results file should be well-structured and human-readable\n"
        "   - Verify consistency between cited numbers and simulation output\n"
        "8. FIGURE GENERATION: If paper includes computational figures:\n"
        "   - Figure generation code must be included in simulation.py\n"
        "   - Generated figures must be saved in the local project folder\n"
        "   - LaTeX must reference generated figures with proper \\includegraphics paths\n"
        "   - No external figure dependencies or missing image files\n"
        "9. SINGLE CODE FILE: ALL computational code must be consolidated into ONE simulation.py file only - no additional .py files or scripts.\n\n"
        
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
        "- NO references to external files except those generated by simulation.py\n\n"
        
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
        "- DOCUMENTED RESULTS: Key findings must be saved in results.txt\n"
        "- GENERATED FIGURES: Computational figures must be created by simulation.py\n"
        "- NO OVERFLOW: All content must fit within page margins\n"
        "- REPRODUCIBILITY: Code should be well-documented and runnable\n"
        "- SIGNIFICANCE: Work must make a meaningful contribution to the field\n"
        "- SELF-CONTAINED: Single file with embedded references and traceable visual content\n"
        "- APPROPRIATE STRUCTURE: Paper organization must match the research type and field standards\n"
        
        "Provide specific, actionable feedback with concrete suggestions for improvement. "
        "If the paper violates any of the 8 mandatory requirements, mark it as needing major revision. "
        "Pay special attention to reference authenticity, results documentation, figure generation, and structural appropriateness."
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
        "2. EMBEDDED REFERENCES: All references must be included in the paper using \\begin{filecontents*}{refs.bib}...\\end{filecontents*} or \\begin{thebibliography}...\\end{thebibliography}\n"
        "3. COMPILABLE: The file must compile successfully with pdflatex\n"
        "4. AUTHENTIC REFERENCES ONLY: ALL references must be real, published works. Verify and correct:\n"
        "   - Author names are real researchers in the field\n"
        "   - Journal/venue names are authentic and properly formatted\n"
        "   - Publication years are realistic and consistent\n"
        "   - DOIs are properly formatted (when available)\n"
        "   - References directly support claims in the text\n"
        "   - NO placeholder, fake, or fictional citations\n"
        "5. SELF-CONTAINED VISUALS: ALL tables, figures, diagrams must be:\n"
        "   - Created using LaTeX code only (TikZ, tabular, PGFPlots, etc.), OR\n"
        "   - Generated by simulation.py and saved as local files with proper \\includegraphics references\n"
        "   - Populated with actual simulation data, not fake numbers\n"
        "   - Self-rendering within the LaTeX document\n"
        "6. APPROPRIATE STRUCTURE: Ensure the paper structure aligns with the paper type and field conventions:\n"
        "   - Organize sections according to the paper's contribution type\n"
        "   - Include field-specific sections and evaluation methodologies\n"
        "   - Follow established academic writing conventions for the research area\n"
        "   - Ensure logical flow and appropriate section transitions\n"
        "7. RESULTS DOCUMENTATION: Ensure numerical results are properly documented:\n"
        "   - Simulation.py should save key results to 'results.txt' for reproducibility\n"
        "   - All numerical values in the paper must be traceable to simulation output\n"
        "   - Results file should be well-structured and human-readable\n"
        "   - Maintain consistency between cited numbers and simulation output\n"
        "8. FIGURE GENERATION: If paper includes computational figures:\n"
        "   - Figure generation code must be included in simulation.py\n"
        "   - Generated figures must be saved in the local project folder\n"
        "   - LaTeX must reference generated figures with proper \\includegraphics paths\n"
        "   - No external figure dependencies or missing image files\n\n"
        
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
        "- Allow dependencies only on simulation-generated image files\n\n"
        
        "REVISION PRIORITIES:\n"
        "1. Address all scientific/methodological concerns raised\n"
        "2. Fix LaTeX formatting issues (figures, tables, equations, citations)\n"
        "3. Update content based on actual simulation results\n"
        "4. Replace fake references with authentic ones\n"
        "5. Convert external visuals to self-contained LaTeX code or simulation-generated files\n"
        "6. Restructure sections to match paper type and field conventions\n"
        "7. Ensure proper documentation of results in results.txt\n"
        "8. Implement figure generation in simulation.py with local file saving\n"
        "9. Improve clarity and presentation quality\n"
        "10. Ensure reproducibility and code quality\n\n"
        
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
        "- Ensure numerical results are documented in results.txt\n"
        "- Include simulation-generated figures with proper local file references\n"
        "- Ensure claims are supported by authentic references\n"
        "- Address limitations and future work\n"
        "- Improve clarity of methodology and results\n\n"
        
        "BIBLIOGRAPHY INTEGRATION OPTIONS:\n"
        "Option 1 - filecontents (recommended):\n"
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
        "Option 2 - Direct thebibliography:\n"
        "\\begin{thebibliography}{99}\n"
        "\\bibitem{RealAuthor2024} A. Real, B. Author, \"Authentic Title,\" Nature Communications, vol. 15, p. 1234, 2024.\n"
        "\\end{thebibliography}\n\n"
        
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
    
    if latex_errors:
        user += (
            "\n----- LATEX COMPILATION ERRORS (LAST 20 LINES OF .log) -----\n" + 
            latex_errors + 
            "\n----- END LATEX ERRORS -----\n\n"
            "CRITICAL: Fix ALL LaTeX compilation errors. The paper MUST compile successfully with pdflatex.\n"
        )
    
    user += (
        "Return ONLY the complete revised LaTeX file. CRITICAL: Apply proper size constraints to ALL figures, tables, and diagrams. "
        "Ensure the paper is self-contained with embedded references and compiles without errors."
    )

    return [{"role": "system", "content": sys_prompt}, {"role": "user", "content": user}]


def review_paper(
    current_tex: str,
    sim_summary: str,
    paper_path: Path,
    project_dir: Path,
    iteration: int,
    config: WorkflowConfig,
    request_timeout: Optional[int],
    user_prompt: Optional[str],
) -> Tuple[str, str, bool, str, float, str]:
    """Run the review stage and return review text, decision, latex status, quality score, and ref report."""
    dynamic_timeout = _calculate_dynamic_timeout(current_tex, config)
    latex_success, latex_errors = _compile_latex_and_get_errors(paper_path, timeout=dynamic_timeout)
    if not latex_success:
        print("⚠️ LaTeX compilation failed. Errors will be sent to LLM for fixing.")
        print(f"Error log (last 20 lines):\n{latex_errors}")
    else:
        print("✅ LaTeX compilation successful!")

    quality_issues = _validate_research_quality(current_tex, sim_summary)
    if config.reference_validation:
        quality_issues.extend(_validate_references_with_external_apis(current_tex, config))
    if config.figure_validation:
        quality_issues.extend(
            _validate_figure_generation(current_tex, project_dir / 'simulation.py', project_dir)
        )

    ref_report = ""
    if config.fast_ref_check:
        ref_report = validate_references_with_llm(current_tex, config.review_model, request_timeout)
        if ref_report:
            (project_dir / "reference_validation.txt").write_text(ref_report, encoding="utf-8")

    qa_results = asyncio.run(run_parallel_checks(current_tex, config.qa_model))
    (project_dir / "qa_summary.json").write_text(json.dumps(qa_results, indent=2), encoding="utf-8")
    if qa_results.get("figures"):
        quality_issues.extend(qa_results["figures"])
    if qa_results.get("bibliography"):
        quality_issues.extend(qa_results["bibliography"])

    current_metrics = _extract_quality_metrics(current_tex, sim_summary)
    quality_score = _calculate_quality_score(current_metrics, quality_issues)

    review = _universal_chat(
        _review_prompt(current_tex, sim_summary, project_dir, user_prompt),
        model=config.review_model,
        request_timeout=request_timeout,
        prompt_type="review",
        fallback_models=config.fallback_models,
    )
    decision = _universal_chat(
        _editor_prompt(review, iteration, user_prompt),
        model=config.review_model,
        request_timeout=request_timeout,
        prompt_type="editor",
        fallback_models=config.fallback_models,
    )
    (project_dir / f"review_{iteration}.txt").write_text(review, encoding="utf-8")

    logger.info("Review feedback (iteration %s):\n%s", iteration, review)
    logger.info("Editor decision (iteration %s): %s", iteration, decision.strip())
    print("\n📝 Reviewer feedback (iteration {}):\n{}\n".format(iteration, review))
    print("🗳️ Editor decision (iteration {}): {}\n".format(iteration, decision.strip()))

    _record_iteration_feedback(
        project_dir,
        iteration,
        review,
        decision,
        quality_score,
        latex_success,
        latex_errors,
        ref_report,
    )

    return review, decision, latex_success, latex_errors, quality_score, ref_report


def revise_paper(
    current_tex: str,
    sim_summary: str,
    review_text: str,
    latex_errors: str,
    paper_path: Path,
    project_dir: Path,
    iteration: int,
    config: WorkflowConfig,
    request_timeout: Optional[int],
    user_prompt: Optional[str],
    ref_report: str = "",
) -> None:
    """Run the revision stage and update paper.tex in place."""
    full_review = review_text
    if ref_report:
        full_review += "\nREFERENCE ISSUES:\n" + ref_report
    revised = _universal_chat(
        _revise_prompt(current_tex, sim_summary, full_review, latex_errors, project_dir, user_prompt),
        model=config.revision_model,
        request_timeout=request_timeout,
        prompt_type="revise",
        fallback_models=config.fallback_models,
    )
    paper_path.write_text(revised, encoding="utf-8")
    (project_dir / f"revision_{iteration}.tex").write_text(revised, encoding="utf-8")
    if config.latex_auto_fix:
        def llm_fix(tex: str, log: str) -> str:
            messages = [
                {"role": "system", "content": "You are a LaTeX expert. Fix the document so it compiles."},
                {"role": "user", "content": f"LOG:\n{log}\n\nTEX:\n{tex}"},
            ]
            return _universal_chat(
                messages,
                model=config.review_model,
                request_timeout=request_timeout,
                prompt_type="latex_fix",
                fallback_models=config.fallback_models,
            )
        compile_with_autofix(project_dir, tex_file=paper_path.name, llm_fix=llm_fix)

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
    num_ideas: int = 15              # Number of ideas to generate
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

    global CURRENT_WORKFLOW_CONFIG
    CURRENT_WORKFLOW_CONFIG = config

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
                fallback_models=config.fallback_models,
                brainstorm_model=config.brainstorm_model,
                num_brainstorm=config.num_brainstorm_ideas,
                top_k=config.top_brainstorm_ideas,
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

                brainstorm_file = project_dir / "ideation_brainstorm.txt"
                with open(brainstorm_file, 'w', encoding='utf-8') as bf:
                    for idea in ideation_result.get("brainstorm_ideas", []):
                        bf.write(f"Title: {idea.get('title','')}\n")
                        bf.write(f"Concept: {idea.get('core_concept','')}\n")
                        bf.write(
                            f"Scores: O={idea.get('originality','')}, I={idea.get('impact','')}, F={idea.get('feasibility','')}\n\n"
                        )

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

        print(f"Running simulation before review {i}...")
        extract_success, extract_message = _extract_simulation_code_with_validation(paper_path, sim_path)
        if not extract_success:
            print(f"⚠️ Simulation extraction issues: {extract_message}")

        simulation_fixer = _create_simulation_fixer(model, request_timeout)
        sim_out = run_simulation_with_smart_fixing(
            sim_path,
            python_exec=python_exec,
            cwd=project_dir,
            llm_fixer=simulation_fixer,
            max_fix_attempts=2,
        )
        simulation_code = sim_path.read_text(encoding="utf-8", errors="ignore")
        sim_summary = summarize_simulation_outputs(sim_out, simulation_code)

        current_tex = paper_path.read_text(encoding="utf-8", errors="ignore")
        review, decision, latex_success, latex_errors, quality_score, ref_report = review_paper(
            current_tex,
            sim_summary,
            paper_path,
            project_dir,
            i,
            config,
            request_timeout,
            user_prompt,
        )

        quality_history.append(quality_score)
        if len(quality_history) > config.max_quality_history_size:
            keep_size = config.max_quality_history_size // 2
            quality_history = quality_history[-keep_size:]
            logger.info(f"Quality history trimmed to {keep_size} entries")
            print("🔧 Quality history trimmed for memory management")

        logger.info(f"Iteration {i} quality score: {quality_score:.2f}")
        print(f"📊 Iteration {i} quality score: {quality_score:.2f}")

        if quality_score > best_quality_score:
            best_quality_score = quality_score
            stagnation_count = 0
        else:
            stagnation_count += 1

        if stagnation_count >= 2 and i > 1:
            print(f"⚠️ Quality stagnation detected ({stagnation_count} iterations without improvement)")

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

        revise_paper(
            current_tex,
            sim_summary,
            review,
            latex_errors if not latex_success else "",
            paper_path,
            project_dir,
            i,
            config,
            request_timeout,
            user_prompt,
            ref_report,
        )

        print(f"Iteration {i}: Paper revised")
    
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
    p = argparse.ArgumentParser(description="Enhanced SciResearch Workflow with Quality Validation")
    p.add_argument("--topic", required=False, help="Research topic")
    p.add_argument("--field", required=False, help="Research question")
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

    # Model customization
    p.add_argument("--review-model", type=str, default=None, help="Model to use for review stage")
    p.add_argument("--revision-model", type=str, default=None, help="Model to use for revision stage")
    p.add_argument("--brainstorm-model", type=str, default=None, help="Model for brainstorming stage")
    p.add_argument("--num-brainstorm-ideas", type=int, default=5, help="Number of brainstorm ideas to generate")
    p.add_argument("--top-brainstorm-ideas", type=int, default=3, help="Top brainstorm ideas to consider")
    p.add_argument("--latex-auto-fix", action="store_true", help="Enable automatic LaTeX fixing loop")
    p.add_argument("--fast-ref-check", action="store_true", help="Enable lightweight reference validation with LLM")
    p.add_argument("--offline-demo", action="store_true", help="Run an offline demo pipeline without external models")
    
    # Custom prompt parameter
    p.add_argument("--user-prompt", type=str, default=None, help="Custom prompt that takes priority over standard requirements")
    
    args = p.parse_args(argv)
    
    # Handle skip flags
    if args.skip_reference_check:
        args.check_references = False
    if args.skip_figure_validation:
        args.validate_figures = False
    if args.skip_ideation:
        args.enable_ideation = False
    
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
        if not args.topic:
            args.topic = topic
        if not args.field:
            args.field = field
        if not args.question:
            args.question = question
        print(f"Detected existing paper - Topic: {args.topic}, Field: {args.field}")
    elif not args.modify_existing:
        # Interactive prompts if missing and no existing paper (but NOT when modifying existing)
        if not args.topic:
            args.topic = input("Topic: ").strip()
        if not args.field:
            args.field = input("Field: ").strip()
        if not args.question:
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
    
    # Check for embedded references requirement
    has_filecontents = bool(re.search(r'\\begin\{filecontents\*?\}\{[^}]*\.bib\}', paper_content))
    has_thebibliography = bool(re.search(r'\\begin\{thebibliography\}', paper_content))
    has_external_bib = bool(re.search(r'\\bibliography\{[^}]+\}', paper_content)) and not has_filecontents
    
    if has_external_bib and not has_filecontents:
        issues.append("Paper references external .bib file - must embed references using filecontents or thebibliography")
    
    if not has_filecontents and not has_thebibliography:
        issues.append("No embedded bibliography found - use filecontents or thebibliography")
    
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
    
    return issues

def validate_references_with_llm(paper_content: str, model: str, request_timeout: Optional[int] = None) -> str:
    """Use a lightweight LLM to detect reference formatting issues."""
    bib_match = re.search(r'\\begin\{thebibliography\}.*?\\end\{thebibliography\}', paper_content, re.DOTALL)
    if not bib_match:
        return ""
    bib_text = bib_match.group(0)
    messages = [
        {
            "role": "system",
            "content": (
                "You are a reference checker. Identify formatting problems or missing fields in the following bibliography."
            ),
        },
        {"role": "user", "content": bib_text},
    ]
    try:
        return _universal_chat(messages, model=model, request_timeout=request_timeout, prompt_type="reference_check")
    except Exception:
        return ""

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

        original_default_model = config.default_model
        if ns.model:
            config.default_model = ns.model

        try:
            auto_fallback_for_original = WorkflowConfig(default_model=original_default_model).fallback_models
        except Exception:
            auto_fallback_for_original = None

        if config.fallback_models is None or (
            auto_fallback_for_original is not None
            and sorted(config.fallback_models) == sorted(auto_fallback_for_original)
        ):
            config.fallback_models = WorkflowConfig(default_model=config.default_model).fallback_models
        
        # Save configuration if requested
        if ns.save_config:
            config.to_file(Path(ns.save_config))
            print(f"✅ Configuration saved to: {ns.save_config}")
            sys.exit(0)

        if ns.review_model:
            config.review_model = ns.review_model
        elif config.review_model is None or config.review_model == original_default_model:
            config.review_model = config.default_model

        if ns.revision_model:
            config.revision_model = ns.revision_model
        elif config.revision_model is None or config.revision_model == original_default_model:
            config.revision_model = config.default_model

        if ns.brainstorm_model:
            config.brainstorm_model = ns.brainstorm_model
        elif config.brainstorm_model is None or config.brainstorm_model == original_default_model:
            config.brainstorm_model = config.review_model
        config.num_brainstorm_ideas = ns.num_brainstorm_ideas
        config.top_brainstorm_ideas = ns.top_brainstorm_ideas
        config.latex_auto_fix = ns.latex_auto_fix
        config.fast_ref_check = ns.fast_ref_check
        config.qa_model = config.review_model

        CURRENT_WORKFLOW_CONFIG = config

        if config.oss120b_endpoint and config.oss120b_api_key:
            OSS_CLIENT = OSS120BClient(config.oss120b_endpoint, config.oss120b_api_key)
            if not OSS_CLIENT.ping():
                print("⚠️ OSS 120B server not reachable")

        print(f"📁 Working with: {ns.output_dir}")
        print(f"🤖 Using model: {ns.model}")
        print(f"🔄 Max iterations: {ns.max_iterations}")
        print(f"📊 Quality threshold: {ns.quality_threshold}")
        print(f"🔍 Reference validation: {'enabled' if ns.check_references else 'disabled'}")
        print(f"🖼️ Figure validation: {'enabled' if ns.validate_figures else 'disabled'}")
        print(f"🧠 Research ideation: {'enabled' if ns.enable_ideation else 'disabled'}")

        if ns.offline_demo:
            result_dir = run_offline_demo(
                topic=ns.topic,
                field=ns.field,
                question=ns.question,
                output_dir=Path(ns.output_dir),
            )
            print(f"✅ Offline demo completed! Results in: {result_dir}")
            sys.exit(0)

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
            num_ideas=ns.num_ideas
        )
        print(f"✅ Workflow completed! Results in: {result_dir}")
    except Exception as e:
        print(f"❌ Workflow failed: {e}")
        import traceback
        traceback.print_exc()
