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
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple

# Local helpers
from utils.sim_runner import ensure_single_tex_py, extract_simulation_from_tex, run_simulation_with_smart_fixing, summarize_simulation_outputs

DEFAULT_MODEL = os.environ.get("SCI_MODEL", "gpt-5")

def _openai_chat(messages: List[Dict[str, str]], model: str, request_timeout: Optional[int] = None) -> str:
    """
    A minimal OpenAI chat wrapper that works with both old and new SDKs.
    Includes retry logic for timeout errors.
    """
    print(f"🤖 Making API call to {model}...")
    
    # Set longer timeout for GPT-5
    if request_timeout is None:
        request_timeout = 300 if "gpt-5" in model.lower() else 120
    
    max_retries = 3
    for attempt in range(max_retries):
        try:
            # Newer SDK
            from openai import OpenAI
            client = OpenAI()
            # GPT-5 only supports temperature=1, other models can use 0.2
            temp = 1.0 if "gpt-5" in model.lower() else 0.2
            print(f"📡 Sending request with temperature={temp}, timeout={request_timeout}s (attempt {attempt + 1}/{max_retries})...")
            
            resp = client.chat.completions.create(
                model=model, 
                messages=messages, 
                temperature=temp, 
                timeout=request_timeout
            )
            print("✅ API call successful!")
            return resp.choices[0].message.content
            
        except Exception as e:
            error_msg = str(e)
            print(f"⚠️ API Error (attempt {attempt + 1}): {error_msg}")
            
            if "timeout" in error_msg.lower() and attempt < max_retries - 1:
                wait_time = (attempt + 1) * 30  # Progressive backoff: 30s, 60s
                print(f"🔄 Retrying in {wait_time} seconds...")
                time.sleep(wait_time)
                continue
            else:
                print(f"❌ OpenAI API Error after {max_retries} attempts: {e}")
                raise

def _create_simulation_fixer(model: str, request_timeout: Optional[int] = None):
    """
    Returns a callable that can analyze simulation errors and decide what to do.
    """
    def _fix_simulation(code: str, stdout: str, stderr: str, return_code: int) -> Dict[str, str]:
        sys_prompt = (
            "You are a Python expert helping with simulation code. Given a Python script and its error output, "
            "decide what action to take. Respond with JSON only.\n\n"
            "Response format:\n"
            "- If the error is acceptable (e.g., encoding issues that don't affect results): {\"action\": \"accept\"}\n"
            "- If the code needs fixing: {\"action\": \"fix_code\", \"fixed_code\": \"<complete fixed code>\"}\n"
            "- If modules need installing: {\"action\": \"install_modules\", \"modules\": [\"module1\", \"module2\"]}\n\n"
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
            response = _openai_chat(
                [{"role": "system", "content": sys_prompt}, {"role": "user", "content": user}],
                model=model,
                request_timeout=request_timeout,
            )
            # Try to parse JSON response
            return json.loads(response.strip())
        except Exception as e:
            print(f"LLM fixer error: {e}")
            return {"action": "accept"}  # Fallback to accepting results
    
    return _fix_simulation

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

def _initial_draft_prompt(topic: str, field: str, question: str) -> List[Dict[str, str]]:
    sys_prompt = (
        "You are a meticulous scientist writing a LaTeX paper suitable for a top journal. "
        "Always produce compilable LaTeX. Figures must use \\includegraphics[width=\\linewidth]{...}. "
        "Wrap wide tables using adjustbox width=\\linewidth. Include Abstract, Intro, Related Work, "
        "Methodology, Experiments, Results, Discussion, Conclusion, References. "
        "If you need code, include Python blocks using lstlisting or minted."
    )
    user_prompt = f"Topic: {topic}\nField: {field}\nResearch Question: {question}\n\nDraft the full LaTeX paper."
    return [{"role": "system", "content": sys_prompt}, {"role": "user", "content": user_prompt}]

def _review_prompt(paper_tex: str, sim_summary: str) -> List[Dict[str, str]]:
    sys_prompt = (
        "Act as a top-tier journal reviewer (Nature, Science, Cell level) with expertise in LaTeX formatting and scientific programming. "
        "Your review must meet the highest academic standards. Be constructive but demanding. "
        "CRITICAL: If the simulation ran successfully and produced actual results, the paper MUST use these real numbers, not fake/placeholder values. "
        
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
        "- Figure width constraints (MUST use width=\\linewidth, never exceed page margins)\n"
        "- Table formatting (ALL wide tables MUST use adjustbox with width=\\linewidth)\n"
        "- TikZ diagrams MUST be constrained (use adjustbox or scale to fit page width)\n"
        "- Mathematical notation and equation formatting\n"
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
        "- NO OVERFLOW: All figures, tables, diagrams must fit within page margins\n"
        "- REPRODUCIBILITY: Code should be well-documented and runnable\n"
        "- SIGNIFICANCE: Work must make a meaningful contribution to the field\n"
        
        "Provide specific, actionable feedback with concrete suggestions for improvement."
    )
    user = (
        "Here is the current paper (LaTeX):\n\n"
        "----- BEGIN LATEX -----\n" + paper_tex + "\n----- END LATEX -----\n\n"
        "Here are the actual simulation results - the paper MUST use these real numbers:\n\n"
        "----- BEGIN SIMULATION & OUTPUTS -----\n" + sim_summary + "\n----- END SIMULATION & OUTPUTS -----\n\n"
        "Write a constructive review. Check that ALL figures/tables/diagrams use proper size constraints to prevent page overflow. "
        "If the simulation produced valid results, ensure the paper uses ONLY these actual numbers, not fake values."
    )
    return [{"role": "system", "content": sys_prompt}, {"role": "user", "content": user}]

def _editor_prompt(review_text: str, iteration_count: int) -> List[Dict[str, str]]:
    sys_prompt = (
        "You are the handling editor of a top-tier journal. Make an informed decision about paper readiness. "
        "Consider both the reviewer's feedback and the iteration history. "
        "Papers should only be accepted when they meet publication standards for impact, rigor, and clarity."
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

def _revise_prompt(paper_tex: str, sim_summary: str, review_text: str) -> List[Dict[str, str]]:
    sys_prompt = (
        "You are the paper author making revisions based on peer review. Your goal is to address ALL reviewer concerns "
        "while maintaining scientific integrity and clarity. Produce a COMPLETE revised LaTeX file.\n\n"
        
        "REVISION PRIORITIES:\n"
        "1. Address all scientific/methodological concerns raised\n"
        "2. Fix LaTeX formatting issues (figures, tables, equations, citations)\n"
        "3. Update content based on actual simulation results\n"
        "4. Improve clarity and presentation quality\n"
        "5. Ensure reproducibility and code quality\n\n"
        
        "FORMATTING REQUIREMENTS (CRITICAL - NO EXCEPTIONS):\n"
        "- ALL figures: \\includegraphics[width=\\linewidth]{...} (never exceed page width)\n"
        "- ALL wide tables: \\begin{adjustbox}{width=\\linewidth}...\\end{adjustbox}\n"
        "- ALL TikZ diagrams: wrap in \\begin{adjustbox}{width=\\linewidth}...\\end{adjustbox}\n"
        "- ALL content must fit within vertical page margins\n"
        "- Proper math environments and symbol usage\n"
        "- Consistent citation style and complete bibliography\n"
        "- Clear section structure and logical flow\n\n"
        
        "CONTENT REQUIREMENTS:\n"
        "- Use ONLY actual simulation results (no fake numbers)\n"
        "- Ensure claims are supported by evidence\n"
        "- Address limitations and future work\n"
        "- Improve clarity of methodology and results\n\n"
        
        "SIZE CONSTRAINT EXAMPLES:\n"
        "- Figures: \\includegraphics[width=\\linewidth]{figure.png}\n"
        "- Tables: \\begin{adjustbox}{width=\\linewidth}\\begin{tabular}...\\end{tabular}\\end{adjustbox}\n"
        "- TikZ: \\begin{adjustbox}{width=\\linewidth}\\begin{tikzpicture}...\\end{tikzpicture}\\end{adjustbox}\n\n"
        
        "Return ONLY the complete revised LaTeX file with ALL issues addressed and ALL size constraints properly applied."
    )
    user = (
        "----- CURRENT PAPER (LATEX) -----\n" + paper_tex + "\n"
        "----- SIMULATION CODE & OUTPUTS -----\n" + sim_summary + "\n"
        "----- REVIEW FEEDBACK -----\n" + review_text + "\n"
        "Return ONLY the complete revised LaTeX file. CRITICAL: Apply proper size constraints to ALL figures, tables, and diagrams."
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
) -> Path:
    """Enhanced workflow with quality validation."""
    
    project_dir = _prepare_project_dir(output_dir, modify_existing)
    paper_path, sim_path = ensure_single_tex_py(project_dir, strict=strict_singletons)

    # Check if this is actually a minimal template (new paper) or has real content
    paper_content = paper_path.read_text(encoding="utf-8").strip()
    is_minimal_template = (paper_content == "\\documentclass{article}\\begin{document}\\end{document}" or len(paper_content) < 200)
    
    # If no real paper content yet (fresh), draft one.
    if is_minimal_template:
        print("Creating new paper from scratch...")
        draft = _openai_chat(_initial_draft_prompt(topic, field, question), model=model, request_timeout=request_timeout)
        paper_path.write_text(draft, encoding="utf-8")
    else:
        print(f"Using existing paper content ({len(paper_content)} characters)")

    # Extract/refresh simulation.py from LaTeX initially
    extract_simulation_from_tex(paper_path, sim_path)

    # Review-Revise loop with quality tracking
    for i in range(1, max_iterations + 1):
        print(f"Starting iteration {i} of {max_iterations}")
        
        # ALWAYS run simulation before each review to get current results
        print(f"Running simulation before review {i}...")
        extract_simulation_from_tex(paper_path, sim_path)  # Refresh code from paper
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
        
        # Quality validation before review
        current_tex = paper_path.read_text(encoding="utf-8", errors="ignore")
        quality_issues = _validate_research_quality(current_tex, sim_summary)
        if quality_issues:
            print(f"Quality issues detected: {', '.join(quality_issues)}")
        
        # Enhanced review with quality metrics
        review = _openai_chat(_review_prompt(current_tex, sim_summary), model=model, request_timeout=request_timeout)
        
        # Enhanced editorial decision with iteration count
        decision = _openai_chat(_editor_prompt(review, i), model=model, request_timeout=request_timeout)
        
        if decision.strip().upper().startswith("YES"):
            print(f"[OK] Editor accepted at iteration {i}.")
            # Final quality check
            final_metrics = _extract_quality_metrics(current_tex, sim_summary)
            print(f"Final paper metrics: {final_metrics}")
            break
        elif decision.strip().upper().startswith("REJECT"):
            print(f"[REJECT] Editor rejected the paper at iteration {i}.")
            print("Paper has fundamental issues but continuing with revisions to improve it...")
            # Don't break - continue with revision to try to fix issues
            
        # Continue with revision (whether NO or REJECT)
        print(f"Revising paper based on feedback...")
        revised = _openai_chat(_revise_prompt(current_tex, sim_summary, review), model=model, request_timeout=request_timeout)
        paper_path.write_text(revised, encoding="utf-8")
        
        print(f"Iteration {i}: Paper revised")

    return project_dir

def _extract_paper_metadata(paper_content: str) -> Tuple[str, str, str]:
    """Extract topic, field, and research question from existing paper content."""
    # Try to extract from title and abstract
    title_match = re.search(r'\\title\{([^}]+)\}', paper_content)
    abstract_match = re.search(r'\\begin\{abstract\}(.*?)\\end\{abstract\}', paper_content, re.DOTALL)
    
    title = title_match.group(1) if title_match else "Research Paper"
    abstract = abstract_match.group(1).strip() if abstract_match else ""
    
    # Extract key concepts from title and abstract to infer field and topic
    field = "Computer Science"  # Default field
    topic = title.split(':')[0] if ':' in title else title
    question = f"How to improve the methodology and results presented in: {title}"
    
    # Try to be more specific based on content
    content_lower = (title + " " + abstract).lower()
    if any(word in content_lower for word in ['quantum', 'qubit', 'entangle']):
        field = "Quantum Computing"
    elif any(word in content_lower for word in ['neural', 'deep learning', 'machine learning', 'ai']):
        field = "Machine Learning"
    elif any(word in content_lower for word in ['biology', 'protein', 'gene', 'medical']):
        field = "Computational Biology"
    elif any(word in content_lower for word in ['security', 'crypto', 'attack', 'defense']):
        field = "Cybersecurity"
    
    return topic, field, question

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
    p = argparse.ArgumentParser()
    p.add_argument("--topic", required=False, help="Research topic")
    p.add_argument("--field", required=False, help="Field")
    p.add_argument("--question", required=False, help="Research question")
    p.add_argument("--output-dir", default="output", help="Output directory root (contains project subfolder)")
    p.add_argument("--model", default=DEFAULT_MODEL, help="OpenAI model to use (default: gpt-5)")
    p.add_argument("--request-timeout", type=int, default=3600, help="Per-request timeout seconds (0 means no timeout)")
    p.add_argument("--max-retries", type=int, default=3, help="Max OpenAI retries")
    p.add_argument("--max-iterations", type=int, default=4, help="Max review->revise iterations")
    p.add_argument("--modify-existing", action="store_true", help="If output dir already has paper.tex, modify in place")
    p.add_argument("--strict-singletons", action="store_true", default=True, help="Keep only paper.tex & simulation.py (others archived)")
    p.add_argument("--python-exec", default=None, help="Python interpreter for running simulation.py")
    args = p.parse_args(argv)
    
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
    else:
        # Interactive prompts if missing and no existing paper
        if not args.topic:
            args.topic = input("Topic: ").strip()
        if not args.field:
            args.field = input("Field: ").strip()
        if not args.question:
            args.question = input("Research question: ").strip()
    
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
    if not re.search(r'\\bibliography\{|\\begin\{thebibliography\}', paper_content):
        issues.append("No bibliography section found")
    
    # Check for reasonable number of references
    bib_entries = len(re.findall(r'\\bibitem\{|@\w+\{', paper_content))
    if bib_entries < 10:
        issues.append(f"Only {bib_entries} bibliography entries (recommend 10+)")
    
    # Add figure/table validation
    figure_table_issues = _validate_figures_tables(paper_content)
    issues.extend(figure_table_issues)
    
    return issues

def _validate_figures_tables(paper_content: str) -> List[str]:
    """Validate figure and table formatting to prevent overflow."""
    issues = []
    
    # Check for proper figure widths - must use width=\linewidth
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
    print("🚀 Starting SciResearch Workflow...")
    try:
        ns = parse_args()
        print(f"📁 Working with: {ns.output_dir}")
        print(f"🤖 Using model: {ns.model}")
        print(f"🔄 Max iterations: {ns.max_iterations}")
        
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
        )
        print(f"✅ Workflow completed! Results in: {result_dir}")
    except Exception as e:
        print(f"❌ Workflow failed: {e}")
        import traceback
        traceback.print_exc()
