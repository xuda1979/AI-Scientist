#!/usr/bin/env python3
"""
Content generation utilities for research papers.
"""
from __future__ import annotations
import difflib
import json
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any


def _save_candidate_diff(old_content: str, new_content: str, candidate_num: int, prefix: str = "candidate"):
    """Save diff between old and new content for debugging."""
    diff_lines = list(difflib.unified_diff(
        old_content.splitlines(keepends=True),
        new_content.splitlines(keepends=True),
        fromfile=f"{prefix}_old.tex",
        tofile=f"{prefix}_new_{candidate_num}.tex",
        n=3
    ))
    
    if diff_lines:
        print(f"\n--- {prefix.upper()} {candidate_num} DIFF ---")
        print(''.join(diff_lines[:50]))  # Show first 50 lines
        if len(diff_lines) > 50:
            print(f"... ({len(diff_lines) - 50} more lines)")
        print("--- END DIFF ---\n")


def _save_iteration_diff(old_content: str, new_content: str, output_dir: Path, iteration: int, filename: str = "paper.tex"):
    """Save iteration diff to file and display summary."""
    if not output_dir.exists():
        output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate diff
    diff_lines = list(difflib.unified_diff(
        old_content.splitlines(keepends=True),
        new_content.splitlines(keepends=True),
        fromfile=f"{filename}",
        tofile=f"{filename}",
        n=3
    ))
    
    if diff_lines:
        # Save to file
        diff_file = output_dir / f"iteration_{iteration}_diff.patch"
        with open(diff_file, 'w', encoding='utf-8') as f:
            f.writelines(diff_lines)
        
        print(f"\n{'='*80}")
        print(f"GIT DIFF FOR ITERATION {iteration} - {filename}")
        print(f"{'='*80}")
        
        # Show diff in git-style format
        print(''.join(diff_lines))
        print(f"{'='*80}\n")


def _parse_combined_response(response: str, project_dir: Path) -> tuple[str, str, dict]:
    """Parse combined review/revision response and extract file changes."""
    import re
    
    # Extract sections using regex
    review_match = re.search(r'## REVIEW\s*\n(.*?)(?=## REVISION DIFFS)', response, re.DOTALL | re.IGNORECASE)
    diffs_match = re.search(r'## REVISION DIFFS.*?\n(.*)', response, re.DOTALL | re.IGNORECASE)
    
    # Extract review text
    review_text = review_match.group(1).strip() if review_match else "No review section found."
    
    # Extract file changes
    file_changes = {}
    if diffs_match:
        diffs_content = diffs_match.group(1)
        
        # Look for file blocks in various formats
        file_patterns = [
            r'```(?:tex|latex)\s*\n# File: ([^\n]+)\n(.*?)```',
            r'```python\s*\n# File: ([^\n]+)\n(.*?)```',
            r'```\s*\n# File: ([^\n]+)\n(.*?)```',
            r'# File: ([^\n]+)\n```(?:tex|latex|python)?\s*\n(.*?)```'
        ]
        
        for pattern in file_patterns:
            matches = re.finditer(pattern, diffs_content, re.DOTALL | re.IGNORECASE)
            for match in matches:
                filename = match.group(1).strip()
                content = match.group(2).strip()
                file_changes[filename] = content
    
    return review_text, "REVISE", file_changes


def _apply_file_changes(file_changes: dict, project_dir: Path, config=None) -> bool:
    """Apply file changes with content protection."""
    from utils.content_protection import ContentProtector
    
    if not file_changes:
        return False
    
    protector = ContentProtector() if config and config.content_protection else None
    changes_applied = False
    
    for filename, new_content in file_changes.items():
        file_path = project_dir / filename
        
        try:
            # Read existing content if file exists
            if file_path.exists():
                old_content = file_path.read_text(encoding='utf-8', errors='ignore')
                
                # Apply content protection if enabled
                if protector and filename.endswith('.tex'):
                    protection_result = protector.validate_changes(old_content, new_content)
                    
                    if not protection_result.approved:
                        print(f"⚠ Content protection prevented changes to {filename}")
                        print(f"  Reason: {protection_result.reason}")
                        continue
                    elif protection_result.warnings:
                        print(f"⚠ Content protection warnings for {filename}:")
                        for warning in protection_result.warnings:
                            print(f"   {warning}")
            else:
                old_content = ""
            
            # Write new content
            file_path.write_text(new_content, encoding='utf-8')
            
            # Show content metrics
            old_words = len(old_content.split()) if old_content else 0
            new_words = len(new_content.split())
            word_change = ((new_words - old_words) / old_words * 100) if old_words > 0 else 100
            
            print(f"✓ Updated {filename}: {old_words:,} → {new_words:,} words ({word_change:+.1f}%)")
            changes_applied = True
            
        except Exception as e:
            print(f"✗ Failed to update {filename}: {e}")
            continue
    
    return changes_applied


def _select_best_candidate_with_llm(candidates: List[str], criteria: str, topic: str, field: str, 
                                   model: str, request_timeout: int) -> Tuple[str, int]:
    """Use LLM to select the best candidate from multiple options."""
    from ai.chat import _universal_chat
    
    if len(candidates) <= 1:
        return candidates[0] if candidates else "", 0
    
    # Prepare comparison prompt
    comparison_text = ""
    for i, candidate in enumerate(candidates, 1):
        comparison_text += f"\n--- CANDIDATE {i} ---\n{candidate[:2000]}...\n"
    
    messages = [{
        "role": "system",
        "content": (
            f"You are an expert evaluator for {field} research papers. "
            "Your task is to select the best candidate based on the given criteria."
        )
    }, {
        "role": "user",
        "content": (
            f"Topic: {topic}\n"
            f"Field: {field}\n"
            f"Evaluation Criteria: {criteria}\n\n"
            f"Please evaluate these candidates and select the best one:\n"
            f"{comparison_text}\n\n"
            f"Respond with only the number of the best candidate (1, 2, 3, etc.) "
            f"followed by a brief explanation."
        )
    }]
    
    try:
        response = _universal_chat(messages, model, request_timeout, "candidate_selection")
        
        # Extract candidate number
        import re
        number_match = re.search(r'\b(\d+)\b', response)
        if number_match:
            selected_idx = int(number_match.group(1)) - 1
            if 0 <= selected_idx < len(candidates):
                return candidates[selected_idx], selected_idx
        
        # Fallback to first candidate
        return candidates[0], 0
        
    except Exception as e:
        print(f"LLM selection failed: {e}, using quality-based selection")
        # Fallback to quality-based selection
        from evaluation.quality import _evaluate_response_quality
        
        best_candidate = candidates[0]
        best_score = 0.0
        best_idx = 0
        
        for i, candidate in enumerate(candidates):
            quality_metrics = _evaluate_response_quality(candidate)
            score = sum(quality_metrics.values()) / len(quality_metrics)
            
            if score > best_score:
                best_score = score
                best_candidate = candidate
                best_idx = i
        
        return best_candidate, best_idx


def _generate_research_ideas(topic: str, field: str, question: str, model: str, 
                           request_timeout: int, num_ideas: int = 10) -> List[Dict[str, Any]]:
    """Generate research ideas for ideation phase."""
    from ai.chat import _universal_chat
    
    messages = [{
        "role": "system",
        "content": (
            "You are a world-class researcher with expertise in generating innovative "
            "research ideas. Provide creative, feasible, and impactful research directions."
        )
    }, {
        "role": "user",
        "content": (
            f"Generate {num_ideas} innovative research ideas related to:\n"
            f"Topic: {topic}\n"
            f"Field: {field}\n"
            f"Question: {question}\n\n"
            
            "For each idea, provide:\n"
            "1. Title (concise, descriptive)\n"
            "2. Description (2-3 sentences)\n"
            "3. Innovation level (1-10)\n"
            "4. Feasibility (1-10)\n"
            "5. Impact potential (1-10)\n\n"
            
            "Format each idea as:\n"
            "IDEA N:\n"
            "Title: [title]\n"
            "Description: [description]\n"
            "Innovation: [score]\n"
            "Feasibility: [score]\n"
            "Impact: [score]\n"
        )
    }]
    
    try:
        response = _universal_chat(messages, model, request_timeout, "ideation")
        return _parse_ideation_response(response)
    except Exception as e:
        print(f"Ideation failed: {e}")
        return []


def _parse_ideation_response(response: str) -> List[Dict[str, Any]]:
    """Parse ideation response and extract structured ideas."""
    ideas = []
    
    # Split into individual ideas
    idea_sections = re.split(r'IDEA \d+:', response, re.IGNORECASE)[1:]
    
    for section in idea_sections:
        try:
            idea = {}
            
            # Extract title
            title_match = re.search(r'Title:\s*([^\n]+)', section, re.IGNORECASE)
            idea['title'] = title_match.group(1).strip() if title_match else "Untitled"
            
            # Extract description
            desc_match = re.search(r'Description:\s*([^\n]+(?:\n[^\n]*)*?)(?=Innovation:|Feasibility:|Impact:|\Z)', 
                                 section, re.IGNORECASE | re.DOTALL)
            idea['description'] = desc_match.group(1).strip() if desc_match else "No description"
            
            # Extract scores
            innovation_match = re.search(r'Innovation:\s*(\d+)', section, re.IGNORECASE)
            feasibility_match = re.search(r'Feasibility:\s*(\d+)', section, re.IGNORECASE)
            impact_match = re.search(r'Impact:\s*(\d+)', section, re.IGNORECASE)
            
            idea['innovation'] = int(innovation_match.group(1)) if innovation_match else 5
            idea['feasibility'] = int(feasibility_match.group(1)) if feasibility_match else 5
            idea['impact'] = int(impact_match.group(1)) if impact_match else 5
            
            # Calculate composite score
            idea['score'] = (idea['innovation'] + idea['feasibility'] + idea['impact']) / 3.0
            
            ideas.append(idea)
            
        except Exception as e:
            print(f"Failed to parse idea section: {e}")
            continue
    
    # Sort by score (highest first)
    ideas.sort(key=lambda x: x['score'], reverse=True)
    
    return ideas


def _extract_paper_metadata(paper_content: str) -> Tuple[str, str, str]:
    """Extract topic, field, and question from existing paper."""
    # Extract title
    title_patterns = [
        r'\\title\{([^}]+)\}',
        r'\\begin\{document\}\s*\\title\{([^}]+)\}',
        r'\\maketitle.*?([^\n]+)',
    ]
    
    title = "Unknown Topic"
    for pattern in title_patterns:
        match = re.search(pattern, paper_content, re.DOTALL | re.IGNORECASE)
        if match:
            title = match.group(1).strip()
            break
    
    # Infer field from content
    content_lower = paper_content.lower()
    field_keywords = {
        'Computer Science': ['algorithm', 'computer', 'software', 'programming', 'network'],
        'Machine Learning': ['machine learning', 'neural network', 'deep learning', 'AI'],
        'Quantum Computing': ['quantum', 'qubit', 'quantum computing', 'quantum algorithm'],
        'Security': ['security', 'cryptography', 'encryption', 'attack', 'defense'],
        'Mathematics': ['theorem', 'proof', 'mathematical', 'equation', 'formula'],
        'Physics': ['physics', 'particle', 'energy', 'force', 'quantum mechanics'],
        'Engineering': ['engineering', 'system', 'design', 'implementation', 'architecture']
    }
    
    field = "Computer Science"  # Default
    max_matches = 0
    
    for field_name, keywords in field_keywords.items():
        matches = sum(1 for keyword in keywords if keyword in content_lower)
        if matches > max_matches:
            max_matches = matches
            field = field_name
    
    # Extract or infer research question
    question_patterns = [
        r'research question[s]?:?\s*([^.]+)',
        r'this paper (?:addresses|investigates|explores|examines)\s+([^.]+)',
        r'we (?:investigate|explore|examine|address)\s+([^.]+)',
    ]
    
    question = f"How to improve {title.lower()}?"
    for pattern in question_patterns:
        match = re.search(pattern, content_lower, re.IGNORECASE)
        if match:
            question = match.group(1).strip()
            break
    
    return title, field, question


def _create_simulation_fixer(model: str, request_timeout: Optional[int] = None):
    """
    Returns a callable that can analyze simulation errors and decide what to do.
    """
    from ..ai.chat import _universal_chat
    from ..core.config import _validate_code_security, SecurityError
    
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
    pdf_path: Optional[Path] = None,
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
        from ..ai.chat import _universal_chat
        response = _universal_chat(
            evaluation_prompt, 
            model=model, 
            request_timeout=request_timeout, 
            prompt_type="revision_candidate_evaluation",
            fallback_models=config.fallback_models,
            pdf_path=None
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
    pdf_path: Optional[Path] = None,
    project_dir: Optional[Path] = None,
) -> str:
    """
    Use an LLM to evaluate and select the best candidate from multiple options.
    
    Args:
        candidates: List of candidates to evaluate
        original_content: Original content for context
        sim_summary: Simulation summary for context
        model: AI model to use for evaluation
        request_timeout: Request timeout
        config: Configuration object
        pdf_path: Not used (kept for compatibility)
        project_dir: Project directory to read local files
    
    Returns:
        LLM response indicating which candidate is best and why
    """
    # Collect lightweight attachments for evaluation context
    attachments_note = ""
    try:
        if project_dir is not None:
            # Include main paper file
            paper_file = project_dir / "paper.tex"
            if paper_file.exists():
                paper_text = paper_file.read_text(encoding="utf-8", errors="ignore")
                attach_preview = (paper_text[:1500] + ("\n...[CONTENT TRUNCATED]...\n" if len(paper_text) > 2200 else "") + paper_text[-700:]) if len(paper_text) > 2200 else paper_text
                attachments_note += f"\nATTACHMENT: paper.tex (preview)\n{attach_preview}\n"
            
            # Include simulation code and other project files (similar to above)
            for filename in ['simulation.py', 'simulation_output.txt', 'results_summary.csv', 'refs.bib']:
                file_path = project_dir / filename
                if file_path.exists():
                    try:
                        file_text = file_path.read_text(encoding="utf-8", errors="ignore")
                        preview_text = file_text[:1200] + ("\n...[CONTENT TRUNCATED]..." if len(file_text) > 1200 else "")
                        attachments_note += f"\nATTACHMENT: {filename} (preview)\n{preview_text}\n"
                    except Exception:
                        pass
                        
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
        from ..ai.chat import _universal_chat
        response = _universal_chat(
            evaluation_prompt, 
            model=model, 
            request_timeout=request_timeout, 
            prompt_type="candidate_evaluation",
            fallback_models=config.fallback_models,
            pdf_path=None
        )
        return response
    except Exception as e:
        print(f"ERROR: LLM evaluation failed: {e}")
        return "SELECTED: 1\nREASONING: Evaluation failed, defaulting to first candidate."


def run_ideation_phase(topic: str, field: str, num_ideas: int, model: str, config) -> dict:
    """
    Run research ideation phase to generate and select research ideas.
    
    Args:
        topic: Research topic or area of interest
        field: Research field
        num_ideas: Number of ideas to generate
        model: AI model to use
        config: Workflow configuration
        
    Returns:
        Selected research idea dictionary with 'title' and 'question' keys
    """
    from ai.chat import _universal_chat
    from prompts.ideation_templates import _ideation_prompt
    
    print(f"🧠 Generating {num_ideas} research ideas for '{topic}' in {field}...")
    
    # Generate research ideas
    ideation_prompt = _ideation_prompt(topic, field, num_ideas)
    
    try:
        ideas_response = _universal_chat(
            prompt=ideation_prompt,
            model=model,
            timeout=config.request_timeout,
            max_retries=config.max_retries
        )
        
        # Parse ideas (simplified - assuming LLM returns structured format)
        # In real implementation, would parse JSON or structured text
        ideas = []
        lines = ideas_response.split('\n')
        current_idea = {}
        
        for line in lines:
            if line.startswith('Title:') or line.startswith('**Title'):
                if current_idea:
                    ideas.append(current_idea)
                current_idea = {'title': line.split(':', 1)[1].strip().replace('*', '')}
            elif line.startswith('Question:') or line.startswith('**Question'):
                current_idea['question'] = line.split(':', 1)[1].strip().replace('*', '')
        
        if current_idea:
            ideas.append(current_idea)
        
        if not ideas:
            # Fallback if parsing fails
            return {
                'title': f"Investigation of {topic} in {field}",
                'question': f"How can we improve understanding of {topic}?"
            }
        
        # For now, return the first idea. In real implementation, 
        # would present options to user or use LLM to select best
        selected_idea = ideas[0] if ideas else {
            'title': f"Investigation of {topic} in {field}",
            'question': f"How can we improve understanding of {topic}?"
        }
        
        return selected_idea
        
    except Exception as e:
        print(f"⚠️ Ideation phase failed, using fallback: {e}")
        return {
            'title': f"Investigation of {topic} in {field}",
            'question': f"How can we improve understanding of {topic}?"
        }
