#!/usr/bin/env python3
"""
Test-time compute scaling and advanced generation functions for research papers.
"""
from __future__ import annotations
import hashlib
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional


def test_time_compute_scaling(
    base_content: str,
    model: str,
    request_timeout: int,
    config: Any,
    compute_multiplier: float = 2.0,
    candidate_count: int = 3
) -> str:
    """
    Implement test-time compute scaling by generating multiple candidates and selecting the best.
    
    Args:
        base_content: Base paper content to improve
        model: AI model to use
        request_timeout: Request timeout
        config: Configuration object
        compute_multiplier: Multiplier for compute resources
        candidate_count: Number of candidates to generate
    
    Returns:
        Best candidate after evaluation
    """
    print(f"Running test-time compute scaling with {candidate_count} candidates...")
    
    candidates = []
    generation_times = []
    
    for i in range(candidate_count):
        print(f"  Generating candidate {i + 1}/{candidate_count}...")
        start_time = time.time()
        
        try:
            # Generate varied candidate
            candidate = _generate_scaled_candidate(base_content, model, request_timeout, config, i)
            generation_time = time.time() - start_time
            
            candidates.append(candidate)
            generation_times.append(generation_time)
            
            print(f"    Candidate {i + 1}: {generation_time:.2f}s, {len(candidate)} chars")
            
        except Exception as e:
            print(f"    Candidate {i + 1} failed: {e}")
            candidates.append("")
            generation_times.append(None)
    
    # Filter valid candidates
    valid_candidates = [(i, c) for i, c in enumerate(candidates) if c.strip()]
    
    if not valid_candidates:
        print("  All candidates failed, returning original content")
        return base_content
    
    print(f"  Evaluating {len(valid_candidates)} valid candidates...")
    
    # Select best candidate using quality metrics
    best_candidate = _select_best_scaled_candidate(valid_candidates, base_content)
    
    total_time = sum(t for t in generation_times if t)
    print(f"  Total compute time: {total_time:.1f}s")
    
    return best_candidate


def _generate_scaled_candidate(base_content: str, model: str, request_timeout: int, config: Any, variant_id: int) -> str:
    """Generate a scaled candidate with variations."""
    from ..ai.chat import _universal_chat
    
    # Create variation prompts for different approaches
    variation_prompts = [
        "Focus on technical depth and mathematical rigor. Add detailed proofs and formal analysis.",
        "Emphasize practical applications and real-world implementation details.",
        "Strengthen experimental validation and comprehensive evaluation metrics.",
        "Expand theoretical foundations and related work connections.",
        "Improve clarity, structure, and presentation quality."
    ]
    
    variation = variation_prompts[variant_id % len(variation_prompts)]
    
    messages = [
        {
            "role": "system",
            "content": f"""You are an expert academic writer improving a research paper. 
            
Your task is to enhance the provided paper content with the following focus:
{variation}

Requirements:
- Maintain the original structure and core contributions
- Make substantial improvements while preserving accuracy
- Ensure all additions are technically sound and well-integrated
- Keep the LaTeX formatting and mathematical expressions correct"""
        },
        {
            "role": "user",
            "content": f"""Please improve the following research paper content:

{base_content[:3000]}...

Focus on: {variation}

Generate an enhanced version that maintains the original structure but includes significant improvements in the specified area."""
        }
    ]
    
    response = _universal_chat(
        messages,
        model=model,
        request_timeout=request_timeout,
        prompt_type="scaling",
        fallback_models=config.fallback_models
    )
    
    return response


def _select_best_scaled_candidate(candidates: List[tuple], original_content: str) -> str:
    """Select the best candidate based on quality metrics."""
    best_score = 0
    best_candidate = candidates[0][1]  # Default to first candidate
    
    for idx, candidate in candidates:
        score = _evaluate_candidate_quality(candidate, original_content)
        print(f"    Candidate {idx + 1} score: {score:.3f}")
        
        if score > best_score:
            best_score = score
            best_candidate = candidate
    
    print(f"  Selected best candidate with score: {best_score:.3f}")
    return best_candidate


def _evaluate_candidate_quality(candidate: str, original: str) -> float:
    """Evaluate the quality of a candidate."""
    score = 0.0
    
    # Length improvement (reasonable increase is good)
    length_ratio = len(candidate) / max(len(original), 1)
    if 1.1 <= length_ratio <= 2.0:  # 10% to 100% increase is good
        score += 0.2
    elif length_ratio > 1.0:
        score += 0.1
    
    # Technical content indicators
    technical_terms = ['algorithm', 'methodology', 'evaluation', 'analysis', 
                      'implementation', 'optimization', 'performance', 'validation']
    tech_score = sum(1 for term in technical_terms if term.lower() in candidate.lower())
    score += min(tech_score / 20, 0.3)
    
    # LaTeX structure quality
    latex_indicators = ['\\section', '\\subsection', '\\cite{', '\\ref{', 
                       '\\label{', '\\begin{equation}', '\\begin{figure}']
    latex_score = sum(1 for indicator in latex_indicators if indicator in candidate)
    score += min(latex_score / 30, 0.2)
    
    # Mathematical content
    math_indicators = ['\\begin{equation}', '\\begin{align}', '$', '\\(', 
                      'theorem', 'proof', 'lemma']
    math_score = sum(1 for indicator in math_indicators if indicator.lower() in candidate.lower())
    score += min(math_score / 15, 0.15)
    
    # Reference quality
    ref_count = candidate.count('\\cite{')
    score += min(ref_count / 25, 0.15)
    
    return score


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
        output_diffs: Whether to output diffs
        pdf_path: Optional PDF path
    
    Returns:
        Best revision candidate based on quality metrics
    """
    print(f"  Generating {candidate_count} revision candidates...")
    
    candidates = []
    generation_times = []
    
    # Generate multiple revision candidates with aggressive variations
    for i in range(candidate_count):
        print(f"    Generating revision candidate {i + 1}/{candidate_count}...")
        
        start_time = time.time()
        
        try:
            # Create varied revision prompts to encourage diversity
            from ..prompts.templates import _revise_prompt
            base_prompt = _revise_prompt(current_tex, sim_summary, review_text, latex_errors, project_dir, user_prompt)
            
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
            from ..ai.chat import _universal_chat
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
                from ..generation.content import _save_candidate_diff
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
    
    from ..generation.content import _select_best_revision_candidate_with_llm
    best_candidate_response = _select_best_revision_candidate_with_llm(
        revision_candidates, current_tex, review_text, model, request_timeout, config, pdf_path=pdf_path, project_dir=project_dir
    )
    
    # DEBUG: Show full LLM evaluation response
    print(f"  DEBUG - LLM Revision Evaluation Response:")
    print(f"  {best_candidate_response}")
    print(f"  {''*80}")
    
    # Parse the LLM selection response
    try:
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
                    from ..generation.content import _save_candidate_diff
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


def _generate_best_initial_draft_candidate(
    topic: str,
    field: str,
    question: str,
    user_prompt: Optional[str],
    model: str,
    request_timeout: int,
    config: Any,
    candidate_count: int = 3,
    output_diffs: bool = False
) -> str:
    """
    Generate multiple initial draft candidates and select the best one.
    
    Args:
        topic: Research topic
        field: Research field
        question: Research question  
        user_prompt: Optional user instructions
        model: AI model to use
        request_timeout: Request timeout
        config: Configuration object
        candidate_count: Number of candidates to generate
        output_diffs: Whether to output diffs
    
    Returns:
        Best initial draft candidate
    """
    print(f"  Generating {candidate_count} initial draft candidates...")
    
    candidates = []
    generation_times = []
    
    for i in range(candidate_count):
        print(f"    Generating initial draft candidate {i + 1}/{candidate_count}...")
        
        start_time = time.time()
        
        try:
            # Create varied prompts for different approaches
            from ..prompts.templates import _initial_draft_prompt
            base_prompt = _initial_draft_prompt(topic, field, question, user_prompt)
            
            if i > 0:
                # Add variation instructions
                variations = [
                    "\n\nFocus on theoretical foundations and mathematical formalism.",
                    "\n\nEmphasize practical applications and experimental validation.",
                    "\n\nPrioritize comprehensive literature review and positioning.",
                    "\n\nConcentrate on novel methodology and technical innovation.",
                    "\n\nStrengthen empirical evaluation and comparative analysis."
                ]
                
                variation = variations[(i - 1) % len(variations)]
                varied_prompt = base_prompt.copy()
                varied_prompt[0]["content"] += variation
            else:
                varied_prompt = base_prompt
            
            # Generate candidate
            from ..ai.chat import _universal_chat
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
    
    # Filter valid candidates
    valid_candidates = [(i, c) for i, c in enumerate(candidates) if c.strip()]
    
    if not valid_candidates:
        print("      All initial draft candidates failed")
        return ""
    
    print(f"  Selecting best candidate from {len(valid_candidates)} valid responses...")
    
    # Evaluate candidates using quality metrics
    best_score = 0
    best_candidate = valid_candidates[0][1]
    
    for idx, (orig_idx, candidate) in enumerate(valid_candidates):
        score = _evaluate_initial_draft_quality(candidate, topic, field, question)
        overall_score = score.get('overall_quality', 0)
        
        print(f"    Candidate {orig_idx + 1} score: {overall_score:.3f}")
        
        if overall_score > best_score:
            best_score = overall_score
            best_candidate = candidate
    
    total_time = sum(t for t in generation_times if t)
    print(f"    Total compute time: {total_time:.1f}s")
    print(f"    Selected best candidate with score: {best_score:.3f}")
    
    return best_candidate


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
    math_indicators = ['\\begin{equation}', '\\begin{align}', '$', '\\(', 
                      'theorem', 'proof', 'lemma']
    math_count = sum(1 for indicator in math_indicators if indicator.lower() in draft_text.lower())
    metrics['mathematical_content'] = min(math_count / 5, 1.0)
    
    # Simulation quality assessment
    from ..evaluation.quality import _evaluate_simulation_content
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
