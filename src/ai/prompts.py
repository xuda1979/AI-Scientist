"""
Prompt templates for SciResearch Workflow.
"""
from __future__ import annotations
from typing import Optional


class PromptTemplates:
    """Centralized prompt template management."""
    
    @staticmethod
    def initial_draft(topic: str, field: str, question: str, user_prompt: Optional[str] = None) -> str:
        """Template for initial paper draft generation."""
        base_prompt = f"""
Generate a complete LaTeX research paper on the following topic:

Topic: {topic}
Field: {field}
Research Question: {question}

Requirements:
- Use standard LaTeX article format
- Include proper sections: abstract, introduction, related work, methodology, experiments, results, conclusion
- Add placeholder simulation code in appropriate sections
- Include relevant citations (embedded in document)
- Ensure academic rigor and clarity
- Target 8-12 pages when compiled

Focus on creating a substantial, publication-ready draft."""

        if user_prompt:
            base_prompt += f"\n\nAdditional Requirements: {user_prompt}"
        
        return base_prompt
    
    @staticmethod
    def combined_ideation_draft(
        topic: str, 
        field: str, 
        question: str, 
        num_ideas: int = 15,
        user_prompt: Optional[str] = None
    ) -> str:
        """Template for combined ideation and draft generation."""
        prompt = f"""
You are a brilliant research strategist and academic writer. Please generate innovative research ideas and create an initial draft paper.

TOPIC: {topic}
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

Focus on creating a substantial, publication-ready draft that integrates the best research direction."""
        return prompt
    
    @staticmethod
    def comprehensive_review_revision(
        current_tex: str,
        sim_summary: str,
        latex_errors: str,
        quality_issues: list,
        user_prompt: Optional[str],
        iteration: int,
        max_iterations: int
    ) -> str:
        """Template for comprehensive review and revision."""
        return f"""
You are an expert academic reviewer and editor. Please provide a comprehensive review and revision of this research paper.

**CURRENT PAPER (Iteration {iteration}/{max_iterations}):**
{current_tex}

**SIMULATION RESULTS:**
{sim_summary}

**LATEX COMPILATION STATUS:**
{latex_errors if latex_errors else "SUCCESS: Compilation successful"}

**QUALITY ISSUES DETECTED:**
{chr(10).join(f"- {issue}" for issue in quality_issues) if quality_issues else "SUCCESS: No major quality issues detected"}

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

Focus on significant improvements that advance toward publication quality."""
    
    @staticmethod
    def simulation_fixing(code: str, error: str, attempt: int) -> str:
        """Template for simulation code fixing."""
        return f"""
Fix the following Python simulation code that has errors:

**CODE:**
```python
{code}
```

**ERROR (Attempt {attempt}):**
{error}

**REQUIREMENTS:**
- Fix all syntax and runtime errors
- Maintain the original functionality
- Ensure code runs without issues
- Add proper error handling if needed
- Keep imports and dependencies minimal

Please provide the corrected code only:
```python
[Fixed code here]
```"""
    
    @staticmethod
    def research_ideation(
        topic: str,
        field: str, 
        question: str,
        num_ideas: int = 15
    ) -> str:
        """Template for research idea generation."""
        return f"""
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
Select the single best idea and explain why it's optimal for development into a research paper."""
