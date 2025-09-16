#!/usr/bin/env python3
"""
Prompt templates for various workflow steps.
"""
from __future__ import annotations
from pathlib import Path
from typing import Optional, Dict, List


def _initial_draft_prompt(topic: str, field: str, question: str, user_prompt: Optional[str] = None) -> List[Dict[str, str]]:
    """Generate prompt for initial draft creation."""
    sys_prompt = (
        "You are a world-class researcher and academic writer. Create a comprehensive research paper "
        "with the following strict requirements:\n\n"
        "🔒 CRITICAL STRUCTURAL REQUIREMENTS:\n"
        "- Paper must be complete, publication-ready LaTeX\n"
        "- Title and author block must follow academic standards: use a main title and optional subtitle, break long titles for readability, and format author(s) and affiliations with clear separation and footnotes if needed.\n"
        "- Abstract must use short, clear sentences and bullet points for contributions or key results. Avoid dense, technical language and long clauses.\n"
        "- Include ALL necessary sections: abstract, introduction, methodology, results, discussion, conclusion\n"
        "- Embed 15-20 authentic, real references using \\begin{thebibliography} (NO separate .bib files)\n"
        "- Include at least 2-3 figures and 1-2 tables with proper captions and labels. Table and figure captions must be concise, clearly separated from the main text, and use smaller font if possible.\n"
        "- Use proper LaTeX document structure with \\documentclass{article}\n"
    "- All display equations must be left-aligned and, if too long to fit within the column or page width, must be split into multiple lines using the align, align*, multline, or split environments, with explicit line breaks (\\) at logical points (e.g., after +, -, =, etc.). If a formula is still too wide, use \\resizebox or similar advanced LaTeX techniques to ensure it fits. Never allow a single equation line to overflow the text width. All variables and symbols must be italicized in math mode. For complex formulas, always prefer breaking into multiple lines or using advanced environments over shrinking font size.\n"
        "  Example:\n"
        "  \\begin{align*}\n"
        "    a &= b + c + d + e + f + g + h \\\\ \n"
        "      &\\quad + i + j + k\n"
        "  \\end{align*}\n"
        "- Use algorithm environments (\\begin{algorithm}, \\begin{algorithmic}, or algorithm2e) for all pseudocode. Indent steps properly, use monospaced font, and ensure step numbers are aligned and readable.\n"
        "- Ensure all figures/tables/algorithms are referenced in text using \\ref{}.\n"
        "- Section and subsection headings must use bold and extra vertical spacing for clear hierarchy.\n"
        "- Glossary and symbols section, if present, should be formatted as a table or placed in an appendix, not directly after the abstract.\n"
        "- Appendices must be clearly separated, each starting on a new page with a clear section header, and placed after the references.\n"
        "- Add extra whitespace between sections, tables, figures, and algorithms. Avoid dense paragraphs; use shorter sentences and more line breaks for readability.\n"
        "- NO CODE BLOCKS: Use algorithm environments for pseudocode only.\n\n"
        "📊 CONTENT REQUIREMENTS:\n"
        "- Present original research with novel contributions\n"
        "- Include comprehensive literature review\n"
        "- Provide detailed methodology section\n"
        "- Present results with proper analysis\n"
        "- Include limitations and future work\n"
        "- Write 6000-8000 words (15-20 pages)\n"
        "- Use academic writing style appropriate for top-tier journals\n\n"
        "🔬 SIMULATION INTEGRATION:\n"
        "- Embed a complete Python simulation in LaTeX using \\begin{filecontents*}{simulation.py}\n"
        "- Simulation should generate realistic data for figures/tables\n"
        "- Include data analysis and visualization code\n"
        "- Ensure simulation runs independently and produces meaningful results\n"
        "- Reference simulation results in the paper text\n\n"
        "📚 REFERENCE REQUIREMENTS:\n"
        "- Use ONLY real, authentic references from reputable journals/conferences\n"
        "- Include recent papers (last 5 years) and seminal works\n"
        "- Format using \\begin{thebibliography}{99} environment\n"
        "- Cite appropriately throughout the paper using \\cite{}\n"
        "- Format all references consistently, including full journal/conference names, page numbers, and DOIs where available.\n"
        "- NO FAKE or PLACEHOLDER references\n\n"
    )
    
    # Add user prompt if provided
    if user_prompt:
        sys_prompt = (
            f"PRIORITY INSTRUCTION FROM USER: {user_prompt}\n\n"
            "The above user instruction takes precedence when generating the paper content. "
            "However, still maintain all technical LaTeX requirements.\n\n"
            + sys_prompt
        )
    
    user_prompt_text = (
        f"Create a comprehensive research paper on the following topic:\n\n"
        f"**Topic:** {topic}\n"
        f"**Field:** {field}\n"
        f"**Research Question:** {question}\n\n"
        
        "Requirements:\n"
        "1. Generate a complete, self-contained LaTeX document\n"
        "2. Include embedded simulation code that generates data for your figures/tables\n"
        "3. Use only real, authentic references - no fake citations\n"
        "4. Ensure the paper makes a novel contribution to the field\n"
        "5. Follow academic writing conventions for the specified field\n"
        "6. Create publication-ready content suitable for a top-tier journal\n\n"
        
        "Provide the complete LaTeX document ready for compilation."
    )
    
    return [
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": user_prompt_text}
    ]


def _review_prompt(paper_tex: str, sim_summary: str, project_dir: Path = None, user_prompt: Optional[str] = None) -> List[Dict[str, str]]:
    """Generate prompt for paper review."""
    
    sys_prompt = (
        "You are a senior academic reviewer for a top-tier journal in the relevant field. "
        "Provide a thorough, constructive review of the submitted paper.\n\n"
        
        "REVIEW CRITERIA:\n"
        "- Scientific rigor and methodology soundness\n"
        "- Novelty and significance of contributions\n"
        "- Literature review completeness and accuracy\n"
        "- Results interpretation and discussion quality\n"
        "- Writing clarity and organization\n"
        "- Reproducibility and technical correctness\n"
    "- Figures and tables quality and relevance\n"
    "- Equation formatting: All display equations must fit within the text width. Require breaking long or complex formulas into multiple lines using align, multline, split, or resizebox as needed. Never allow overflow.\n"
        "- References authenticity and appropriateness\n\n"
        
        "REVIEW STRUCTURE:\n"
        "Provide your review in the following format:\n"
        "- Summary of the paper's contributions\n"
        "- Major strengths\n"
        "- Major weaknesses and concerns\n"
        "- Minor issues and suggestions\n"
        "- Recommendation (accept/minor revision/major revision/reject)\n"
        "- Specific actionable feedback for improvement\n\n"
        
        "Be constructive but thorough in identifying areas for improvement."
    )
    
    # Add user prompt if provided
    if user_prompt:
        sys_prompt = (
            f"PRIORITY INSTRUCTION FROM USER: {user_prompt}\n\n"
            "The above user instruction should guide your review focus. "
            "However, still provide a comprehensive academic review.\n\n"
            + sys_prompt
        )
    
    # Collect project context
    project_context = ""
    if project_dir and project_dir.exists():
        from core.config import _collect_project_files
        project_context = _collect_project_files(project_dir)
    
    user_content = (
        "Please review the following research paper:\n\n"
        "----- PAPER (LaTeX) -----\n" + paper_tex + "\n"
        "----- SIMULATION RESULTS -----\n" + sim_summary + "\n"
    )
    
    if project_context:
        user_content += "----- PROJECT CONTEXT -----\n" + project_context + "\n"
    
    user_content += (
        "\nProvide a comprehensive review focusing on scientific quality, "
        "technical correctness, and potential for publication in a top-tier venue."
    )
    
    return [
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": user_content}
    ]


def _revise_prompt(paper_tex: str, sim_summary: str, review_text: str, latex_errors: str = "", 
                  project_dir: Path = None, user_prompt: Optional[str] = None, 
                  quality_issues: Optional[List[str]] = None) -> List[Dict[str, str]]:
    """Generate prompt for paper revision."""
    
    sys_prompt = (
        "You are an expert academic writer and researcher. Your task is to revise the paper "
        "based on the provided review feedback and address all technical issues.\n\n"
        "🔒 CRITICAL REVISION REQUIREMENTS:\n"
        "- Address ALL reviewer feedback completely\n"
        "- Fix any LaTeX compilation errors\n"
        "- Maintain or improve paper quality\n"
        "- Preserve the paper's core contributions\n"
        "- Use only authentic references\n"
        "- Ensure proper LaTeX structure\n"
        "- Title and author block must follow academic standards: use a main title and optional subtitle, break long titles for readability, and format author(s) and affiliations with clear separation and footnotes if needed.\n"
        "- Abstract must use short, clear sentences and bullet points for contributions or key results. Avoid dense, technical language and long clauses.\n"
    "- All display equations must be left-aligned and, if too long to fit within the column or page width, must be split into multiple lines using the align, align*, multline, or split environments, with explicit line breaks (\\) at logical points (e.g., after +, -, =, etc.). If a formula is still too wide, use \\resizebox or similar advanced LaTeX techniques to ensure it fits. Never allow a single equation line to overflow the text width. All variables and symbols must be italicized in math mode. For complex formulas, always prefer breaking into multiple lines or using advanced environments over shrinking font size.\n"
        "  Example:\n"
        "  \\begin{align*}\n"
        "    a &= b + c + d + e + f + g + h \\\\ \n"
        "      &\\quad + i + j + k\n"
        "  \\end{align*}\n"
        "- Use algorithm environments (\\begin{algorithm}, \\begin{algorithmic}, or algorithm2e) for all pseudocode. Indent steps properly, use monospaced font, and ensure step numbers are aligned and readable.\n"
        "- Include at least 2-3 figures and 1-2 tables with proper captions and labels. Table and figure captions must be concise, clearly separated from the main text, and use smaller font if possible.\n"
        "- Ensure all figures/tables/algorithms are referenced in text using \\ref{}.\n"
        "- Section and subsection headings must use bold and extra vertical spacing for clear hierarchy.\n"
        "- Glossary and symbols section, if present, should be formatted as a table or placed in an appendix, not directly after the abstract.\n"
        "- Appendices must be clearly separated, each starting on a new page with a clear section header, and placed after the references.\n"
        "- Add extra whitespace between sections, tables, figures, and algorithms. Avoid dense paragraphs; use shorter sentences and more line breaks for readability.\n"
        "- Format all references consistently, including full journal/conference names, page numbers, and DOIs where available.\n"
        "- Update simulation if needed to support revisions\n\n"
        "📝 REVISION APPROACH:\n"
        "- Make substantial improvements, not cosmetic changes\n"
        "- Add content rather than remove existing material\n"
        "- Enhance clarity and technical accuracy\n"
        "- Strengthen methodology and results sections\n"
        "- Improve figures and tables as needed\n"
        "- Ensure all references are real and appropriate\n\n"
        "OUTPUT FORMAT:\n"
        "Provide the complete revised LaTeX document. Do not use diff format - "
        "provide the full, revised paper ready for compilation."
    )
    
    # Add user prompt if provided
    if user_prompt:
        sys_prompt = (
            f"PRIORITY INSTRUCTION FROM USER: {user_prompt}\n\n"
            "The above user instruction takes precedence when revising the paper. "
            "However, still address all review feedback and technical issues.\n\n"
            + sys_prompt
        )
    
    # Collect project context
    project_context = ""
    if project_dir and project_dir.exists():
        from core.config import _collect_project_files
        project_context = _collect_project_files(project_dir)
    
    user_content = (
        "Please revise the following paper based on the review feedback:\n\n"
        "----- CURRENT PAPER (LaTeX) -----\n" + paper_tex + "\n"
        "----- REVIEW FEEDBACK -----\n" + review_text + "\n"
        "----- SIMULATION RESULTS -----\n" + sim_summary + "\n"
    )
    
    # Add quality issues if detected
    if quality_issues:
        user_content += (
            "\n----- DETECTED QUALITY ISSUES -----\n"
            "The following specific quality issues have been automatically detected and MUST be addressed:\n\n"
        )
        for issue in quality_issues:
            user_content += f"• {issue}\n"
        user_content += (
            "\n----- END QUALITY ISSUES -----\n"
            "CRITICAL: Your revision MUST specifically address ALL of the above quality issues."
        )
    
    # Add LaTeX compilation errors if any
    if latex_errors:
        user_content += (
            "\n----- LATEX COMPILATION ERRORS -----\n" + latex_errors + 
            "\n----- END LATEX ERRORS -----\n"
            "CRITICAL: Fix ALL LaTeX compilation errors in your revision."
        )
    
    if project_context:
        user_content += "\n----- PROJECT CONTEXT -----\n" + project_context + "\n"
    
    user_content += (
        "\nProvide the complete revised LaTeX document addressing all feedback, "
        "quality issues, and technical problems."
    )
    
    return [
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": user_content}
    ]


def _editor_prompt(review_text: str, iteration_count: int, user_prompt: Optional[str] = None) -> List[Dict[str, str]]:
    """Generate prompt for editorial improvements."""
    
    sys_prompt = (
        "You are a senior academic editor with expertise in improving research papers for "
        "top-tier publication. Your role is to enhance the paper's clarity, structure, "
        "and overall quality based on review feedback.\n\n"
        
        "EDITORIAL FOCUS:\n"
        "- Improve writing clarity and flow\n"
        "- Enhance paper structure and organization\n"
        "- Strengthen arguments and presentation\n"
        "- Ensure consistent terminology and style\n"
        "- Improve figure and table presentation\n"
        "- Enhance readability for target audience\n\n"
        
        "QUALITY STANDARDS:\n"
        "- Publication-ready prose\n"
        "- Logical flow and coherent narrative\n"
        "- Clear and compelling presentation\n"
        "- Appropriate academic tone\n"
        "- Effective use of visuals\n"
        "- Strong conclusions and implications\n\n"
    )
    
    # Add user prompt if provided
    if user_prompt:
        sys_prompt = (
            f"PRIORITY INSTRUCTION FROM USER: {user_prompt}\n\n"
            "The above user instruction should guide your editorial improvements. "
            "However, maintain high academic standards throughout.\n\n"
            + sys_prompt
        )
    
    user_content = (
        f"This is editorial iteration {iteration_count}. Please provide editorial "
        "improvements based on the following review feedback:\n\n"
        "----- REVIEW FEEDBACK -----\n" + review_text + "\n"
        "----- END REVIEW FEEDBACK -----\n\n"
        "Focus on enhancing clarity, structure, and overall presentation quality "
        "to meet the standards of top-tier academic journals."
    )
    
    return [
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": user_content}
    ]

def _combined_review_edit_revise_prompt(paper_tex: str, sim_summary: str, latex_errors: str = "", project_dir: Path = None, user_prompt: Optional[str] = None, iteration_count: int = 1, quality_issues: Optional[List[str]] = None) -> List[Dict[str, str]]:
    "- All display equations must fit within the text width. For long or complex formulas, break into multiple lines using align, multline, split, or use \\resizebox as needed. Never allow overflow.\n"
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
        "- Reproducible results documentation\n\n"
        
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
        "- Include all necessary files in diffs (paper.tex, simulation.py, etc.)\n\n"
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
        from ..core.config import _collect_project_files
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
