"""
Document-specific prompt generators
Creates specialized prompts based on document type and field
"""

from typing import Dict, List, Optional
from document_types import DocumentType, DocumentTemplate, OutputFormat, get_document_template

class DocumentPromptGenerator:
    """Generates document-type-specific prompts"""
    
    @staticmethod
    def get_ideation_prompt(doc_type: DocumentType, topic: str, field: str, 
                           question: str, num_ideas: int = 15) -> str:
        """Generate document-type-specific ideation prompt"""
        template = get_document_template(doc_type)
        
        # Customize ideation based on document type
        if doc_type == DocumentType.FINANCE_RESEARCH:
            focus_areas = "quantitative analysis, risk assessment, market dynamics, regulatory considerations, profitability metrics"
            approaches = "econometric modeling, financial engineering, empirical studies, market analysis, portfolio theory"
        elif doc_type == DocumentType.ENGINEERING_PAPER:
            focus_areas = "technical feasibility, system design, performance optimization, safety analysis, cost-effectiveness"
            approaches = "simulation studies, prototype development, experimental validation, theoretical modeling, case studies"
        elif doc_type == DocumentType.SURVEY_PAPER:
            focus_areas = "comprehensive coverage, trend analysis, comparative evaluation, gap identification, future directions"
            approaches = "systematic literature review, meta-analysis, taxonomical classification, trend analysis, synthesis"
        elif doc_type == DocumentType.PRESENTATION_SLIDES:
            focus_areas = "clear communication, visual appeal, audience engagement, key takeaways, actionable insights"
            approaches = "narrative structure, visual demonstrations, case studies, interactive elements, compelling examples"
        else:  # Default to research paper
            focus_areas = "theoretical innovation, experimental validation, practical applications, comparative analysis, scalability"
            approaches = "theoretical, experimental, algorithmic, systems-based, empirical studies"
        
        return f"""You are a brilliant research strategist tasked with generating innovative research ideas for a {doc_type.value.replace('_', ' ')}.

TOPIC: {topic}
FIELD: {field}
RESEARCH QUESTION: {question}
DOCUMENT TYPE: {doc_type.value.replace('_', ' ').title()}

Your task is to generate {num_ideas} distinct, high-quality research ideas that address this topic/question in the field of {field}. Each idea should be appropriate for a {doc_type.value.replace('_', ' ')} format.

Focus on: {focus_areas}
Consider approaches: {approaches}

For each idea, provide:

1. **Title**: A concise, descriptive title appropriate for {doc_type.value.replace('_', ' ')}
2. **Core Concept**: 2-3 sentences describing the main research direction
3. **Originality Score**: 1-10 (10 = highly novel, never done before)
4. **Impact Score**: 1-10 (10 = revolutionary potential, broad applications)
5. **Feasibility Score**: 1-10 (10 = very feasible with current technology/methods)
6. **Pros**: 2-3 key advantages of this approach
7. **Cons**: 2-3 potential challenges or limitations

Ensure ideas span different sub-approaches within the focus areas above.

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
Rank the top 5 ideas by overall potential (considering originality, impact, feasibility), and explain your ranking criteria specific to {doc_type.value.replace('_', ' ')} requirements.

## RECOMMENDATION
Select the single best idea and explain why it's optimal for development into a {doc_type.value.replace('_', ' ')}.
"""
    
    @staticmethod
    def get_initial_draft_prompt(doc_type: DocumentType, topic: str, field: str, 
                                question: str, user_prompt: Optional[str] = None) -> List[Dict[str, str]]:
        """Generate document-type-specific initial draft prompt"""
        template = get_document_template(doc_type)
        
        base_prompt = DocumentPromptGenerator._get_base_requirements(template)
        structure_prompt = DocumentPromptGenerator._get_structure_requirements(template)
        content_prompt = DocumentPromptGenerator._get_content_requirements(template, field)
        format_prompt = DocumentPromptGenerator._get_format_requirements(template)
        
        sys_prompt = f"""You are a specialized academic writer creating a {template.doc_type.value.replace('_', ' ')} in the field of {field}.

{base_prompt}

{structure_prompt}

{content_prompt}

{format_prompt}

CRITICAL REQUIREMENTS:
1. DOCUMENT TYPE: This is a {template.doc_type.value.replace('_', ' ')} - follow the conventions for this document type
2. OUTPUT FORMAT: Use {template.latex_documentclass} document class with appropriate styling
3. FIELD-SPECIFIC: Adapt content and style to {field} conventions and expectations
4. TARGET AUDIENCE: Write for professionals in {field}
5. FOCUS: Emphasize {template.prompt_focus}

SPECIAL REQUIREMENTS FOR THIS DOCUMENT TYPE:
{chr(10).join(f"- {req}" for req in template.special_requirements)}
"""
        
        if user_prompt:
            sys_prompt += f"\n\nADDITIONAL USER REQUIREMENTS:\n{user_prompt}"

        if template.doc_type == DocumentType.RESEARCH_PAPER:
            sys_prompt += """

INNOVATION EXPECTATIONS:
- Craft an "Innovation Thesis" paragraph in the introduction linking the research gap to the proposed breakthroughs
- Include a labelled "Prior Art Differentiation" subsection comparing against at least three closest works
- Provide an "Innovation Hooks" checklist detailing ablations, stress tests, or design twists that evidence novelty
- Conclude with a forward-looking impact statement describing how the contributions enable future innovations
"""

        user_prompt_text = f"""Create a complete {template.doc_type.value.replace('_', ' ')} on the topic: "{topic}"

Research question/focus: {question}
Field: {field}

The document should be written as a professional {template.doc_type.value.replace('_', ' ')} suitable for {field} professionals, following all the requirements specified above."""

        if template.doc_type == DocumentType.RESEARCH_PAPER:
            user_prompt_text += """

Please ensure the draft:
- Summarises the key innovations in a numbered Contributions section with bullet points
- Adds a brief comparison table or paragraph that highlights differentiation from prior art
- References an Innovation Hooks checklist tying each novel element to evaluation evidence
- Ends with a Vision & Future Work paragraph that extrapolates the broader impact of the contributions
"""

        return [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": user_prompt_text}
        ]
    
    @staticmethod
    def get_review_prompt(doc_type: DocumentType, paper_tex: str, sim_summary: str, 
                         field: str, project_dir=None, user_prompt: Optional[str] = None) -> List[Dict[str, str]]:
        """Generate document-type-specific review prompt"""
        template = get_document_template(doc_type)
        
        review_criteria = DocumentPromptGenerator._get_review_criteria(template, field)
        structure_check = DocumentPromptGenerator._get_structure_check(template)
        content_check = DocumentPromptGenerator._get_content_check(template, field)
        
        sys_prompt = f"""Act as a senior editor/reviewer for {template.doc_type.value.replace('_', ' ')}s in {field}.

Your task is to review this {template.doc_type.value.replace('_', ' ')} according to professional standards in {field}.

DOCUMENT TYPE SPECIFIC CRITERIA:
{review_criteria}

STRUCTURE REQUIREMENTS:
{structure_check}

CONTENT REQUIREMENTS:
{content_check}

CRITICAL EVALUATION POINTS:
- Explicitly assess originality/innovation: identify the precise novel contributions, contrast them with 2–3 closest prior works, and judge incremental vs. substantive novelty
- Focus on {template.prompt_focus}
- Ensure content meets {field} professional standards
- Verify document serves its intended purpose as a {template.doc_type.value.replace('_', ' ')}
- Check that special requirements are met: {', '.join(template.special_requirements)}

Provide detailed feedback on strengths, weaknesses, and specific improvements needed."""

        if user_prompt:
            sys_prompt += f"\n\nADDITIONAL REVIEW CRITERIA:\n{user_prompt}"

        user_content = f"""DOCUMENT TYPE: {template.doc_type.value.replace('_', ' ')}
FIELD: {field}

CURRENT DOCUMENT (LATEX):
{paper_tex}

SIMULATION SUMMARY:
{sim_summary}

Please provide a thorough review focusing on the criteria specified for this document type."""

        return [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": user_content}
        ]
    
    @staticmethod
    def get_revision_prompt(doc_type: DocumentType, paper_tex: str, review_text: str, 
                           field: str, latex_errors: str = "", project_dir=None, 
                           user_prompt: Optional[str] = None) -> List[Dict[str, str]]:
        """Generate document-type-specific revision prompt"""
        template = get_document_template(doc_type)
        
        revision_guidelines = DocumentPromptGenerator._get_revision_guidelines(template, field)
        
        sys_prompt = f"""You are revising a {template.doc_type.value.replace('_', ' ')} based on expert feedback.

DOCUMENT TYPE: {template.doc_type.value.replace('_', ' ')}
FIELD: {field}
FOCUS: {template.prompt_focus}

REVISION GUIDELINES:
{revision_guidelines}

DOCUMENT-SPECIFIC REQUIREMENTS:
- Maintain the conventions of {template.doc_type.value.replace('_', ' ')}s in {field}
- Ensure content serves the document's intended purpose
- Follow the structure appropriate for this document type
- Address all reviewer concerns while preserving document quality
- Strengthen the articulation of novelty and positioning vs. prior work; add a brief "Novelty and Positioning" paragraph in the Introduction (or Related Work) if missing, explicitly stating what is new and how it differs from 2–3 closest works

OUTPUT REQUIREMENTS:
- Provide complete revised document in LaTeX format
- Use {template.latex_documentclass} document class
- Include all required packages: {', '.join(template.required_packages)}
- Ensure document compiles successfully"""

        if user_prompt:
            sys_prompt += f"\n\nADDITIONAL REVISION REQUIREMENTS:\n{user_prompt}"

        user_content = f"""ORIGINAL DOCUMENT (LATEX):
{paper_tex}

REVIEWER FEEDBACK:
{review_text}

LATEX ERRORS (if any):
{latex_errors}

Please provide the complete revised document addressing all feedback while maintaining the quality and conventions of a {template.doc_type.value.replace('_', ' ')} in {field}."""

        return [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": user_content}
        ]
    
    @staticmethod
    def _get_base_requirements(template: DocumentTemplate) -> str:
        """Get base requirements for document type"""
        reqs = [
            f"SINGLE FILE ONLY: Create ONE {template.output_format.value} file",
            "EMBEDDED REFERENCES: Include all references directly in the document",
            "COMPILABLE: Document must compile successfully",
            "NO FILENAMES: Don't reference specific files in the text"
        ]
        
        if template.requires_simulation:
            reqs.append("SIMULATION CODE: Include comprehensive simulation.py with real results")
        
        if template.max_pages:
            reqs.append(f"LENGTH: Target {template.max_pages} pages maximum")
            
        return "\n".join(f"- {req}" for req in reqs)
    
    @staticmethod
    def _get_structure_requirements(template: DocumentTemplate) -> str:
        """Get structure requirements for document type"""
        structure = f"""DOCUMENT STRUCTURE ({template.doc_type.value.replace('_', ' ')}):

REQUIRED SECTIONS:
{chr(10).join(f"- {section}" for section in template.typical_sections)}

OPTIONAL SECTIONS (include if relevant):
{chr(10).join(f"- {section}" for section in template.optional_sections)}"""

        if template.doc_type == DocumentType.PRESENTATION_SLIDES:
            structure += """

SLIDE STRUCTURE:
- Keep slides concise with bullet points
- Use visual hierarchy
- Include speaker notes where appropriate
- Limit text per slide (6x6 rule: max 6 bullets, 6 words each)"""

        return structure
    
    @staticmethod
    def _get_content_requirements(template: DocumentTemplate, field: str) -> str:
        """Get content requirements for document type"""
        content_reqs = []
        
        if template.requires_figures:
            content_reqs.append("FIGURES: Include relevant diagrams, charts, or plots using TikZ/pgfplots")
        
        if template.requires_tables:
            content_reqs.append("TABLES: Include data tables using booktabs package")
            
        if template.requires_algorithms:
            content_reqs.append("ALGORITHMS: Include detailed algorithms using algorithm2e package")

        if template.requires_financial_data:
            content_reqs.append("FINANCIAL DATA: Include real market data, financial models, and quantitative analysis")

        if template.doc_type == DocumentType.RESEARCH_PAPER:
            content_reqs.extend([
                "CONTRIBUTIONS: Provide a dedicated section with a concise bullet list of 3-4 novel contributions",
                "DIFFERENTIATION: Add a \"Prior Art Differentiation\" subsection contrasting against at least three closest works",
                "INNOVATION ROADMAP: Describe experimental or theoretical innovation hooks that go beyond standard baselines",
                "IMPACT ANALYSIS: Explain why the proposed innovations matter for the broader research community",
            ])

        if template.doc_type == DocumentType.EQUITY_RESEARCH:
            content_reqs.extend([
                "VALUATION MODELS: Include DCF, comparable company analysis",
                "PRICE TARGET: Provide specific price target with justification",
                "INVESTMENT RECOMMENDATION: Clear buy/hold/sell recommendation"
            ])
        elif template.doc_type == DocumentType.FINANCE_RESEARCH:
            content_reqs.extend([
                "MARKET ANALYSIS: Include market trends and dynamics",
                "RISK ASSESSMENT: Quantify and analyze risks",
                "FINANCIAL MODELS: Include relevant financial models and calculations"
            ])
        elif template.doc_type == DocumentType.PRESENTATION_SLIDES:
            content_reqs.extend([
                "VISUAL FOCUS: Emphasize visual elements over text",
                "KEY MESSAGES: Highlight main takeaways clearly",
                "ACTIONABLE INSIGHTS: Provide clear next steps or conclusions"
            ])
            
        return "\n".join(f"- {req}" for req in content_reqs) if content_reqs else "- Content should be appropriate for the document type and field"
    
    @staticmethod
    def _get_format_requirements(template: DocumentTemplate) -> str:
        """Get formatting requirements for document type"""
        format_reqs = [
            f"DOCUMENT CLASS: Use \\documentclass{{{template.latex_documentclass}}}",
            f"REQUIRED PACKAGES: Include {', '.join(template.required_packages)}",
            f"CITATION STYLE: Use {template.citation_style} citation format"
        ]
        
        if template.output_format == OutputFormat.BEAMER_SLIDES:
            format_reqs.extend([
                "BEAMER THEME: Use professional theme (default or Madrid)",
                "SLIDE TRANSITIONS: Keep simple and professional",
                "FRAME TITLES: Each slide should have clear title"
            ])
            
        return "\n".join(f"- {req}" for req in format_reqs)
    
    @staticmethod
    def _get_review_criteria(template: DocumentTemplate, field: str) -> str:
        """Get review criteria for document type"""
        if template.doc_type == DocumentType.EQUITY_RESEARCH:
            return """- Investment thesis clarity and support
- Valuation methodology appropriateness
- Financial model accuracy
- Risk assessment comprehensiveness
- Price target justification
- Recommendation strength"""
        elif template.doc_type == DocumentType.FINANCE_RESEARCH:
            return """- Market analysis depth
- Financial model robustness
- Risk quantification accuracy
- Regulatory compliance consideration
- Data sources credibility
- Practical applicability"""
        elif template.doc_type == DocumentType.PRESENTATION_SLIDES:
            return """- Message clarity and focus
- Visual design effectiveness
- Logical flow between slides
- Audience engagement potential
- Time appropriateness
- Actionability of conclusions"""
        elif template.doc_type == DocumentType.ENGINEERING_PAPER:
            return """- Technical implementation quality
- Performance analysis rigor
- Algorithm efficiency and correctness
- System design soundness
- Experimental methodology
- Reproducibility of results"""
        else:
            return f"""- Originality/innovation of contributions and clear differentiation from prior work
- Scientific rigor and methodology
- Literature review completeness
- Results interpretation accuracy
- Conclusions support by evidence
- {field} field-specific standards
- Professional presentation quality"""
    
    @staticmethod
    def _get_structure_check(template: DocumentTemplate) -> str:
        """Get structure checking requirements"""
        return f"""Check that the document includes all required sections:
{chr(10).join(f"✓ {section}" for section in template.typical_sections)}

Verify optional sections are included where relevant:
{chr(10).join(f"? {section}" for section in template.optional_sections)}

Ensure sections are well-balanced and logically ordered."""
    
    @staticmethod
    def _get_content_check(template: DocumentTemplate, field: str) -> str:
        """Get content checking requirements"""
        checks = [f"Content appropriate for {field} professionals"]
        
        if template.requires_simulation:
            checks.append("Simulation results are integrated and discussed")
        if template.requires_figures:
            checks.append("Figures are clear, relevant, and well-integrated")
        if template.requires_tables:
            checks.append("Tables are informative and properly formatted")
        if template.requires_algorithms:
            checks.append("Algorithms are detailed and implementable")
        if template.requires_financial_data:
            checks.append("Financial data is current, accurate, and well-analyzed")
            
        return "\n".join(f"- {check}" for check in checks)
    
    @staticmethod
    def _get_revision_guidelines(template: DocumentTemplate, field: str) -> str:
        """Get revision guidelines for document type"""
        guidelines = [
            f"Maintain {template.doc_type.value.replace('_', ' ')} conventions",
            f"Ensure content serves document purpose in {field}",
            "Address reviewer feedback comprehensively",
            "Preserve document quality and professionalism"
        ]
        
        if template.doc_type == DocumentType.PRESENTATION_SLIDES:
            guidelines.extend([
                "Keep slides concise and visually appealing",
                "Ensure logical flow between slides",
                "Maintain consistent formatting"
            ])
        elif template.requires_financial_data:
            guidelines.extend([
                "Update financial data if needed",
                "Verify calculations and models",
                "Ensure regulatory compliance"
            ])
            
        return "\n".join(f"- {guideline}" for guideline in guidelines)
