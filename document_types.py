"""
Document Type Configuration System
Defines different document types with their specific requirements, formats, and prompts.
"""

from enum import Enum
from dataclasses import dataclass
from typing import List, Dict, Optional, Any
from pathlib import Path

class DocumentType(Enum):
    """Supported document types"""
    RESEARCH_PAPER = "research_paper"
    ENGINEERING_PAPER = "engineering_paper"
    FINANCE_RESEARCH = "finance_research"
    EQUITY_RESEARCH = "equity_research"
    SURVEY_PAPER = "survey_paper"
    PRESENTATION_SLIDES = "presentation_slides"
    TECHNICAL_REPORT = "technical_report"
    WHITE_PAPER = "white_paper"
    CONFERENCE_PAPER = "conference_paper"
    JOURNAL_ARTICLE = "journal_article"

class OutputFormat(Enum):
    """Supported output formats"""
    LATEX_PDF = "latex_pdf"
    BEAMER_SLIDES = "beamer_slides"
    MARKDOWN_SLIDES = "markdown_slides"
    HTML_PRESENTATION = "html_presentation"
    TECHNICAL_MEMO = "technical_memo"

@dataclass
class DocumentTemplate:
    """Template configuration for a document type"""
    doc_type: DocumentType
    output_format: OutputFormat
    latex_documentclass: str
    required_packages: List[str]
    typical_sections: List[str]
    optional_sections: List[str]
    requires_simulation: bool
    requires_figures: bool
    requires_tables: bool
    requires_algorithms: bool
    requires_financial_data: bool
    max_pages: Optional[int]
    citation_style: str
    prompt_focus: str
    special_requirements: List[str]
    requires_embedded_references: bool = True
    min_references: int = 15
    enforce_strict_sections: bool = True

# Document Templates Configuration
DOCUMENT_TEMPLATES = {
    DocumentType.RESEARCH_PAPER: DocumentTemplate(
        doc_type=DocumentType.RESEARCH_PAPER,
        output_format=OutputFormat.LATEX_PDF,
        latex_documentclass="article",
        required_packages=["amsmath", "amsfonts", "graphicx", "cite"],
        typical_sections=["Abstract", "Introduction", "Related Work", "Methodology", 
                         "Results", "Discussion", "Conclusion"],
        optional_sections=["Appendix", "Acknowledgments"],
        requires_simulation=True,
        requires_figures=True,
        requires_tables=True,
        requires_algorithms=False,
        requires_financial_data=False,
        max_pages=15,
        citation_style="ieee",
        prompt_focus="scientific rigor and methodology",
        special_requirements=["peer-review quality", "reproducible results"]
    ),
    
    DocumentType.ENGINEERING_PAPER: DocumentTemplate(
        doc_type=DocumentType.ENGINEERING_PAPER,
        output_format=OutputFormat.LATEX_PDF,
        latex_documentclass="article",
        required_packages=["amsmath", "algorithm2e", "tikz", "pgfplots", "booktabs"],
        typical_sections=["Abstract", "Introduction", "System Design", "Implementation", 
                         "Performance Analysis", "Evaluation", "Conclusion"],
        optional_sections=["Related Work", "Technical Specifications", "Appendix"],
        requires_simulation=True,
        requires_figures=True,
        requires_tables=True,
        requires_algorithms=True,
        requires_financial_data=False,
        max_pages=12,
        citation_style="ieee",
        prompt_focus="technical implementation and performance",
        special_requirements=["detailed algorithms", "performance benchmarks", "system diagrams"]
    ),

    DocumentType.FINANCE_RESEARCH: DocumentTemplate(
        doc_type=DocumentType.FINANCE_RESEARCH,
        output_format=OutputFormat.LATEX_PDF,
        latex_documentclass="article",
        required_packages=["amsmath", "booktabs", "pgfplots", "siunitx", "financial"],
        typical_sections=["Executive Summary", "Market Analysis", "Methodology", 
                         "Financial Analysis", "Risk Assessment", "Recommendations", "Conclusion"],
        optional_sections=["Regulatory Environment", "Appendix", "Data Sources"],
        requires_simulation=True,
        requires_figures=True,
        requires_tables=True,
        requires_algorithms=False,
        requires_financial_data=True,
        max_pages=20,
        citation_style="apa",
        prompt_focus="financial analysis and market insights",
        special_requirements=["financial models", "market data", "risk metrics", "regulatory compliance"],
        min_references=12
    ),

    DocumentType.EQUITY_RESEARCH: DocumentTemplate(
        doc_type=DocumentType.EQUITY_RESEARCH,
        output_format=OutputFormat.LATEX_PDF,
        latex_documentclass="article",
        required_packages=["amsmath", "booktabs", "pgfplots", "siunitx"],
        typical_sections=["Investment Summary", "Company Overview", "Financial Analysis", 
                         "Valuation", "Risk Factors", "Price Target", "Recommendation"],
        optional_sections=["Industry Analysis", "Peer Comparison", "Sensitivity Analysis"],
        requires_simulation=True,
        requires_figures=True,
        requires_tables=True,
        requires_algorithms=False,
        requires_financial_data=True,
        max_pages=25,
        citation_style="financial",
        prompt_focus="investment analysis and valuation",
        special_requirements=["DCF models", "comparable analysis", "price targets", "investment thesis"],
        min_references=10
    ),
    
    DocumentType.PRESENTATION_SLIDES: DocumentTemplate(
        doc_type=DocumentType.PRESENTATION_SLIDES,
        output_format=OutputFormat.BEAMER_SLIDES,
        latex_documentclass="beamer",
        required_packages=["graphicx", "tikz", "booktabs"],
        typical_sections=["Title Slide", "Agenda", "Introduction", "Main Content", 
                         "Key Findings", "Conclusions", "Q&A"],
        optional_sections=["Background", "Methodology", "References"],
        requires_simulation=False,
        requires_figures=True,
        requires_tables=False,
        requires_algorithms=False,
        requires_financial_data=False,
        max_pages=30,
        citation_style="minimal",
        prompt_focus="clear communication and visual impact",
        special_requirements=["concise bullet points", "visual emphasis", "speaker notes"],
        requires_embedded_references=False,
        min_references=0,
        enforce_strict_sections=False
    ),
    
    DocumentType.SURVEY_PAPER: DocumentTemplate(
        doc_type=DocumentType.SURVEY_PAPER,
        output_format=OutputFormat.LATEX_PDF,
        latex_documentclass="article",
        required_packages=["amsmath", "cite", "graphicx"],
        typical_sections=["Abstract", "Introduction", "Background", "Literature Review", 
                         "Classification", "Comparative Analysis", "Future Directions", "Conclusion"],
        optional_sections=["Taxonomy", "Open Challenges", "Appendix"],
        requires_simulation=False,
        requires_figures=True,
        requires_tables=True,
        requires_algorithms=False,
        requires_financial_data=False,
        max_pages=25,
        citation_style="acm",
        prompt_focus="comprehensive literature coverage",
        special_requirements=["extensive citations", "comparative tables", "classification schemes"],
        min_references=25
    )
}

def get_document_template(doc_type: DocumentType) -> DocumentTemplate:
    """Get the template configuration for a document type"""
    return DOCUMENT_TEMPLATES.get(doc_type, DOCUMENT_TEMPLATES[DocumentType.RESEARCH_PAPER])

def get_available_document_types() -> List[str]:
    """Get list of available document type names"""
    return [doc_type.value for doc_type in DocumentType]

def infer_document_type(topic: str, field: str, question: str) -> DocumentType:
    """Infer document type based on topic, field, and question"""
    topic_lower = topic.lower()
    field_lower = field.lower()
    question_lower = question.lower()
    
    # Finance-related keywords
    finance_keywords = ["finance", "financial", "investment", "equity", "stock", "market", 
                       "trading", "portfolio", "valuation", "banking", "economic"]
    
    # Engineering keywords
    engineering_keywords = ["algorithm", "system", "implementation", "performance", 
                          "optimization", "architecture", "design", "engineering"]
    
    # Presentation keywords
    presentation_keywords = ["presentation", "slides", "talk", "lecture", "seminar", "workshop"]
    
    # Survey keywords
    survey_keywords = ["survey", "review", "literature", "comparative", "analysis of"]
    
    combined_text = f"{topic_lower} {field_lower} {question_lower}"
    
    if any(keyword in combined_text for keyword in presentation_keywords):
        return DocumentType.PRESENTATION_SLIDES
    elif any(keyword in combined_text for keyword in survey_keywords):
        return DocumentType.SURVEY_PAPER
    elif "equity" in combined_text and any(keyword in combined_text for keyword in finance_keywords):
        return DocumentType.EQUITY_RESEARCH
    elif any(keyword in combined_text for keyword in finance_keywords):
        return DocumentType.FINANCE_RESEARCH
    elif any(keyword in combined_text for keyword in engineering_keywords):
        return DocumentType.ENGINEERING_PAPER
    else:
        return DocumentType.RESEARCH_PAPER

def get_field_specific_packages(field: str) -> List[str]:
    """Get additional packages based on research field"""
    field_lower = field.lower()
    
    if "finance" in field_lower or "economic" in field_lower:
        return ["financial", "econometrics"]
    elif "computer" in field_lower or "engineering" in field_lower:
        return ["algorithm2e", "listings", "tikz"]
    elif "math" in field_lower or "statistics" in field_lower:
        return ["amsthm", "mathtools", "probability"]
    elif "physics" in field_lower:
        return ["physics", "units"]
    elif "chemistry" in field_lower:
        return ["chemfig", "chemistry"]
    else:
        return []
