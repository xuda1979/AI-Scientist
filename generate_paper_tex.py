"""Generate a LaTeX paper (paper.tex) from a DOCX source.

Heuristic outline construction:
 - Abstract: first non-empty paragraph (trimmed)
 - Headings are inferred by short paragraphs (< 12 words) with Title Case or ending with ':'
 - Background: paragraphs until a detected heading resembling problem/definition
 - Method: paragraphs containing keywords (algorithm, approach, method)
 - Results: paragraphs containing keywords (result, experiment, performance, test)
 - Discussion: remaining paragraphs not already assigned
If a section would be empty, we insert a placeholder so that downstream workflow can refine.

This is intentionally lightweight; the main workflow (modify-existing mode) can further improve the draft.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import List

try:
    from docx import Document  # type: ignore
except ImportError as e:  # pragma: no cover
    raise SystemExit("python-docx not installed. Run: pip install python-docx") from e

DOCX_REL_PATH = Path("output/max_cut/Max-cut_problem.docx")
OUTPUT_TEX_PATH = Path("output/max_cut/paper.tex")

SECTION_KEYWORDS = {
    "method": {"algorithm", "approach", "method", "procedure", "pipeline"},
    "results": {"result", "experiment", "performance", "evaluation", "benchmark"},
}

LATEX_SPECIALS = {
    "\\": r"\textbackslash{}",
    "&": r"\&",
    "%": r"\%",
    "$": r"\$",
    "#": r"\#",
    "_": r"\_",
    "{": r"\{",
    "}": r"\}",
    "~": r"\textasciitilde{}",
    "^": r"\textasciicircum{}",
}


def sanitize(text: str) -> str:
    """Escape LaTeX special characters in text."""
    out = []
    for ch in text:
        out.append(LATEX_SPECIALS.get(ch, ch))
    # Collapse excessive whitespace
    s = re.sub(r"\s+", " ", "".join(out)).strip()
    return s


def is_heading(paragraph: str) -> bool:
    p = paragraph.strip()
    if not p:
        return False
    if len(p.split()) <= 12 and (p.endswith(":") or p == p.title()):
        return True
    # ALL CAPS short segment
    if len(p.split()) <= 8 and p.isupper():
        return True
    return False


def classify_paragraph(p: str) -> str | None:
    low = p.lower()
    for section, kws in SECTION_KEYWORDS.items():
        if any(k in low for k in kws):
            return section
    return None


def load_docx_paragraphs(path: Path) -> List[str]:
    if not path.exists():
        raise SystemExit(f"Source DOCX not found: {path}")
    doc = Document(str(path))
    paras = [p.text.strip() for p in doc.paragraphs if p.text.strip()]
    # Deduplicate consecutive identical paragraphs (common in exports)
    cleaned: List[str] = []
    prev = None
    for p in paras:
        if p != prev:
            cleaned.append(p)
        prev = p
    return cleaned


def build_sections(paragraphs: List[str]) -> dict:
    sections = {"abstract": "", "background": [], "method": [], "results": [], "discussion": []}
    if not paragraphs:
        sections["abstract"] = "(No content extracted from DOCX.)"
        return sections
    sections["abstract"] = paragraphs[0]
    for p in paragraphs[1:]:
        cls = classify_paragraph(p)
        if cls == "method":
            sections["method"].append(p)
        elif cls == "results":
            sections["results"].append(p)
        else:
            sections["background"].append(p)
    # If background huge and method empty, split heuristically
    if sections["method"] == [] and len(sections["background"]) > 6:
        mid = len(sections["background"]) // 2
        sections["method"] = sections["background"][mid:]
        sections["background"] = sections["background"][:mid]
    # Discussion = tail of results if empty
    if not sections["discussion"]:
        tail_source = sections["results"] or sections["method"]
        if tail_source:
            sections["discussion"] = tail_source[-2:]
    return sections


def render_latex(sections: dict) -> str:
    def wrap_paras(paras: List[str]) -> str:
        return "\n\n".join(sanitize(p) for p in paras) if paras else "(Placeholder for future expansion.)"

    return f"""% Auto-generated draft from DOCX by generate_paper_tex.py
% Date: AUTO
\documentclass[11pt]{{article}}
\usepackage[utf8]{{inputenc}}
\usepackage[T1]{{fontenc}}
\usepackage{{lmodern}}
\usepackage{{geometry}}
\usepackage{{graphicx}}
\usepackage{{hyperref}}
\usepackage{{amsmath, amssymb}}
\usepackage{{microtype}}
\geometry{{margin=1in}}

\title{{Preliminary Draft: Max-Cut Problem Paper}}
\author{{Automated Generation}}
\date{{}}

\begin{{document}}
\maketitle

\begin{{abstract}}
{sanitize(sections['abstract']) or '(No abstract content)'}
\end{{abstract}}

\section{{Introduction and Background}}
{wrap_paras(sections['background'])}

\section{{Method}}
{wrap_paras(sections['method'])}

\section{{Results}}
{wrap_paras(sections['results'])}

\section{{Discussion}}
{wrap_paras(sections['discussion'])}

\section{{Limitations}}
This automatically generated draft may omit structural nuances from the source DOCX. A refined pass should improve clarity, add formal definitions, and integrate citations.

\section{{Future Work}}
Integrate formal proofs (if applicable), complexity analysis, comparative evaluation against approximation algorithms, and domain-specific applications.

\section{{Conclusion}}
This draft presents an initial structured synthesis of the provided Max-Cut problem document. Subsequent automated refinement (modify-existing workflow) can enhance coherence, technical depth, and scholarly apparatus.

\section*{{References}}
% Placeholder bibliography entries; integrate with workflow citation system.
\begin{{itemize}}
  \item Placeholder reference 1.
  \item Placeholder reference 2.
\end{{itemize}}

\end{{document}}
"""


def main() -> None:
    paragraphs = load_docx_paragraphs(DOCX_REL_PATH)
    sections = build_sections(paragraphs)
    tex = render_latex(sections)
    OUTPUT_TEX_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_TEX_PATH.write_text(tex, encoding="utf-8")
    print(f"Wrote LaTeX draft to {OUTPUT_TEX_PATH}")
    print("To refine further with workflow: python main.py --modify-existing --output-dir output/max_cut")


if __name__ == "__main__":
    main()
