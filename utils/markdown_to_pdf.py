"""Utilities for converting Markdown documents to PDF."""
from __future__ import annotations

import argparse
import re
from html import escape
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, StyleSheet1, getSampleStyleSheet
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.cidfonts import UnicodeCIDFont
from reportlab.platypus import (
    Paragraph,
    Preformatted,
    SimpleDocTemplate,
    Spacer,
)


_HEADING_SIZES = {
    1: 22,
    2: 18,
    3: 16,
    4: 14,
    5: 13,
    6: 12,
}


def _register_fonts() -> None:
    """Register fonts that support simplified Chinese text."""
    # ``STSong-Light`` is bundled with ReportLab and provides wide Unicode coverage
    # suitable for simplified Chinese content.
    pdfmetrics.registerFont(UnicodeCIDFont("STSong-Light"))


def _build_styles() -> StyleSheet1:
    styles = getSampleStyleSheet()
    body = styles["BodyText"]
    body.fontName = "STSong-Light"
    body.fontSize = 11
    body.leading = 16
    body.spaceAfter = 10

    styles["Normal"].fontName = "STSong-Light"
    styles["Normal"].fontSize = 11
    styles["Normal"].leading = 16

    for level in range(1, 7):
        style_name = f"Heading{level}"
        if style_name in styles:
            styles[style_name].fontName = "STSong-Light"
            styles[style_name].fontSize = _HEADING_SIZES.get(level, 12)
            styles[style_name].leading = styles[style_name].fontSize + 4
            styles[style_name].spaceAfter = 12

    bullet = ParagraphStyle(
        name="BodyBullet",
        parent=body,
        leftIndent=18,
        bulletIndent=9,
        spaceAfter=6,
    )
    styles.add(bullet)

    numbered = ParagraphStyle(
        name="BodyNumbered",
        parent=body,
        leftIndent=20,
        bulletIndent=9,
        spaceAfter=6,
    )
    styles.add(numbered)

    if "Code" in styles:
        code = styles["Code"]
        code.parent = body
        code.fontName = "Courier"
        code.fontSize = 9
        code.leading = 12
        code.leftIndent = 12
        code.backColor = colors.whitesmoke
    else:
        code = ParagraphStyle(
            name="Code",
            parent=body,
            fontName="Courier",
            fontSize=9,
            leading=12,
            leftIndent=12,
            backColor=colors.whitesmoke,
        )
        styles.add(code)

    return styles


def _format_inline(text: str) -> str:
    text = text.strip()
    text = escape(text)

    # Bold / italics
    text = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", text)
    text = re.sub(r"\*(.+?)\*", r"<i>\1</i>", text)

    # Inline code
    text = re.sub(r"`([^`]+)`", r"<font face=\"Courier\">\1</font>", text)

    # Links
    text = re.sub(r"\[(.+?)\]\((.+?)\)", r"<link href=\"\2\">\1</link>", text)

    # Collapsing multiple spaces that may appear after stripping markdown symbols.
    text = re.sub(r"\s+", " ", text)
    return text


_ListEntry = Tuple[str, int, str]


def _flush_paragraph(paragraph_lines: List[str], flowables: List, styles: StyleSheet1) -> None:
    if not paragraph_lines:
        return
    content = " ".join(paragraph_lines)
    if not content.strip():
        paragraph_lines.clear()
        return
    flowables.append(Paragraph(_format_inline(content), styles["BodyText"]))
    flowables.append(Spacer(1, 6))
    paragraph_lines.clear()


def _flush_list(
    list_items: List[_ListEntry],
    flowables: List,
    styles: StyleSheet1,
    style_cache: dict[Tuple[str, int], ParagraphStyle],
) -> None:
    if not list_items:
        return

    for marker, indent, content in list_items:
        style_key = ("BodyBullet" if marker == "•" else "BodyNumbered", indent)
        if style_key not in style_cache:
            base = styles[style_key[0]]
            style_cache[style_key] = ParagraphStyle(
                name=f"{style_key[0]}-{indent}",
                parent=base,
                leftIndent=base.leftIndent + indent,
                bulletIndent=base.bulletIndent + indent,
            )
        style = style_cache[style_key]
        flowables.append(
            Paragraph(
                _format_inline(content),
                style,
                bulletText=marker,
            )
        )
    flowables.append(Spacer(1, 6))
    list_items.clear()


def _flush_code_block(code_lines: List[str], flowables: List, styles: StyleSheet1) -> None:
    if not code_lines:
        return
    flowables.append(Preformatted("\n".join(code_lines), styles["Code"]))
    flowables.append(Spacer(1, 6))
    code_lines.clear()


def _parse_markdown(lines: Iterable[str], styles: StyleSheet1) -> List:
    flowables: List = []
    paragraph_lines: List[str] = []
    list_items: List[_ListEntry] = []
    style_cache: dict[Tuple[str, int], ParagraphStyle] = {}
    code_lines: List[str] = []
    in_code_block = False

    for raw_line in lines:
        line = raw_line.rstrip("\n")
        if in_code_block:
            if line.strip().startswith("```"):
                _flush_code_block(code_lines, flowables, styles)
                in_code_block = False
            else:
                code_lines.append(line)
            continue

        stripped = line.strip()
        if stripped.startswith("```"):
            _flush_paragraph(paragraph_lines, flowables, styles)
            _flush_list(list_items, flowables, styles, style_cache)
            in_code_block = True
            code_lines.clear()
            continue

        if not stripped:
            _flush_paragraph(paragraph_lines, flowables, styles)
            _flush_list(list_items, flowables, styles, style_cache)
            continue

        heading_match = re.match(r"^(#{1,6})\s+(.*)", stripped)
        if heading_match:
            _flush_paragraph(paragraph_lines, flowables, styles)
            _flush_list(list_items, flowables, styles, style_cache)
            level = len(heading_match.group(1))
            text = heading_match.group(2).strip()
            style_name = f"Heading{level}"
            style = styles.get(style_name, styles["Heading6"])
            flowables.append(Paragraph(_format_inline(text), style))
            flowables.append(Spacer(1, 8))
            continue

        list_match = re.match(r"^(\s*)([-+*]|\d+[.)])\s+(.*)", line)
        if list_match:
            _flush_paragraph(paragraph_lines, flowables, styles)
            indent_spaces = len(list_match.group(1))
            marker_raw = list_match.group(2)
            content = list_match.group(3)
            marker = "•" if marker_raw in {"-", "+", "*"} else marker_raw
            list_items.append((marker, indent_spaces, content))
            continue

        # Default: accumulate into current paragraph.
        paragraph_lines.append(stripped)

    _flush_code_block(code_lines, flowables, styles)
    _flush_paragraph(paragraph_lines, flowables, styles)
    _flush_list(list_items, flowables, styles, style_cache)
    return flowables


def convert_markdown_to_pdf(source: Path, destination: Path) -> None:
    if not source.exists():
        raise FileNotFoundError(f"Markdown file not found: {source}")

    _register_fonts()
    styles = _build_styles()

    with source.open("r", encoding="utf-8") as fh:
        flowables = _parse_markdown(fh, styles)

    destination.parent.mkdir(parents=True, exist_ok=True)
    doc = SimpleDocTemplate(
        str(destination),
        pagesize=A4,
        topMargin=50,
        bottomMargin=50,
        leftMargin=50,
        rightMargin=50,
    )
    doc.build(flowables)


def _parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert a Markdown file to PDF.")
    parser.add_argument("source", type=Path, help="Path to the Markdown file")
    parser.add_argument("destination", type=Path, help="Destination PDF path")
    return parser.parse_args(argv)


def main(argv: Optional[Iterable[str]] = None) -> None:
    args = _parse_args(argv)
    convert_markdown_to_pdf(args.source, args.destination)


if __name__ == "__main__":
    main()
