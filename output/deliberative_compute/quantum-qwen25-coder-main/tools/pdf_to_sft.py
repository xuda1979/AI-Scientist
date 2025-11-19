#!/usr/bin/env python3
"""Convert quantum-computing PDFs into supervised fine-tuning data.

This utility reads one or more PDF files, extracts their text, splits the
content into manageable chunks, and emits a JSON Lines file that can be used
with ``train_sft.py``. Each output record contains a ``prompt`` that instructs
the model what to do with the text and a ``code`` field that serves as the
training target. Thousands of papers can be processed in a single run – the
script offers optional multiprocessing, deduplication, and lightweight text
clean-up to make large-scale ingestion practical.

Examples
--------
Create a dataset from all PDFs in ``papers/`` and save it as
``data/papers.jsonl`` while stripping the reference section and dropping short
chunks::

    python tools/pdf_to_sft.py \
        --input papers \
        --output data/papers.jsonl \
        --chunk-size 800 \
        --chunk-overlap 120 \
        --strip-references \
        --min-chunk-length 200

By default, the generated prompt asks the model to总结 the provided chunk and
the target text is simply the original chunk. You can customise the instruction
and the target via ``--instruction-template`` and ``--target-template`` and even
add an ``analysis`` field using ``--analysis-template``.

Requirements
------------
This script depends on :mod:`pypdf` (preferred) or :mod:`PyPDF2`. Install one of
these libraries before running the script::

    pip install pypdf

"""

from __future__ import annotations

import argparse
import json
import logging
import re
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from pathlib import Path
from typing import Iterable, Iterator, List, Optional, Sequence, Tuple


logger = logging.getLogger(__name__)


def extract_text_from_pdf(path: Path) -> str:
    """Extract text from a PDF file.

    The function tries to import :mod:`pypdf` first and falls back to
    :mod:`PyPDF2` if necessary. An informative error message is raised when no
    suitable backend is available.
    """

    try:  # Prefer the actively maintained package name.
        from pypdf import PdfReader  # type: ignore
    except ImportError:
        try:
            from PyPDF2 import PdfReader  # type: ignore
        except ImportError as exc:  # pragma: no cover - dependency error path
            raise ImportError(
                "Missing dependency: install either 'pypdf' or 'PyPDF2' to read PDF files."
            ) from exc

    reader = PdfReader(str(path))
    pages: List[str] = []
    for page_index, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        cleaned = _clean_page_text(text)
        if cleaned:
            pages.append(cleaned)
        else:
            logger.debug("Discarded empty page %s in %s", page_index, path)
    return "\n\n".join(pages)


def _clean_page_text(text: str) -> str:
    """Normalise whitespace and repair simple hyphenation issues."""

    if not text:
        return ""

    text = text.replace("\r", "\n").replace("\u00ad", "")
    lines = [line.strip() for line in text.splitlines()]

    merged: List[str] = []
    for line in lines:
        if not line:
            if merged and merged[-1]:
                merged.append("")
            continue
        normalised = re.sub(r"\s+", " ", line)
        if merged and merged[-1].endswith("-") and normalised:
            merged[-1] = merged[-1][:-1] + normalised.lstrip()
        else:
            merged.append(normalised)

    cleaned = "\n".join(segment for segment in merged if segment is not None)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()


def iter_pdf_files(paths: Sequence[Path]) -> Iterator[Path]:
    """Yield all PDF files under the provided paths."""

    for path in paths:
        if path.is_dir():
            for candidate in sorted(path.rglob("*")):
                if candidate.is_file() and candidate.suffix.lower() == ".pdf":
                    yield candidate
        elif path.is_file() and path.suffix.lower() == ".pdf":
            yield path


def chunk_text(
    text: str,
    chunk_size: int,
    chunk_overlap: int,
) -> List[Tuple[str, int, int]]:
    """Split *text* into overlapping chunks.

    Returns a list of ``(chunk, start_offset, end_offset)`` tuples where the
    offsets are measured on the normalised text.
    """

    if chunk_size <= 0:
        raise ValueError("chunk_size must be a positive integer")
    if not 0 <= chunk_overlap < chunk_size:
        raise ValueError("chunk_overlap must be in [0, chunk_size)")

    text = " ".join(text.split())  # normalise whitespace
    if not text:
        return []

    chunks: List[Tuple[str, int, int]] = []
    start = 0
    length = len(text)
    while start < length:
        end = min(start + chunk_size, length)
        chunk = text[start:end]
        chunks.append((chunk, start, end))
        if end == length:
            break
        start = max(0, end - chunk_overlap)
    return chunks


def build_records(
    pdf_paths: Iterable[Path],
    chunk_size: int,
    chunk_overlap: int,
    instruction_template: str,
    target_template: str,
    analysis_template: Optional[str],
    min_chunk_length: int,
    strip_references: bool,
) -> Iterator[dict]:
    """Create JSONL records for all PDF chunks."""

    for pdf_path in pdf_paths:
        text = extract_text_from_pdf(pdf_path)
        if strip_references:
            text = _strip_reference_section(text)

        for index, (chunk, start, end) in enumerate(
            chunk_text(text, chunk_size, chunk_overlap),
            start=1,
        ):
            if len(chunk) < min_chunk_length:
                continue
            prompt = instruction_template.format(chunk=chunk, source=str(pdf_path), index=index)
            target = target_template.format(chunk=chunk, source=str(pdf_path), index=index)
            record = {
                "prompt": prompt,
                "code": target,
                "metadata": {
                    "source": str(pdf_path),
                    "chunk_index": index,
                    "char_start": start,
                    "char_end": end,
                },
            }
            if analysis_template:
                record["analysis"] = analysis_template.format(
                    chunk=chunk, source=str(pdf_path), index=index
                )
            yield record


def _strip_reference_section(text: str) -> str:
    """Remove reference/bibliography sections heuristically."""

    if not text:
        return text

    pattern = re.compile(r"\n(?:references|bibliography)\b", re.IGNORECASE)
    match = pattern.search(text)
    if not match:
        return text
    cutoff = match.start()
    if cutoff < len(text) * 0.4:  # Keep the text if the heading is too early.
        return text
    return text[:cutoff].strip()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert PDF papers to JSONL for SFT training")
    parser.add_argument(
        "--input",
        type=str,
        nargs="+",
        required=True,
        help="One or more PDF files or directories containing PDFs.",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Path to the output JSONL file.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=1024,
        help="Maximum number of characters per chunk (default: 1024).",
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=128,
        help="Number of characters to overlap between consecutive chunks (default: 128).",
    )
    parser.add_argument(
        "--instruction-template",
        type=str,
        default="请阅读以下论文片段并总结其关键技术要点：\n\n{chunk}",
        help="Template used to build the prompt. Available variables: {chunk}, {source}, {index}.",
    )
    parser.add_argument(
        "--target-template",
        type=str,
        default="{chunk}",
        help="Template used to build the target output. Variables: {chunk}, {source}, {index}.",
    )
    parser.add_argument(
        "--analysis-template",
        type=str,
        default=None,
        help="Optional template used to populate an 'analysis' field in each record.",
    )
    parser.add_argument(
        "--min-chunk-length",
        type=int,
        default=128,
        help="Discard chunks shorter than this many characters after normalisation.",
    )
    parser.add_argument(
        "--strip-references",
        action="store_true",
        help="Heuristically drop the References/Bibliography section from each PDF.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=0,
        help="Number of worker processes for parallel PDF parsing (0 disables multiprocessing).",
    )
    parser.add_argument(
        "--dedupe",
        action="store_true",
        help="Skip output records whose target text has already been emitted.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Reduce logging verbosity.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    logging.basicConfig(level=logging.WARNING if args.quiet else logging.INFO, format="%(levelname)s: %(message)s")

    input_paths = [Path(path).expanduser() for path in args.input]
    pdf_paths = list(iter_pdf_files(input_paths))
    if not pdf_paths:
        raise FileNotFoundError("No PDF files found in the provided input paths.")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info("Discovered %s PDF files", len(pdf_paths))

    record_iter: Iterator[dict]

    if args.workers and args.workers > 1:
        logger.info("Processing PDFs with %s worker processes", args.workers)
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            worker = partial(
                _process_single_pdf,
                chunk_size=args.chunk_size,
                chunk_overlap=args.chunk_overlap,
                instruction_template=args.instruction_template,
                target_template=args.target_template,
                analysis_template=args.analysis_template,
                min_chunk_length=args.min_chunk_length,
                strip_references=args.strip_references,
            )
            record_iter = _chain_iterables(executor.map(worker, pdf_paths))
    else:
        logger.info("Processing PDFs sequentially")
        record_iter = build_records(
            pdf_paths=pdf_paths,
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap,
            instruction_template=args.instruction_template,
            target_template=args.target_template,
            analysis_template=args.analysis_template,
            min_chunk_length=args.min_chunk_length,
            strip_references=args.strip_references,
        )

    total_records = 0
    seen_targets: set[str] = set()
    with output_path.open("w", encoding="utf-8") as f:
        for record in record_iter:
            target_text = record.get("code") or ""
            if args.dedupe:
                if target_text in seen_targets:
                    continue
                seen_targets.add(target_text)
            json.dump(record, f, ensure_ascii=False)
            f.write("\n")
            total_records += 1

    logger.info("Wrote %s records derived from %s PDFs to %s", total_records, len(pdf_paths), output_path)
    print(f"Wrote {total_records} records derived from {len(pdf_paths)} PDFs to {output_path}")


def _process_single_pdf(
    pdf_path: Path,
    *,
    chunk_size: int,
    chunk_overlap: int,
    instruction_template: str,
    target_template: str,
    analysis_template: Optional[str],
    min_chunk_length: int,
    strip_references: bool,
) -> List[dict]:
    return list(
        build_records(
            pdf_paths=[pdf_path],
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            instruction_template=instruction_template,
            target_template=target_template,
            analysis_template=analysis_template,
            min_chunk_length=min_chunk_length,
            strip_references=strip_references,
        )
    )


def _chain_iterables(iterables: Iterable[Iterable[dict]]) -> Iterator[dict]:
    for iterable in iterables:
        for item in iterable:
            yield item


if __name__ == "__main__":
    main()
