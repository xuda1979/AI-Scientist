# Simple SciResearch Workflow

This repository contains a simple automation pipeline that uses OpenAI's GPT-5 to generate innovative research ideas, draft full research papers in LaTeX (with code if necessary), peer review them, and iterate until the paper meets top-journal standards.

## Features
  * Prompt the user for a research topic, field, and specific research question (via command-line arguments or interactive prompts).
  * Generate a high-value, innovative and practical research idea using GPT-5.
  * Draft a full-length research paper in LaTeX with sections (Abstract, Introduction, Related Work, Methodology, Experiments, Results, Discussion, Conclusion, References) and associated code, saved to a timestamped subdirectory inside the chosen output directory.
  * Automatically detect and improve existing papers: When a folder contains `paper.tex`, the system intelligently improves the existing work while preserving its core topic, field, and research question.
  * Peer review the generated paper using GPT-5 acting as a top journal reviewer and provide constructive feedback on rigor and clarity. The reviewer is instructed not to recommend rejection, and the editor must offer suggestions for improvement when the paper is not yet ready.
  * Automatically decide whether the paper is ready for submission. If not, revise the paper based on the review feedback and produce a unified diff showing the changes.
  * Repeat the review→evaluation→revision cycle until the editor responds "YES" or a maximum iteration count is reached.
  * Apply the diff to update the paper and any associated code.
  * Wait up to one hour for each OpenAI response by default, with optional timeout and retry count for robustness.

## NEW in this patch

**LaTeX compile checks + auto-correction inside the loop**
* Every generated or revised `paper.tex` is sanitized, compiled with `latexmk`, and if the compile fails the model is prompted with the `.log` to repair the file (up to `--max-compile-fixes` attempts).

**No more over-wide tables/figures**
* All `\includegraphics` are forced to `width=\linewidth`.
* Tables are wrapped in `\begin{adjustbox}{width=\linewidth} ... \end{adjustbox}` when needed.
* Adds safe preamble packages (`graphicx, adjustbox, booktabs, tabularx, caption, subcaption, geometry`).

**Exactly one `.tex` + one `.py` per project**
* Code blocks in the LaTeX (e.g., `lstlisting` or `minted` with Python) are consolidated into a single `simulation.py`.
* We run `simulation.py` and pass the outputs back to the model to improve the text/tables/plots.
* Any extra `.tex`/`.py` files are moved into `archive/` so the folder holds only `paper.tex` and `simulation.py`.

## Requirements

  * Python 3.8+.
  * `openai` Python package.
  * Set the `OPENAI_API_KEY` environment variable with your OpenAI API key. The default model used is `gpt-5`, but you can specify another model.
  * TeX toolchain: `latexmk` recommended (or `pdflatex` as fallback).

## Usage

```bash
pip install openai

export OPENAI_API_KEY=YOUR_API_KEY

    # Create a new paper
    python sciresearch_workflow.py --topic "quantum computing algorithms" --field "Computer Science" --question "How can error rates be reduced?" --output-dir ./output

    # Force modification of existing paper
    python sciresearch_workflow.py --topic "new research direction" --field "Computer Science" --question "New research question?" --output-dir ./existing_paper_folder --modify-existing
```

### New CLI flags

```
--max-compile-fixes INT     # default 2. Attempts to auto-fix broken LaTeX via the model.
--no-latex-autofix          # if set, only checks compile; does not call the model to auto-fix.
--strict-singletons         # default true. Keep only `paper.tex` and `simulation.py` in the project folder (others archived).
--python-exec PATH          # which Python to use for running simulation.py (default: current interpreter).
```

### Modifying Existing Papers

The script will generate a research idea, write a paper into a timestamped subdirectory of the specified output directory (for example, `output/20250101_120000/paper.tex`), consolidate any Python code from `lstlisting`/`minted` blocks into a single `simulation.py`, run the simulation, and then review/revise the paper using those results. The loop continues until the paper is accepted or `--max-iterations` is reached.

Optional flags `--request-timeout` and `--max-retries` control how long the workflow waits for each OpenAI response and how many times it retries a failed request. By default, each request waits up to one hour; pass `--request-timeout 0` to wait indefinitely.

If `--field` or `--question` is omitted, the script will prompt for these values interactively.
 
