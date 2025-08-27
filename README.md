# Simple SciResearch Workflow

This repository contains a simple automation pipeline that uses OpenAI's GPT-5 to generate innovative research ideas, draft full research papers in LaTeX (with code if necessary), peer review them, and iterate until the paper meets top-journal standards.

## Features

- Prompt the user for a research topic, field, and specific research question (via command-line arguments or interactive prompts).
- Generate a high-value, innovative and practical research idea using GPT-5.
 
- Draft a full-length research paper in LaTeX with sections (Abstract, Introduction, Related Work, Methodology, Experiments, Results, Discussion, Conclusion, References) and associated code, saved to a timestamped subdirectory inside the chosen output directory.
- Peer review the generated paper using GPT-5 acting as a top journal reviewer and provide constructive feedback on rigor and clarity. The reviewer is instructed not to recommend rejection, and the editor must offer suggestions for improvement when the paper is not yet ready.
 
- Automatically decide whether the paper is ready for submission. If not, revise the paper based on the review feedback and produce a unified diff showing the changes.
- Repeat the review→evaluation→revision cycle until the editor responds "YES" or a maximum iteration count is reached.
- Apply the diff to update the paper and any associated code.
- Wait up to one hour for each OpenAI response by default, with optional timeout and retry count for robustness.

## Requirements

- Python 3.8+.
- [openai](https://pypi.org/project/openai/) Python package.
- Set the `OPENAI_API_KEY` environment variable with your OpenAI API key. The default model used is `gpt-5`, but you can specify another model.

## Usage

```bash
pip install openai

export OPENAI_API_KEY=YOUR_API_KEY

python sciresearch_workflow.py --topic "quantum computing algorithms" --field "Computer Science" --question "How can error rates be reduced?" --output-dir ./output
```

 The script will generate a research idea, write a paper into a timestamped subdirectory of the specified output directory (for example, `output/20250101_120000/paper.tex`), extract any Python code from `lstlisting` blocks into `code_<n>.py`, and then review and revise the paper if necessary.

Optional flags `--request-timeout` and `--max-retries` control how long the workflow waits for each OpenAI response and how many times it retries a failed request. By default, each request waits up to one hour; pass `--request-timeout 0` to wait indefinitely.

If `--field` or `--question` is omitted, the script will prompt for these values interactively.
 
