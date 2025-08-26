# Simple SciResearch Workflow

This repository contains a simple automation pipeline that uses OpenAI's GPT-5 to generate innovative research ideas, draft full research papers (with code if necessary), peer review them, and iterate until the paper meets top-journal standards.

## Features

- Prompt the user for a research topic, field, and specific research question (via command-line arguments or interactive prompts).
- Generate a high-value, innovative and practical research idea using GPT-5.
- Draft a full research paper with sections (Abstract, Introduction, Methodology, Experiments, Results, Conclusion) and associated code, saved to an output directory.
- Peer review the generated paper using GPT-5 acting as a top journal reviewer and provide constructive feedback.
- Automatically decide whether the paper is ready for submission. If not, revise the paper based on the review feedback and produce a unified diff showing the changes.
- Apply the diff to update the paper and any associated code.

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

The script will generate a research idea, write a paper into the specified output directory (`output/paper.md`), review and revise it if necessary.

If `--field` or `--question` is omitted, the script will prompt for these values interactively.
