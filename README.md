# Simple SciResearch Workflow

This repository contains a simple automation pipeline that uses OpenAI's GPT-5 to generate innovative research ideas, draft full research papers (with code if necessary), peer review them, and iterate until the paper meets top-journal standards.

## Features

- Prompt the user for a research topic.
- Generate a high-value, innovative and practical research idea using GPT-5.
- Draft a full-length research paper in LaTeX with sections (Abstract, Introduction, Related Work, Methodology, Experiments, Results, Discussion, Conclusion, References) and associated code, saved to an output directory.
- Peer review the generated paper using GPT-5 acting as a top journal reviewer and provide constructive feedback on rigor and clarity.
- Automatically decide whether the paper is ready for submission. If not, revise the paper based on the review feedback and produce a unified diff showing the changes.
- Repeat the review→evaluation→revision cycle until the editor responds "YES" or a maximum iteration count is reached.
- Apply the diff to update the paper and any associated code.

## Requirements

- Python 3.8+.
- [openai](https://pypi.org/project/openai/) Python package.
- Set the `OPENAI_API_KEY` environment variable with your OpenAI API key. The default model used is `gpt-5`, but you can specify another model.

## Usage

```bash
pip install openai

export OPENAI_API_KEY=YOUR_API_KEY

python sciresearch_workflow.py --topic "quantum computing algorithms" --output-dir ./output --max-iters 3
```

The script will generate a research idea, write a paper into the specified output directory (`output/paper.tex`), and iteratively review and revise it until the editor approves or the iteration limit is reached.
