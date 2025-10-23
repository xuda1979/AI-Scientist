# Comprehensive Research Workflow

The comprehensive workflow combines automated planning, drafting, validation, and dissemination so the toolkit can generate complete research papers—not just supporting documentation. The implementation lives in `src/core/comprehensive_workflow.py` and can be executed through the convenience wrapper `run_comprehensive_workflow.py`.

```bash
python run_comprehensive_workflow.py "Quantum Reservoir Computing" physics "Develop a robust benchmarking suite for quantum reservoir computing architectures" --dataset "Synthetic qubit time-series benchmarks" --print-json
```

The command above orchestrates every stage, stores JSON and Markdown artifacts under `output/comprehensive_workflow/<topic-slug>/`, and optionally streams the structured plan to stdout for downstream automation.

## Research Preparation
- Generate research topics and ideas tailored to emerging areas of interest.
- Analyze recent literature trends to position projects within the current scientific landscape.
- Build research hypotheses and innovation points that differentiate the study.
- Automatically search and summarize relevant literature.
- Draft literature review sections that synthesize key findings.

## ✍️ Writing Stage
- Generate a paper structure and outline aligned with target publication standards.
- Optimize paragraph logic and flow for clarity and coherence.
- Automatically write the abstract, introduction, methods, results, and discussion sections.
- Provide statements on research significance, limitations, and future directions.

## 📊 Data Analysis & Figures
- Automatically generate results descriptions from processed data.
- Provide statistical analysis statements and interpretations for transparency.
- Generate and optimize figures, tables, and captions suitable for publication.

## 🧾 Language & Formatting
- Check grammar, spelling, and logical consistency across the manuscript.
- Improve academic tone and clarity for professional presentation.
- Translate between English and other languages while preserving academic style.
- Automatically format references and document layout per journal guidelines.

## 🧮 Scientific Reasoning & Validation
- Simulate reviewer questions and suggestions to stress-test the manuscript.
- Assist in drafting rebuttals or responses to reviewers.
- Analyze experimental design and logical soundness.
- Generate study limitations and improvement suggestions to strengthen credibility.

## 📤 Submission & Dissemination
- Match the manuscript to suitable journals based on scope and impact.
- Write tailored cover letters for submission.
- Generate concise social media summaries to promote published findings.

Each stage produces structured JSON so downstream automation—such as LaTeX generation, simulation orchestration, or submission tooling—can consume the outputs without brittle parsing. Markdown summaries are generated in parallel for quick human review.

