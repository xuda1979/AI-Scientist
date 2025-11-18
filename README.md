# AI-Scientist

_Automated research partner that turns open-ended ideas into simulation-backed, publication-ready LaTeX papers._

> **AI-Scientist guarantees two mission-critical safeguards:** every citation must refer to an authentic, verifiable publication, and every figure/table must be generated inside the LaTeX manuscript from real simulation outputs—no placeholders, no external assets.

---

## Table of Contents
1. [Project Overview](#project-overview)
2. [Feature Highlights](#feature-highlights)
3. [End-to-End Workflow](#end-to-end-workflow)
4. [Architecture & Directory Map](#architecture--directory-map)
5. [Installation](#installation)
6. [Configuration & Credentials](#configuration--credentials)
7. [Running the Workflow](#running-the-workflow)
8. [Outputs & Artifacts](#outputs--artifacts)
9. [Quality Gates & Validation](#quality-gates--validation)
10. [Troubleshooting & FAQ](#troubleshooting--faq)
11. [Development & Contributions](#development--contributions)
12. [Citation](#citation)

---

## Project Overview
AI-Scientist is a programmable orchestration layer that combines state-of-the-art language models, deterministic simulations, and strict validation to produce fully reproducible research packages. It removes the repetitive work of scoping topics, writing LaTeX, crafting figures, fixing compilation errors, and responding to peer review so that you can focus on the underlying research question.

### Why it exists
- **Idea-to-paper automation:** Starting from a short topic description, the workflow brainstorms possible studies, selects a promising angle, and builds an outline before any text is drafted.
- **Executable evidence:** The generated `simulation.py` is executed, and its outputs become the only source of numbers and figures referenced in the paper, making the manuscript self-contained and auditable.
- **Transparent iteration:** Every draft, review, revision, and quality score is logged, letting you inspect what decisions were made and why.

---

## Feature Highlights
| Capability | Details |
| --- | --- |
| **Brainstorming & Planning** | Structured ideation generates multiple candidate studies, scores them, and produces section-by-section plans tailored to your prompt. |
| **Single-file LaTeX Drafting** | Produces a compilable `paper.tex` with embedded bibliography and LaTeX-native figures/tables, ensuring portability across TeX engines. |
| **Authentic References** | References must be verifiable publications; automated checks flag placeholders, suspicious author strings, or fabricated journals before acceptance. |
| **Executable Simulations** | Simulation code is extracted from the draft, run via `utils.sim_runner`, and summarized for the reviewer/reviser agents. |
| **Quality Control Loop** | Reviewer personas validate structure, citations, figures, and numerical claims; author personas revise until the configured score threshold is met. |
| **Offline Demo Mode** | A deterministic sample project can be generated without API keys, showcasing orchestration, simulation execution, and artifact packaging entirely locally. |
| **Detailed Logs & Metadata** | Each run emits timestamped log files plus JSON summaries of ideation, reviews, code execution, and quality metrics for later auditing. |

---

## End-to-End Workflow
1. **Collect Inputs** – Provide topic, field, research question, and optional user instructions via CLI or config file.
2. **Ideate** – Generate multiple research ideas (`--num-ideas`) and retain top candidates for deeper planning; can be skipped or forced via CLI flags.
3. **Draft** – Create `paper.tex` and `simulation.py`, embedding a bibliography and placeholder-free LaTeX figures in a single file.
4. **Execute Simulations** – Extract code, run it (with auto-fixing when enabled), and gather structured summaries of the outputs for downstream agents.
5. **Review & Score** – Reviewer LLM checks correctness, citations, and visuals. Additional parallel validators inspect DOIs, LaTeX structure, and figure existence.
6. **Revise** – Author persona updates the manuscript according to review feedback until the quality threshold is satisfied or `--max-iterations` is reached.
7. **Deliver Artifacts** – Store the final paper, simulation outputs, logs, and metadata inside `output/<project_name>/`.

---

## Architecture & Directory Map
```
AI-Scientist/
├─ sciresearch_workflow.py      # CLI orchestrator, logging setup, offline demo, validation loop
├─ src/                         # Extended workflow modules and helpers
├─ utils/                       # Simulation runners, LaTeX tooling, model clients, validation utilities
├─ docs/                        # Design notes, requirements, prompt templates, whitepapers
├─ output/                      # Generated per-run artifacts (created at runtime)
├─ logs/                        # Rolling workflow logs (auto-created)
├─ requirements.txt             # Python dependencies
└─ config_example.json          # Editable JSON configuration template
```
- **`sciresearch_workflow.py`** wires together CLI arguments, configuration loading, Google/OSS model clients, simulation execution, LaTeX compilation, and reviewer/revision agents.
- **`utils/`** contains `sim_runner.py`, `latex_tools.py`, `parallel_checks.py`, and `model_client.py`, providing the deterministic machinery that keeps drafts compilable and simulations reproducible.
- **`docs/`** includes enhanced workflow requirements describing the authenticity and self-contained-visual mandates enforced throughout the pipeline.

---

## Installation
### Prerequisites
- Python **3.10+** with `pip`.
- A modern TeX distribution (TeX Live, MiKTeX, or MacTeX) for compiling `paper.tex`.
- At least one LLM endpoint (OpenAI GPT, OSS-120B, or Google Gemini). Offline demo mode can run without network access but will not contact external APIs.

### Setup Steps
```bash
git clone https://github.com/AI-Scientist/AI-Scientist.git
cd AI-Scientist
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```
> The CLI automatically rescues the common typo `OPEANAI_API_KEY` by copying it into `OPENAI_API_KEY`, but you should still fix the environment variable for future runs.

---

## Configuration & Credentials
### Environment Variables
- `OPENAI_API_KEY` – OpenAI credential (detected automatically even if misspelled as `OPEANAI_API_KEY`).
- `SCI_MODEL` – Default LLM name when `--model` is omitted.
- `GOOGLE_API_KEY` / `GEMINI_API_KEY` – Required for Gemini access (optionally pair with `GOOGLE_API_PROXY`).
- `OSS120B_API_KEY` – API key for a self-hosted OSS-120B endpoint.

### JSON Configuration File
Copy `config_example.json` to `config.json`, then edit the fields below. You can pass the path with `--config config.json` or persist the live settings using `--save-config config.json`. Representative options include:

| Field | Purpose |
| --- | --- |
| `quality_threshold` | Minimum reviewer score before acceptance (default 0.85). |
| `max_iterations` | Maximum review/revision loops (default 15). |
| `simulation_timeout` / `latex_timeout_base` | Upper bounds for running simulations and compiling LaTeX. |
| `figure_validation` / `reference_validation` | Toggles for strict authenticity checks. |
| `default_model`, `review_model`, `revision_model`, `brainstorm_model` | Choose dedicated models for each phase. |
| `num_brainstorm_ideas`, `top_brainstorm_ideas` | Control how many ideas to generate and keep. |
| `oss120b_endpoint`, `oss120b_api_key`, `google_api_key`, `google_api_proxy` | Credentials for alternative providers. |
| `latex_auto_fix`, `fast_ref_check`, `fallback_models` | Advanced behaviors for faster iteration or better resilience.

> When a Gemini model is requested without `google_api_key`, the workflow raises a descriptive error and explains how to configure the credential.

---

## Running the Workflow
### Minimal Example
```bash
python sciresearch_workflow.py \
  --topic "Quantum-resistant encryption" \
  --field "Cryptography" \
  --question "How can lattice methods improve post-quantum key exchange?"
```
This command creates `output/<timestamped_project>/` with a full LaTeX manuscript, simulation code, simulation outputs, and review logs.

### Offline Demo (no API keys needed)
```bash
python sciresearch_workflow.py \
  --topic "Autonomous alignment curricula" \
  --field "AI Safety" \
  --question "How can curriculum learning improve alignment heuristics?" \
  --offline-demo
```
Offline runs fabricate a lightweight simulation locally and still build the complete artifact bundle so you can inspect workflow structure without external dependencies.

### Revising Existing Projects
Use `--modify-existing` to point the workflow at an already generated directory. The orchestrator will respect the current `paper.tex`/`simulation.py`, run validation, and apply revisions in-place while archiving superseded drafts if `--strict-singletons` remains enabled.

### Helpful CLI Flags
| Flag | Description |
| --- | --- |
| `--output-dir` | Root directory where project subfolders are created. |
| `--quality-threshold` / `--max-iterations` | Override acceptance criteria per run. |
| `--check-references` / `--skip-reference-check` | Force on/off external citation validation. |
| `--validate-figures` / `--skip-figure-validation` | Control expensive figure existence checks. |
| `--enable-ideation` / `--skip-ideation` | Toggle brainstorming phase; `--num-ideas` sets the breadth. |
| `--review-model`, `--revision-model`, `--brainstorm-model` | Override phase-specific models without editing config. |
| `--latex-auto-fix` | Enable automatic compile-fix loop for stubborn LaTeX drafts. |
| `--fast-ref-check` | Use lightweight LLM heuristics before paying for full DOI validation. |
| `--python-exec` | Select the interpreter used for running `simulation.py`. |
| `--user-prompt` | Inject high-priority instructions (e.g., custom style guides). |
| `--offline-demo` | Run the deterministic demo pipeline with no external API calls.

---

## Outputs & Artifacts
Each run (online or offline) creates a timestamped folder under `output/` containing:
- `paper.tex` – Self-contained LaTeX manuscript with embedded bibliography and figures.
- `simulation.py` – Executable code used to produce every numerical result.
- `simulation_outputs/` – JSON/CSV/plot artifacts produced during simulation execution.
- `logs/` – Structured workflow logs, reviewer comments, and quality metrics for auditing.
- Optional `ideation_summary.json`, `reviews.json`, and intermediate files that make it easy to trace back every decision.

Global logs are also written to `logs/workflow_<timestamp>.log`, making it trivial to debug when combined with the per-project data.

---

## Quality Gates & Validation
AI-Scientist enforces strict acceptance criteria before a project is marked complete:
- **Reference authenticity** – Regex heuristics and DOI lookups detect placeholders or suspicious references and block publication until fixed.
- **Self-contained visuals** – Review prompts demand TikZ/PGFPlots/tabular figures exclusively, rejecting any `\includegraphics` pointing to external files.
- **Compilation assurance** – `utils.latex_tools.compile_with_autofix` can repair minor LaTeX errors automatically when `--latex-auto-fix` is enabled, ensuring the final PDF builds cleanly.
- **Simulation fidelity** – `utils.sim_runner` guarantees only one `paper.tex`/`simulation.py` per project, extracts runnable code, executes it (with retries/fixes), and summarizes outputs for reviewers.
- **Parallel checks** – `utils.parallel_checks` runs DOI validation, figure counting, and LaTeX hygiene checks alongside LLM reviewers for defense-in-depth.

---

## Troubleshooting & FAQ
- **Compilation errors** – Re-run with `--latex-auto-fix` to enable the automated fixer, or inspect `logs/workflow_*.log` for the exact TeX engine output.
- **Credential issues** – Ensure `OPENAI_API_KEY` is exported; the CLI prints a warning if it only finds the misspelled `OPEANAI_API_KEY`. For Gemini, set `GOOGLE_API_KEY` or add it to your config file.
- **Slow DOI validation** – Lower `doi_rate_limit_delay` or temporarily pass `--fast-ref-check` to switch to heuristic screening before running full checks.
- **Re-running simulations** – Delete `simulation_outputs/` inside a project folder or pass `--modify-existing` so the orchestrator recomputes results and overwrites stale assets.

---

## Development & Contributions
1. **Set up tooling** – Install dev dependencies (same as runtime) and ensure `pytest` is available.
2. **Run tests** – Execute `pytest` from the repo root to validate helper utilities and regression suites (see `tests/`).
3. **Coding guidelines** – Follow the single-file LaTeX and authentic-reference rules described in `docs/`. Keep modules small, prefer pure functions, and avoid adding new runtime dependencies without discussion.
4. **Pull requests** – Include reproduction steps, mention any new CLI flags or config fields, and update this README plus `docs/` when behavior changes.

Community issues and feature requests are welcome through GitHub discussions or pull requests.

---

## Citation
If AI-Scientist contributes to a publication, please cite it as:
```bibtex
@software{ai_scientist,
  author = {AI-Scientist Developers},
  title = {AI-Scientist},
  year = {2025},
  url = {https://github.com/AI-Scientist/AI-Scientist}
}
```
Replace the URL or year with the specific release or commit you used.
