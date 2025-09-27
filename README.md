# AI Scientist Workflow

AI Scientist is an automated research assistant that plans, drafts, and iteratively polishes publication-ready papers. The workflow orchestrates ideation, blueprint planning, drafting, simulation, review, and rigorous quality validation so that every run produces a coherent LaTeX manuscript backed by executable evidence.

## Key Capabilities

- **Blueprint-guided drafting** – automatically produces a structured research blueprint before writing begins. The draft generator is required to follow all [CRITICAL] and [IMPORTANT] items so sections, experiments, and visuals align with the plan.
- **Document-aware prompting** – adapts prompts to the detected document type (research paper, survey, engineering report, etc.) and field-specific conventions.
- **Simulation integration** – extracts, executes, and validates `simulation.py` blocks so figures and tables are tied to reproducible code.
- **Iterative quality loop** – combines automated reviews, revision prompts, LaTeX compilation, and statistical rigor checks until quality thresholds are met.
- **Content protection & auditing** – preserves previous drafts, enforces reference authenticity, and logs every change with optional diff artifacts.

## Repository Layout

```
├── main.py / sciresearch_workflow.py      # CLI entry points and orchestration
├── src/                                  # Modular workflow engine used by GUI and APIs
│   ├── ai/                               # Chat interfaces and retry logic
│   ├── core/                             # Config, logging, and workflow façade
│   ├── processing/                       # LaTeX and simulation helpers
│   └── legacy/                           # Historical monolithic workflow
├── workflow_steps/                       # High-level pipeline phases (ideation, planning, drafting, review)
├── prompts/                              # Prompt templates, blueprint planner, review enhancements
├── quality_enhancements/                 # Validators, scoring metrics, hallucination guards
├── ui/                                   # Desktop GUI built on the same workflow API
├── out/ & output/                        # Example results and persisted run artifacts
└── docs/                                 # Design notes and implementation walkthroughs
```

## Installation

```bash
# (Optional) create a virtual environment
python -m venv .venv
source .venv/bin/activate

# Install dependencies and the workflow package
pip install -U pip
pip install .
```

The CLI entry point `sciresearch-workflow` becomes available after installation. For development you can also run `python sciresearch_workflow.py` directly from the repository root.

## Command Line Usage

```bash
sciresearch-workflow "Neural Architecture Search" "Computer Science" \
  "How can we reduce hardware cost while preserving accuracy?" \
  --output-dir out/nas_study --max-iterations 3
```

Commonly used flags:

| Flag | Description |
| --- | --- |
| `--disable-blueprint-planning` | Skip the new planning stage and draft immediately (default is to enable planning). |
| `--draft-candidates N` | Generate `N` initial drafts with test-time compute scaling and select the highest quality version. |
| `--use-test-time-scaling` | Enable multi-candidate revisions in later iterations. |
| `--skip-ideation` | Use the provided topic/question without generating new ideas. |
| `--enable-pdf-review` | Provide compiled PDFs to the reviewer model for layout-aware feedback. |
| `--skip-reference-check` | Bypass external reference validation when you trust the citations. |

Run `sciresearch-workflow --help` to see every option, including configuration file support and content protection controls.

## Workflow Phases

1. **Ideation (optional)** – Explores multiple angles for the topic and records the selected concept in `ideation_analysis.txt`.
2. **Blueprint Planning** – Uses `prompts/planning.py` to create a Markdown blueprint with section priorities, experiment plans, and a checklist. Saved as `research_blueprint.md` in the project directory.
3. **Initial Drafting** – `workflow_steps/initial_draft.py` combines the blueprint, document-type prompt, and user guidance to generate a LaTeX manuscript. When test-time scaling is enabled, multiple drafts are scored by `_evaluate_initial_draft_quality` and the best candidate is chosen.
4. **Simulation Synchronization** – `simulation.py` is extracted, executed, and its outputs summarized. Failures halt progression so code stays reproducible.
5. **Review & Revision Loop** – Each iteration compiles LaTeX, runs the quality validator, gathers reviewer feedback, and applies revisions. The loop stops early if the quality score plateaus or after `--max-iterations` passes.
6. **Quality Validation** – `quality_enhancements/quality_validator.py` enforces statistical rigor, formatting rules, citation checks, and optional content protection thresholds before finalizing the paper.

## Outputs

Each run creates a timestamped project directory (or reuses an existing one when `--modify-existing` is set) containing:

- `paper.tex` – the evolving manuscript.
- `simulation.py` – executable experiment script extracted from the LaTeX source.
- `research_blueprint.md` – the planning artifact that guided drafting.
- `logs/` – timestamped workflow logs.
- `diffs/` (optional) – review-to-review LaTeX diffs when `--output-diffs` is enabled.

## Graphical Interface

Launch the Tkinter-based GUI with:

```bash
sciresearch-workflow-gui
```

The GUI mirrors every CLI option, including blueprint planning, and provides live log streaming with a cancel button.

## Extending the Workflow

- Add new document templates in `document_types.py` and corresponding prompt logic in `document_prompts.py`.
- Introduce additional validators under `quality_enhancements/` and register them inside `quality_validator.py`.
- Customize prompts (or add organization-specific requirements) in the `prompts/` package. The blueprint planner can be adapted for new research domains by editing `prompts/planning.py`.

Pull requests and research contributions are welcome—see `CONTRIBUTING.md` if available or open an issue describing your enhancement idea.
