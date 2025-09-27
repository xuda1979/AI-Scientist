# SciResearch Workflow

A streamlined, modular research paper generation workflow with AI assistance.

## Features

- **Modular Architecture**: Clean separation of concerns with organized source code
- **Professional Logging**: Unicode-safe logging without emojis for clean terminal output
- **GPT-5 Compatible**: Support for latest OpenAI models with automatic parameter handling
- **LaTeX Processing**: Automatic compilation and error handling
- **Quality Assessment**: Iterative improvement with comprehensive scoring
- **Ideation System**: Generate and analyze multiple research ideas with scoring

## Quick Start

```bash
# Install the workflow as a package (run from the repository root)
pip install .

# Generate a new research paper
sciresearch-workflow "Neural Networks" "AI" "How to improve training efficiency?" --output-dir out/neural_nets

# Use GPT-5 with single iteration
sciresearch-workflow "Quantum Computing" "Physics" "Quantum advantage?" --model gpt-5 --max-iterations 1 --output-dir out/quantum

# Modify existing paper
sciresearch-workflow --modify-existing --output-dir out/existing_paper --max-iterations 2
```

### GUI Usage

Prefer a graphical interface? Launch the desktop application after installation:

```bash
sciresearch-workflow-gui
```

The GUI provides forms for every workflow argument, including checkboxes for boolean flags
(`--skip-ideation`, `--enable-pdf-review`, `--disable-content-protection`, and more), numeric
spinboxes for iteration counts and thresholds, and dropdowns populated from
`document_types.get_available_document_types()`. Use the directory pickers to choose an output
location, then monitor progress in the live log viewer while the workflow runs on a background
thread. A Cancel button signals the workflow to stop after the current phase completes.

> **Note:** Tkinter ships with the standard Python distribution on Windows, macOS, and most
> Linux environments. If your Python build omits Tkinter, install the appropriate package from
> your system package manager (for example, `sudo apt install python3-tk`).

## Project Structure

```
├── main.py                    # Main entry point
├── src/                       # Core modular architecture
│   ├── core/                  # Core workflow components
│   ├── ai/                    # AI interface modules
│   ├── latex/                 # LaTeX processing
│   ├── quality/              # Quality assessment
│   └── utils/                # Utility functions
├── out/                      # Generated output papers
├── docs/                     # Documentation
├── utils/                    # Legacy utilities
├── pyproject.toml            # Project metadata and dependencies
└── requirements.txt          # Legacy dependency list
```

## Command Line Options

- `--model`: AI model to use (default: gpt-4)
- `--max-iterations`: Maximum revision iterations (default: 4)
- `--output-dir`: Output directory for generated papers
- `--skip-ideation`: Skip the ideation phase
- `--user-prompt`: Custom prompt for AI interactions
- `--verbose`: Enable detailed logging

## Recent Updates

- ✅ Professional logging without Unicode characters
- ✅ GPT-5 API compatibility 
- ✅ Modular architecture with clean imports
- ✅ Robust LaTeX parsing and compilation
- ✅ Comprehensive ideation system

See `CHANGELOG.md` for detailed changes.

## Legacy Files

- `src/legacy/legacy_monolithic_workflow.py`: Original monolithic version (kept for reference)
- `docs/archive/`: Detailed implementation documentation

## License

See `LICENSE` file.
