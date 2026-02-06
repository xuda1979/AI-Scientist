# AI Scientist VS Code GUI

A custom GUI for the AI Scientist workflow based on VS Code OSS architecture, featuring iteration control.

## Key Feature

**Iteration Counter**: The primary addition to the standard VS Code interface is a prominent iteration counter that allows users to specify exactly how many refinement cycles the AI Scientist should perform.

## Features

- 🔄 **Iteration Control**: Set the number of workflow iterations (1-100)
- 🤖 **Multi-Model Support**: GPT-5, GPT-4, Claude 4, Gemini models
- 📊 **Real-time Progress**: Live progress tracking and console output
- 🎨 **VS Code Dark Theme**: Familiar VS Code interface
- ⚡ **Lightweight**: Based on Electron, minimal dependencies

## Installation

```powershell
cd ai-scientist-vscode-gui
npm install
```

## Development

```powershell
# Compile TypeScript
npm run compile

# Watch mode (auto-compile on changes)
npm run watch

# Run the application
npm start

# Or run in development mode
npm run dev
```

## Usage

1. Launch the application
2. **Set the number of iterations** (the key feature!)
3. Select your AI model
4. Enter your research topic
5. Specify output directory
6. Click "Start Workflow"

## Architecture

```
ai-scientist-vscode-gui/
├── src/
│   └── main.ts          # Electron main process
├── renderer/
│   ├── index.html       # Main UI
│   ├── styles.css       # VS Code-inspired styling
│   └── renderer.js      # Frontend logic
├── package.json
└── tsconfig.json
```

## Integration with AI Scientist

This GUI integrates with the existing `sciresearch_workflow.py` by:
- Passing the iteration count as a command-line argument
- Configuring the model selection
- Managing output directories
- Displaying real-time progress

## Difference from VS Code OSS

The **only** difference from standard VS Code is the addition of the **iteration counter input** in the UI, which allows precise control over how many times the AI Scientist workflow should iterate on the research paper.

## Building for Production

```powershell
# Build for Windows
npm run build:win

# Build for macOS
npm run build:mac

# Build for Linux
npm run build:linux
```

## License

MIT (same as VS Code OSS)
