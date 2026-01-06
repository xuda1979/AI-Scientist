# AI Scientist Research GUI

A VSCode-style graphical interface for autonomous AI research with two powerful modes.

## Quick Start

```bash
# Launch the GUI
python launch_research_gui.py

# Or on Windows, double-click:
Launch_Research_GUI.bat
```

## Features

### 🤖 Auto Mode (Fully Autonomous)
- Agent autonomously completes the entire research workflow
- Just set your topic, field, and the number of iterations
- The AI handles ideation, drafting, reviewing, and revision
- Perfect for generating complete research papers hands-free

### 💬 Interactive Mode (User-Controlled)
- Guide the research with custom prompts at each step
- Control each iteration and provide feedback
- Similar to VS Code + Copilot experience
- Input your specific research questions or instructions

### 🔄 Iteration Control
- Slider to set the number of research iterations (1-20)
- Quick presets:
  - **Quick (2)**: Fast generation, basic quality
  - **Standard (4)**: Good balance of speed and quality
  - **Thorough (8)**: Detailed research with multiple revisions
  - **Deep (15)**: Comprehensive research with extensive refinement

### 📝 Research Prompt Templates
- **Survey**: Comprehensive literature surveys
- **Novel Method**: Propose new methodologies
- **Benchmark**: Create evaluation benchmarks
- **Analysis**: Critical analysis of approaches

### 💬 Chat Interface
- Real-time chat with AI research assistant
- Quick action buttons for common tasks:
  - Generate research ideas
  - Suggest methodologies
  - Find related work
  - Identify research gaps
- Context-aware responses based on your research topic

### 📁 File Explorer
- VSCode-style file tree
- Open and edit project files directly
- Automatic project folder detection

### 📊 Output Panel
- Real-time progress updates
- Color-coded messages (info, success, error, warning)
- Detailed logging of each workflow step

## Layout

```
┌─────────────────────────────────────────────────────────────────┐
│  File  │  Run                                                   │
├────────┼─────────────────────────────┬──────────────────────────┤
│        │                             │                          │
│  📁    │                             │  Workflow Tab            │
│        │     File Editor             │  ├── Mode Selection      │
│ File   │                             │  ├── Iteration Control   │
│ Tree   │                             │  ├── Research Prompt     │
│        │                             │  └── Run/Cancel          │
│        ├─────────────────────────────┤                          │
│        │  📋 Output                  │  Chat Tab                │
│        │  Real-time workflow logs    │  └── AI Assistant        │
│        │                             │                          │
├────────┴─────────────────────────────┴──────────────────────────┤
│  Ready │ Mode: Auto │ Iterations: 4                             │
└─────────────────────────────────────────────────────────────────┘
```

## Usage Guide

### 1. Select Research Mode

**Auto Mode** (Default):
- Best for generating complete papers from scratch
- Set your topic and let the AI work

**Interactive Mode**:
- Best when you want to guide the research
- Provide custom prompts and feedback

### 2. Configure Iterations

Choose based on your needs:
- Quick experiments: 2 iterations
- Standard papers: 4 iterations  
- Publication-ready: 8+ iterations

### 3. Enter Research Prompt

Write your research question or use a template:
```
How can we improve the performance of large language models 
for scientific research tasks?
```

### 4. Set Project Directory

Either:
- Click "Open Folder" in the file tree to select an existing project
- Or set the directory in the configuration dialog

### 5. Start Research

Click "▶️ Start Research" (or press F5) to begin.

Watch the output panel for real-time progress:
```
==========================================
🚀 Starting AUTO Mode Research
📊 Max Iterations: 4
📁 Output Directory: ./my_research
📝 Research Prompt: How can we improve...
==========================================

Step 1: Generating research blueprint...
Step 2: Creating initial draft...
...
```

## Keyboard Shortcuts

| Shortcut | Action |
|----------|--------|
| `Ctrl+S` | Save current file |
| `Ctrl+O` | Open folder |
| `F5` | Start research |
| `Ctrl+Enter` | Send chat message |

## API Configuration

The GUI uses the same API configuration as the CLI:
- Set `OPENAI_API_KEY` environment variable
- Or configure in the "Models & API" tab

## Tips

1. **Start with Auto Mode** if you're new - it handles everything automatically

2. **Use Interactive Mode** when you have specific ideas or want to refine the output

3. **Higher iterations** produce better results but take longer

4. **Use the Chat** to brainstorm ideas before starting a workflow

5. **Check the Output panel** for detailed progress and any errors

## Troubleshooting

### GUI won't start
```bash
# Check Python version (3.8+ required)
python --version

# Install tkinter if missing
pip install tk
```

### API errors
- Verify your API key is set correctly
- Check network connectivity
- Try a different model in the configuration

### Workflow hangs
- Check the output panel for error messages
- Try reducing the iteration count
- Ensure your API has sufficient quota

## Related Documentation

- [CLI Usage Guide](./README.md)
- [Workflow Configuration](./ENHANCED_WORKFLOW_DOCUMENTATION.md)
- [API Setup](./config_example.json)
