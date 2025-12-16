# Enhanced AI Scientist GUI Documentation

## Overview

The Enhanced AI Scientist GUI provides a comprehensive interface for managing and executing research paper workflows. It includes all command-line features plus additional capabilities for interactive research.

## Quick Start

### Launching the GUI

**Windows:**
```bash
launch_enhanced_gui.bat
```

**Linux/Mac:**
```bash
chmod +x launch_enhanced_gui.sh
./launch_enhanced_gui.sh
```

**Alternative (any platform):**
```bash
python -m ui.enhanced_gui
```

## Features

### 1. API Key Management (Tab: API Config)

#### Environment Variables Support
The GUI automatically reads API keys from environment variables:
- `OPENAI_API_KEY` - OpenAI API key
- `YUNWU_API_KEY` - Yunwu API key (OpenAI-compatible)
- `YUNWU_API_BASE` - Yunwu API base URL
- `SCI_MODEL` - Default model to use

You can override these by entering values directly in the GUI.

#### OpenAI Configuration
1. Click "Connect OpenAI..." button
2. Enter your API key (or it will auto-detect from environment)
3. The system validates and stores it encrypted
4. Status shows: Connected/Disconnected/Invalid

#### Yunwu API Configuration (OpenAI-Compatible)
1. Check "Enable Yunwu API" checkbox
2. Enter API Key and Base URL
3. Click "Test Connection" to verify
4. When enabled, all requests route through Yunwu instead of OpenAI

### 2. Workflow Configuration (Tab: Workflow)

#### Project Details
- **Topic**: Research area of interest
- **Field**: Academic field (e.g., Computer Science, Biology)
- **Research Question**: Specific question to investigate
- **Document Type**: Choose from:
  - auto (auto-detect)
  - research_paper
  - engineering_paper
  - finance_research
  - survey_paper
  - technical_report
  - white_paper
  - conference_paper
  - journal_article
  - presentation_slides
- **Model**: AI model to use (default: gpt-5-pro)
- **Output Directory**: Where to save results

#### Execution Settings
- **Request Timeout**: Maximum time per API call (seconds)
- **Max Retries**: Number of retry attempts for failed API calls
- **Max Iterations**: Number of review-revision cycles
- **Modify Existing**: Work on existing project instead of creating new
- **Enforce Single Files**: Keep only paper.tex and simulation.py
- **Disable Blueprint Planning**: Skip research planning phase
- **Python Executable**: Custom Python interpreter path
- **Config File**: Load settings from JSON file

#### Quality & Validation
- **Quality Threshold**: Minimum quality score (0.0-1.0)
- **Check References**: Validate external references
- **Validate Figures**: Ensure figures are generated correctly
- **Enable PDF Review**: Send PDF files to AI during review
- **Enable Ideation**: Generate research ideas automatically
- **Specify Idea**: Use a specific research idea (skips ideation)
- **Number of Ideas**: How many ideas to generate (10-20)

#### Advanced Options
- **Disable Content Protection**: Turn off deletion protection (⚠️ DANGEROUS)
- **Auto-Approve Changes**: Automatically approve safe changes
- **Content Protection Threshold**: Max allowed content reduction (default: 15%)
- **Save Output Diffs**: Track changes between iterations
- **Enable Test-Time Scaling**: Use multiple revision candidates
- **Revision Candidates**: Number of parallel revisions to generate
- **Draft Candidates**: Number of initial drafts to generate
- **All-Code Mode**: Allow unrestricted code generation
- **Code Output Dir**: Directory for generated code files
- **Science-Only Mode**: Simplified workflow (diff output only)

#### Custom User Prompt
Add custom instructions that take priority over standard requirements.

#### Workflow Output
Real-time log of the workflow execution with progress updates.

### 3. Review Options (Tab: Review Options)

#### Review Items to Check
Select specific aspects for the reviewer to focus on:
- ☑ Paper Structure & Organization
- ☑ Content Quality & Depth
- ☑ Methodology & Approach
- ☑ Results & Analysis
- ☑ References & Citations
- ☑ Figures & Tables
- ☑ Writing Quality & Clarity
- ☑ Novelty & Contribution
- ☑ Reproducibility
- ☑ Statistical Rigor

**If nothing is selected**, a general comprehensive review will be performed.

**If items are selected**, the reviewer will focus specifically on those aspects.

#### Review/Revision Execution Mode

**Combined (Single API Call)** - Default
- Review and revision happen in one message
- Faster and more cost-effective
- Good for most cases

**Separated (Two API Calls)**
- Review happens first (separate message)
- Revision happens after review (separate message)
- More thorough but slower and costs 2x API calls
- Better for complex papers needing careful review

#### Custom Review Instructions
Add specific instructions for the reviewer, such as:
- "Focus on statistical methodology"
- "Ensure all figures have proper captions"
- "Check for consistency in terminology"
- "Verify mathematical notation is correct"

### 4. Chat with LLM (Tab: Chat)

Interactive chat interface for direct communication with the AI model.

#### Features
- **Chat History**: Scrollable conversation history
- **Model Selection**: Choose which model to chat with
- **Send Message**: Type and send messages to the AI
- **Clear History**: Reset the conversation

#### Use Cases
- Ask questions about your paper
- Get suggestions for improvements
- Brainstorm research ideas
- Get help with LaTeX formatting
- Debug simulation code

### 5. Error Log

Click "Show Error Log" button in the status bar to view all errors and warnings.

#### Features
- Comprehensive error tracking
- Stack traces for debugging
- Clear errors button
- Automatically captures exceptions from workflow

## Workflow Execution

### Basic Workflow
1. Configure API key (API Config tab)
2. Enter project details (Workflow tab)
3. Optionally configure review options (Review Options tab)
4. Click "Run Workflow"
5. Monitor progress in output log
6. Check error log if issues occur

### Review-Revision Process

The workflow performs multiple review-revision cycles:

**Iteration Flow:**
```
1. Generate initial draft
2. Compile LaTeX and run simulation
3. Review paper (check selected items)
4. Generate revision based on review
5. Apply changes and validate
6. Repeat steps 2-5 for max_iterations
```

**Review Customization:**
- Select specific review items to focus the reviewer
- Choose combined or separated execution mode
- Add custom review instructions

### Keyboard Shortcuts
- `Ctrl+Enter` - Start workflow (when in entry fields)
- `Escape` - Cancel workflow (when running)

## Tips & Best Practices

### API Key Management
1. **Use Environment Variables**: Set `OPENAI_API_KEY` or `YUNWU_API_KEY` in your system
2. **Override When Needed**: Enter keys in GUI to override environment
3. **Test Connection**: Always test Yunwu connection before running workflow
4. **Encrypted Storage**: OpenAI keys are stored encrypted locally

### Review Options
1. **Start General**: First run with no specific items selected
2. **Focus Second Pass**: Select specific items for targeted improvements
3. **Combined Mode**: Use for faster iterations
4. **Separated Mode**: Use when you need careful, thorough review

### Workflow Configuration
1. **Start Simple**: Use defaults for first run
2. **Increase Iterations**: Add more iterations for complex papers
3. **Enable Diffs**: Keep output diffs enabled to track changes
4. **Content Protection**: Keep enabled unless you know what you're doing

### Chat Interface
1. **Ask Specific Questions**: Get better answers with specific questions
2. **Context**: Mention your paper topic for relevant responses
3. **Model Selection**: Use same model as workflow for consistency

### Error Handling
1. **Check Error Log**: Always check error log after failed runs
2. **API Issues**: Verify API key and connection
3. **Timeout Errors**: Increase request timeout for slow models
4. **Content Issues**: Check content protection threshold

## Configuration Files

### Saving Configuration
1. Enter save path in "Save Config To" field
2. Run workflow (config is saved before execution)
3. Reuse config by loading it in "Config File" field

### Configuration Format
```json
{
  "enable_pdf_review": false,
  "reference_validation": true,
  "figure_validation": true,
  "research_ideation": true,
  "diff_output_tracking": true,
  "content_protection": true,
  "auto_approve_changes": false,
  "content_protection_threshold": 0.15,
  "use_test_time_scaling": false,
  "revision_candidates": 3,
  "initial_draft_candidates": 1,
  "quality_threshold": 1.0,
  "max_iterations": 4,
  "request_timeout": 3600,
  "max_retries": 3
}
```

## Troubleshooting

### GUI Won't Start
```bash
# Install dependencies
pip install -r requirements.txt

# Check Python version (3.8+ required)
python --version
```

### API Connection Failed
1. Verify API key is correct
2. Check internet connection
3. For Yunwu: Verify base URL is correct
4. Check firewall settings

### Workflow Errors
1. Check error log for details
2. Verify output directory permissions
3. Ensure sufficient disk space
4. Check API rate limits

### Chat Not Responding
1. Verify API connection is active
2. Check model name is correct
3. Increase timeout if needed
4. Check error log for API issues

## Command Line Parity

The Enhanced GUI includes **ALL** command-line features:
- All workflow parameters available
- Same execution logic
- Same quality checks
- Same output format

**Advantage of GUI**: 
- Easier parameter management
- Visual feedback
- Interactive chat
- Error log visualization
- No need to remember command syntax

## Advanced Features

### Test-Time Compute Scaling
Enable in Advanced Options to generate multiple revision candidates and select the best one.

### All-Code Mode
Enable to allow AI to generate any code files (not just simulation.py) and execute commands iteratively.

### Science-Only Mode
Simplified workflow that only improves scientific content and outputs git diff format (skips validation and iteration).

### Custom Prompts
Use the custom prompt field to add specific requirements or constraints that override standard workflow prompts.

## Support

For issues or questions:
1. Check error log in GUI
2. Review documentation
3. Check GitHub issues
4. Create new issue with error log

## Updates

The GUI automatically inherits all updates to the command-line workflow through the unified `workflow_wrapper` module. No manual synchronization needed.

---

**Version**: 1.0  
**Last Updated**: December 2025  
**Compatible With**: AI Scientist v2.0+
