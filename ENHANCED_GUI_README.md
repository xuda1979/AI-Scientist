# Enhanced GUI - Quick Start Guide

## 🚀 Launch the GUI

**Windows:**
```bash
launch_enhanced_gui.bat
```

**Linux/Mac:**
```bash
chmod +x launch_enhanced_gui.sh
./launch_enhanced_gui.sh
```

**Any Platform:**
```bash
python -m ui.enhanced_gui
```

## ✨ Key Features Implemented

### 1. **API Key Management with Environment Variables** ✓
- Automatically reads `OPENAI_API_KEY`, `YUNWU_API_KEY` from environment
- GUI fields to override environment variables
- Secure encrypted storage for OpenAI keys
- Yunwu API support (OpenAI-compatible endpoint)
- Test connection button for verification
- Status indicators showing connection state

### 2. **Review-Revision Configuration** ✓
- **Checkboxes for review items:**
  - Paper Structure & Organization
  - Content Quality & Depth
  - Methodology & Approach
  - Results & Analysis
  - References & Citations
  - Figures & Tables
  - Writing Quality & Clarity
  - Novelty & Contribution
  - Reproducibility
  - Statistical Rigor
- **Review mode selector:**
  - Combined (Single API Call) - faster, cheaper
  - Separated (Two API Calls) - more thorough
- **Custom review instructions field**
- If nothing selected, general review is performed

### 3. **Chat Window** ✓
- Interactive chat with LLM
- Model selection
- Chat history with color coding:
  - Blue = User messages
  - Green = Assistant responses
  - Gray = System messages
- Clear history button
- Send messages with `Enter` key

### 4. **Error Log** ✓
- Dedicated error log window
- Captures all exceptions and errors
- Stack traces for debugging
- Clear errors button
- Persistent across workflow runs

### 5. **All CLI Functionality** ✓
- Every command-line option available in GUI
- Organized in tabs for easy access
- Tooltips and descriptions
- Default values pre-filled
- Validation before execution

## 📋 Tabs Overview

### Workflow Tab
Main configuration for running research workflows:
- Project details (topic, field, question)
- Model selection
- Output directory
- Execution settings
- Quality & validation options
- Advanced features
- Custom prompts
- Real-time workflow output log

### Review Options Tab
Configure how reviews are performed:
- Select specific review items to check
- Choose combined or separated execution mode
- Add custom review instructions

### Chat Tab
Direct interaction with the LLM:
- Ask questions about your paper
- Get suggestions and feedback
- Brainstorm ideas
- Debug issues

### API Config Tab
Manage API connections:
- OpenAI API configuration
- Yunwu API configuration (OpenAI-compatible)
- Environment variable display
- Connection testing

## 🔧 Configuration

### Environment Variables (Recommended)

Set these in your system for automatic detection:

**Windows PowerShell:**
```powershell
$env:OPENAI_API_KEY = "sk-..."
$env:YUNWU_API_KEY = "yw-..."
$env:YUNWU_API_BASE = "https://yunwu.ai/v1"
$env:SCI_MODEL = "gpt-5-pro"
```

**Linux/Mac:**
```bash
export OPENAI_API_KEY="sk-..."
export YUNWU_API_KEY="yw-..."
export YUNWU_API_BASE="https://yunwu.ai/v1"
export SCI_MODEL="gpt-5-pro"
```

### GUI Override
You can override environment variables by entering values directly in the GUI fields.

## 🎯 Usage Examples

### Example 1: Basic Research Paper
1. **API Config Tab**: Connect your OpenAI API key
2. **Workflow Tab**:
   - Topic: "Transformer Architectures"
   - Field: "Computer Science"
   - Question: "How can we improve attention mechanism efficiency?"
   - Output Directory: "output/transformers"
3. Click **Run Workflow**

### Example 2: Focused Review on Methodology
1. **Review Options Tab**:
   - ☑ Check "Methodology & Approach"
   - ☑ Check "Statistical Rigor"
   - Mode: "Combined (Single API Call)"
   - Custom Instructions: "Ensure all experiments are reproducible"
2. **Workflow Tab**: Configure project
3. Click **Run Workflow**

### Example 3: Using Chat for Guidance
1. **Chat Tab**:
   - Model: "gpt-5-pro"
   - Message: "What statistical tests should I use for comparing two ML models?"
2. Click **Send**
3. Review response and apply to your paper

### Example 4: Separated Review-Revision
1. **Review Options Tab**:
   - Mode: "Separated (Two API Calls)"
   - ☑ Check specific items you want thorough review on
2. **Workflow Tab**: Configure project
3. Click **Run Workflow**
   - First API call: Review
   - Second API call: Revision based on review

## 🐛 Troubleshooting

### GUI Won't Start
```bash
pip install -r requirements.txt
```

### API Connection Issues
1. Check API key is correct
2. Verify internet connection
3. For Yunwu: Test connection with button
4. Check firewall settings

### Workflow Errors
1. Click "Show Error Log" button
2. Review stack traces
3. Check API quotas/rate limits
4. Verify output directory permissions

## 📊 Status Indicators

- **Green "Connected"**: API ready to use
- **Red "Disconnected"**: No API configured
- **Yellow "Invalid"**: API key validation failed
- **Running...**: Workflow in progress
- **✓ Completed**: Workflow finished successfully
- **✗ Failed**: Error occurred (check error log)

## ⌨️ Keyboard Shortcuts

- `Ctrl+Enter`: Start workflow (when in text field)
- `Escape`: Cancel running workflow
- `Enter` in chat: Send message

## 📝 Notes

- All parameters are validated before workflow starts
- Progress is shown in real-time in the output log
- Review files are saved in the project directory
- Diffs are tracked automatically (if enabled)
- GUI inherits ALL CLI features automatically

## 🔗 Related Documentation

- Full documentation: `ENHANCED_GUI_DOCUMENTATION.md`
- Test suite: `ui/test_enhanced_gui.py`
- Original GUI: `ui/gui_app.py` (legacy)

## ✅ Testing

Run the test suite to verify installation:
```bash
python ui/test_enhanced_gui.py
```

All tests should pass before using the GUI.

---

**Status**: ✅ Fully Implemented and Tested  
**Version**: 1.0  
**Date**: December 2025
