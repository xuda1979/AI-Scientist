# Enhanced GUI - Visual Structure

```
┌─────────────────────────────────────────────────────────────────────┐
│ AI Scientist - Enhanced Research Workflow                           │
├─────────────────────────────────────────────────────────────────────┤
│                                                                       │
│ ┌───────────┬───────────┬──────────┬────────────┐                  │
│ │ Workflow  │  Review   │   Chat   │ API Config │ ◄── Tabs         │
│ │  (Active) │  Options  │          │            │                   │
│ └───────────┴───────────┴──────────┴────────────┘                  │
│                                                                       │
│ ┌─── Workflow Tab ──────────────────────────────────────────────┐  │
│ │                                                                 │  │
│ │ ┏━━━ Project Details ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓  │  │
│ │ ┃ Topic: [___________________________________________]      ┃  │  │
│ │ ┃ Field: [___________________________________________]      ┃  │  │
│ │ ┃ Research Question: [_______________________________]      ┃  │  │
│ │ ┃ Document Type: [auto ▼]    Model: [gpt-5-pro      ]      ┃  │  │
│ │ ┃ Output Dir: [___________________] [Browse...]            ┃  │  │
│ │ ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛  │  │
│ │                                                                 │  │
│ │ ┏━━━ Execution Settings ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓  │  │
│ │ ┃ Request Timeout: [3600]  Max Retries: [3]               ┃  │  │
│ │ ┃ Max Iterations: [4]                                      ┃  │  │
│ │ ┃ ☐ Modify Existing  ☑ Enforce Single Files              ┃  │  │
│ │ ┃ ☐ Disable Blueprint Planning                            ┃  │  │
│ │ ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛  │  │
│ │                                                                 │  │
│ │ ┏━━━ Quality & Validation ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓  │  │
│ │ ┃ Quality Threshold: [1.0]                                 ┃  │  │
│ │ ┃ ☑ Check References  ☑ Validate Figures                 ┃  │  │
│ │ ┃ ☐ Enable PDF Review  ☑ Enable Ideation                ┃  │  │
│ │ ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛  │  │
│ │                                                                 │  │
│ │ ┏━━━ Advanced Options ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓  │  │
│ │ ┃ ☐ Disable Content Protection  ☐ Auto-Approve Changes   ┃  │  │
│ │ ┃ Content Protection Threshold: [0.15]                     ┃  │  │
│ │ ┃ ☑ Save Output Diffs                                     ┃  │  │
│ │ ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛  │  │
│ │                                                                 │  │
│ │ ┏━━━ Custom User Prompt ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓  │  │
│ │ ┃ [                                                        ]┃  │  │
│ │ ┃ [                                                        ]┃  │  │
│ │ ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛  │  │
│ │                                                                 │  │
│ │ ┏━━━ Workflow Output ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓  │  │
│ │ ┃ [INFO] Starting workflow...                             ┃  │  │
│ │ ┃ [INFO] Generating initial draft...                      ┃  │  │
│ │ ┃ [INFO] Running simulation...                            ┃  │  │
│ │ ┃ ...                                                      ┃  │  │
│ │ ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛  │  │
│ └─────────────────────────────────────────────────────────────────┘  │
│                                                                       │
├─────────────────────────────────────────────────────────────────────┤
│ Status: Idle                     [Show Error Log] [Cancel] [Run]    │
└─────────────────────────────────────────────────────────────────────┘


┌─────────────────────────────────────────────────────────────────────┐
│ Review Options Tab                                                   │
├─────────────────────────────────────────────────────────────────────┤
│                                                                       │
│ ┏━━━ Review Items to Check ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓     │
│ ┃ Select which aspects to focus on during review:              ┃     │
│ ┃                                                               ┃     │
│ ┃ ☐ Paper Structure         ☐ Content Quality                ┃     │
│ ┃ ☐ Methodology             ☐ Results & Analysis             ┃     │
│ ┃ ☐ References              ☐ Figures & Tables               ┃     │
│ ┃ ☐ Writing Quality         ☐ Novelty & Contribution         ┃     │
│ ┃ ☐ Reproducibility         ☐ Statistical Rigor              ┃     │
│ ┃                                                               ┃     │
│ ┃ If nothing selected: Comprehensive general review            ┃     │
│ ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛     │
│                                                                       │
│ ┏━━━ Review/Revision Execution Mode ━━━━━━━━━━━━━━━━━━━━━━━┓     │
│ ┃ ◉ Combined (Single API Call)                                 ┃     │
│ ┃   Review and revision in one message                         ┃     │
│ ┃                                                               ┃     │
│ ┃ ○ Separated (Two API Calls)                                  ┃     │
│ ┃   Review first, then revision in separate calls              ┃     │
│ ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛     │
│                                                                       │
│ ┏━━━ Custom Review Instructions ━━━━━━━━━━━━━━━━━━━━━━━━━━━┓     │
│ ┃ [Add specific instructions for the reviewer...              ]┃     │
│ ┃ [                                                            ]┃     │
│ ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛     │
└─────────────────────────────────────────────────────────────────────┘


┌─────────────────────────────────────────────────────────────────────┐
│ Chat Tab                                                             │
├─────────────────────────────────────────────────────────────────────┤
│                                                                       │
│ ┏━━━ Chat History ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓     │
│ ┃ You: How do I improve my paper's methodology section?       ┃     │
│ ┃                                                               ┃     │
│ ┃ Assistant: To improve your methodology section, consider:    ┃     │
│ ┃ 1. Clearly state your research design...                     ┃     │
│ ┃ 2. Describe data collection methods...                       ┃     │
│ ┃ 3. Explain analysis procedures...                            ┃     │
│ ┃                                                               ┃     │
│ ┃ You: What about statistical tests?                           ┃     │
│ ┃                                                               ┃     │
│ ┃ Assistant: For comparing models, consider...                 ┃     │
│ ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛     │
│                                                                       │
│ Your message:                                                         │
│ ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓     │
│ ┃ [Type your message here...                                   ]┃     │
│ ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛     │
│                                                                       │
│ Model: [gpt-5-pro ▼]        [Clear History]  [Send]                 │
└─────────────────────────────────────────────────────────────────────┘


┌─────────────────────────────────────────────────────────────────────┐
│ API Config Tab                                                       │
├─────────────────────────────────────────────────────────────────────┤
│                                                                       │
│ ┏━━━ OpenAI API Configuration ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓     │
│ ┃ Status: Connected (••••Ab3D)                                  ┃     │
│ ┃ Added: 2025-12-12 • Last used: Never • Models: 42            ┃     │
│ ┃                                                               ┃     │
│ ┃ [Connect OpenAI...]  [Disconnect]                            ┃     │
│ ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛     │
│                                                                       │
│ ┏━━━ Yunwu API Configuration ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓     │
│ ┃ Status: Not configured                                        ┃     │
│ ┃                                                               ┃     │
│ ┃ ☐ Enable Yunwu API (OpenAI-compatible endpoint)             ┃     │
│ ┃                                                               ┃     │
│ ┃ API Key: [********************************]                   ┃     │
│ ┃ Base URL: [https://yunwu.ai/v1                  ]            ┃     │
│ ┃                                                               ┃     │
│ ┃ [Test Connection]                                             ┃     │
│ ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛     │
│                                                                       │
│ ┏━━━ Environment Variables ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓     │
│ ┃ The GUI automatically uses these if set:                     ┃     │
│ ┃ • OPENAI_API_KEY - OpenAI API key                           ┃     │
│ ┃ • YUNWU_API_KEY - Yunwu API key                             ┃     │
│ ┃ • YUNWU_API_BASE - Yunwu API base URL                       ┃     │
│ ┃ • SCI_MODEL - Default model                                 ┃     │
│ ┃                                                               ┃     │
│ ┃ Current Environment:                                          ┃     │
│ ┃ ┌─────────────────────────────────────────────────────────┐ ┃     │
│ ┃ │ OPENAI_API_KEY: sk-proj-...Ab3D                         │ ┃     │
│ ┃ │ YUNWU_API_KEY: Not set                                  │ ┃     │
│ ┃ │ SCI_MODEL: gpt-5-pro                                    │ ┃     │
│ ┃ └─────────────────────────────────────────────────────────┘ ┃     │
│ ┃                                                               ┃     │
│ ┃ [Refresh Environment Info]                                    ┃     │
│ ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛     │
└─────────────────────────────────────────────────────────────────────┘


┌─────────────────────────────────────────────────────────────────────┐
│ Error Log Window (Popup)                                            │
├─────────────────────────────────────────────────────────────────────┤
│                                                                       │
│ ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓     │
│ ┃ [2025-12-12 10:30:15] ERROR: API connection failed           ┃     │
│ ┃ Traceback (most recent call last):                           ┃     │
│ ┃   File "workflow.py", line 123, in run                       ┃     │
│ ┃     response = api.call()                                     ┃     │
│ ┃ ConnectionError: Network unreachable                          ┃     │
│ ┃                                                               ┃     │
│ ┃ [2025-12-12 10:31:20] ERROR: Invalid output directory       ┃     │
│ ┃ PermissionError: [Errno 13] Permission denied: '/output'     ┃     │
│ ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛     │
│                                                                       │
│                          [Clear Errors]  [Close]                     │
└─────────────────────────────────────────────────────────────────────┘
```

## Key Features Highlighted

### 1. Workflow Tab
- Project configuration (topic, field, question)
- Execution settings (timeout, retries, iterations)
- Quality & validation options
- Advanced features (content protection, scaling, etc.)
- Custom prompts
- Real-time output log

### 2. Review Options Tab  
- 10 checkboxes for specific review aspects
- Radio buttons for combined/separated modes
- Custom review instructions field
- Clear descriptions and guidance

### 3. Chat Tab
- Scrollable chat history
- Color-coded messages (blue=user, green=assistant)
- Message input area
- Model selection dropdown
- Clear history button

### 4. API Config Tab
- OpenAI connection status and controls
- Yunwu API configuration with test button
- Environment variable viewer
- Refresh button for env vars

### 5. Error Log
- Popup window showing all errors
- Stack traces for debugging
- Timestamp and error type
- Clear errors button

## Navigation
- **Tabs:** Click to switch between sections
- **Buttons:** Click to perform actions
- **Fields:** Type or select values
- **Checkboxes:** Toggle options on/off
- **Scroll:** All content areas scrollable

## Status Bar
- **Left:** Current workflow status
- **Center:** Error log button
- **Right:** Cancel and Run buttons
