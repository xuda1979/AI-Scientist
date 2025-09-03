# Custom Prompt Feature Documentation

## 🎯 Overview
The Enhanced SciResearch Workflow now supports **custom user prompts** that take priority over standard requirements when conflicts arise. This allows users to specify their preferences for writing style, focus areas, or specific requirements.

## ✨ Key Features

### 1. **Interactive Prompt Input**
When no `--user-prompt` is provided via command line, the system prompts the user interactively:

```
🎯 CUSTOM PROMPT INPUT
============================================================
You can provide a custom prompt that will be integrated into all AI interactions.
This prompt will take priority over standard requirements when conflicts arise.
Examples:
  - 'Focus on mathematical rigor and formal proofs'
  - 'Emphasize practical applications and real-world examples'
  - 'Use a conversational writing style suitable for broader audiences'
  - 'Include extensive experimental validation and statistical analysis'

Leave empty to use standard prompts only.
------------------------------------------------------------
Enter your custom prompt (or press Enter to skip):
```

### 2. **Command Line Integration**
Users can also specify the custom prompt directly via command line:

```bash
python sciresearch_workflow.py \
  --topic "Quantum Error Correction" \
  --user-prompt "Focus on mathematical rigor and include detailed proofs for all theoretical claims"
```

### 3. **Priority System**
The custom prompt takes precedence over standard requirements in case of conflicts:

```
PRIORITY INSTRUCTION FROM USER: Focus on mathematical rigor and formal proofs

The above user instruction takes precedence over any conflicting requirements below.
However, still maintain the critical technical requirements (single file, embedded references, compilable LaTeX).
```

## 🔧 Implementation Details

### Prompt Integration Points
The custom prompt is integrated into **all** AI interaction functions:

1. **Initial Draft** (`_initial_draft_prompt()`)
2. **Review** (`_review_prompt()`)
3. **Editorial Decision** (`_editor_prompt()`)
4. **Revision** (`_revise_prompt()`)

### Technical Requirements Preserved
Even with custom prompts, these critical requirements are always maintained:
- ✅ Single LaTeX file (no separate bibliography files)
- ✅ Embedded references
- ✅ Compilable with pdflatex
- ✅ Results documentation in results.txt
- ✅ Figure generation in simulation.py

## 📋 Usage Examples

### Example 1: Mathematical Focus
```bash
python sciresearch_workflow.py \
  --topic "Algorithm Optimization" \
  --user-prompt "Emphasize theoretical foundations with formal complexity analysis and mathematical proofs"
```

### Example 2: Practical Applications
```bash
python sciresearch_workflow.py \
  --topic "Machine Learning Security" \
  --user-prompt "Focus on real-world applications and include extensive experimental validation with industry datasets"
```

### Example 3: Accessible Writing Style
```bash
python sciresearch_workflow.py \
  --topic "Quantum Computing" \
  --user-prompt "Use clear, accessible language suitable for interdisciplinary audiences while maintaining technical accuracy"
```

### Example 4: Survey Paper Style
```bash
python sciresearch_workflow.py \
  --topic "Blockchain Scalability" \
  --user-prompt "Structure as a comprehensive survey with systematic classification and comparative analysis of existing solutions"
```

## 🎨 Custom Prompt Ideas

### Writing Style Preferences
- `"Use a conversational, engaging writing style while maintaining academic rigor"`
- `"Write in a formal, traditional academic style with precise technical language"`
- `"Balance technical depth with accessibility for broader scientific audiences"`

### Content Focus Areas
- `"Emphasize practical applications and real-world case studies"`
- `"Focus on theoretical foundations and mathematical rigor"`
- `"Prioritize experimental validation and empirical evidence"`
- `"Include extensive related work and comparative analysis"`

### Paper Type Specifications
- `"Structure as a survey paper with comprehensive literature review"`
- `"Focus on system design and implementation details"`
- `"Emphasize algorithmic contributions with complexity analysis"`
- `"Structure as an experimental paper with detailed methodology"`

### Field-Specific Requirements
- `"Include security threat models and attack scenarios for cybersecurity topics"`
- `"Emphasize clinical relevance and validation for medical applications"`
- `"Focus on scalability and performance optimization for systems papers"`
- `"Include reproducibility guidelines and open science practices"`

## 🔄 How It Works

### 1. Prompt Collection
```python
if user_prompt is None:
    # Interactive prompt collection with examples
    user_prompt = input("Enter your custom prompt: ").strip()

if user_prompt:
    print(f"✅ Custom prompt set: {user_prompt[:100]}...")
```

### 2. Prompt Integration
```python
if user_prompt:
    sys_prompt = (
        f"PRIORITY INSTRUCTION FROM USER: {user_prompt}\n\n"
        "The above user instruction takes precedence when conflicts arise.\n\n"
        + standard_sys_prompt
    )
```

### 3. Consistent Application
The custom prompt is passed to **all** AI interaction functions ensuring consistent application throughout the workflow.

## ⚖️ Conflict Resolution

### Priority Order
1. **🥇 User Custom Prompt** - Highest priority
2. **🥈 Technical Requirements** - Must be maintained
3. **🥉 Standard Workflow Prompts** - Applied when no conflicts

### Example Conflict Resolution
If user says: `"Use informal language and skip mathematical proofs"`
- ✅ **Informal language** - Applied (user preference)
- ❌ **Skip proofs** - Overridden if paper type requires mathematical rigor
- ✅ **Technical requirements** - Always maintained

## 📊 Benefits

### 1. **Flexibility**
- Adapt to different paper types and requirements
- Accommodate various writing styles and preferences
- Support diverse research areas and methodologies

### 2. **User Control**
- Direct influence over paper style and focus
- Priority system ensures user preferences are respected
- Interactive guidance with helpful examples

### 3. **Consistency**
- Custom prompt applied across all workflow stages
- Maintains coherent style throughout iterations
- Balances user preferences with quality requirements

### 4. **Quality Assurance**
- Critical technical requirements always preserved
- Quality validation still applies
- Reference authenticity and compilation checks maintained

## 🚀 Getting Started

### Quick Start (Interactive Mode)
```bash
python sciresearch_workflow.py --topic "Your Research Topic"
# System will prompt for custom prompt interactively
```

### Command Line Mode
```bash
python sciresearch_workflow.py \
  --topic "Your Research Topic" \
  --user-prompt "Your custom requirements and preferences"
```

### Skip Custom Prompt
```bash
python sciresearch_workflow.py --topic "Your Research Topic"
# Press Enter when prompted to use standard prompts only
```

The custom prompt feature provides unprecedented flexibility while maintaining the workflow's quality assurance and technical requirements. Users can now shape their papers according to specific needs, preferences, and target audiences while benefiting from the comprehensive validation and iteration system.
