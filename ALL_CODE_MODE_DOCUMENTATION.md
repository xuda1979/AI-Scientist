# All-Code Mode: Unrestricted Code Generation & Iterative Execution

## Overview

The `--all-code` flag enables a powerful new mode in AI-Scientist that removes restrictions on code file generation and adds iterative command execution with feedback loops. This transforms the system from a paper-focused tool into a full-fledged codebase development assistant.

## Key Features

### 1. **Unrestricted Code Generation**
- **No Restrictions**: LLM can generate ANY code files, not just `simulation.py`
- **Multiple Languages**: Python, JavaScript, TypeScript, C++, Java, Go, Rust, R, SQL, Bash, etc.
- **Complex Structures**: Full directory trees with nested folders
- **Real Projects**: Create complete applications, libraries, frameworks

### 2. **Command Execution & Feedback**
- **Extract Commands**: Parse execution commands from LLM responses
- **Auto-Execute**: Run commands automatically in the project directory
- **Capture Output**: Log stdout, stderr, exit codes
- **Feed Back**: Send execution results to LLM in next iteration

### 3. **Iterative Development**
- **Continuous Improvement**: LLM sees execution results and fixes errors
- **Error Handling**: LLM automatically debugs based on error messages
- **Performance Tuning**: LLM optimizes based on benchmark results
- **Test-Driven**: LLM can write tests, run them, and fix failures

## Usage

### Basic Command

```bash
python main.py \
  --topic "Machine Learning Framework" \
  --field "Software Engineering" \
  --question "How to build a modular deep learning library?" \
  --all-code \
  --max-iterations 5 \
  --output-dir output/ml_framework
```

### With Custom Code Directory

```bash
python main.py \
  --all-code \
  --code-output-dir src \
  --execution-log build_log.txt \
  --modify-existing \
  --output-dir output/my_project
```

## How It Works

### Step-by-Step Workflow

#### **Iteration 1: Initial Code Generation**

1. **LLM receives prompt** with all-code instructions
2. **LLM generates code files** (e.g., `main.py`, `utils.py`, `tests.py`)
3. **System extracts code** using pattern matching
4. **System saves files** to `<project>/code/` directory
5. **LLM provides commands** (e.g., `python main.py`, `pytest tests.py`)
6. **System executes commands** and captures output
7. **Results logged** to `execution_log.txt`

#### **Iteration 2: Error Fixing**

1. **Previous execution results** sent to LLM
2. **LLM sees errors** (e.g., `ModuleNotFoundError`, syntax errors)
3. **LLM fixes code** by providing updated file contents
4. **System generates diffs** showing changes
5. **System re-runs commands** with fixed code
6. **New results** logged and sent to LLM

#### **Iteration N: Refinement**

1. **LLM reviews** successful execution
2. **LLM adds features** (e.g., new functions, optimizations)
3. **LLM adds tests** to verify correctness
4. **System runs** all code and tests
5. **Cycle continues** until max iterations or success

### Code File Format

The LLM can specify code files in multiple formats:

**Format 1: Language + Filepath**
```
```python src/neural_net.py
import numpy as np

class NeuralNetwork:
    def __init__(self, layers):
        self.layers = layers
```
```

**Format 2: File Comment**
```
File: src/utils/helper.py
```python
def normalize(data):
    return (data - data.mean()) / data.std()
```
```

**Format 3: Nested Paths**
```
```javascript src/components/App.js
import React from 'react';

export default function App() {
    return <div>Hello World</div>;
}
```
```

### Command Formats

The LLM can specify commands in several ways:

**Format 1: Execute Directive**
```
Execute: python train.py --epochs 10
```

**Format 2: Shell Block**
```bash
python preprocess.py --data dataset.csv
python train.py --model lstm --lr 0.001
python evaluate.py --checkpoint best_model.pkl
```

**Format 3: Run/Command**
```
Run: npm install
Run: npm test
Command: cargo build --release
```

## Execution Log Format

The execution log (`execution_log.txt`) contains detailed results:

```
================================================================================
COMMAND EXECUTION LOG - 2025-10-28T15:30:00
================================================================================

[1/3] Execute: python train.py --epochs 10
Command: python train.py --epochs 10
Shell: powershell
--------------------------------------------------------------------------------
Exit Code: 0

STDOUT:
Epoch 1/10: loss=0.456, acc=0.823
Epoch 2/10: loss=0.321, acc=0.887
...
Epoch 10/10: loss=0.089, acc=0.976

✓ SUCCESS

================================================================================

[2/3] Execute: python evaluate.py --model best_model.pkl
Command: python evaluate.py --model best_model.pkl
Shell: powershell
--------------------------------------------------------------------------------
Exit Code: 0

STDOUT:
Test Accuracy: 0.965
F1 Score: 0.971
Precision: 0.968
Recall: 0.974

✓ SUCCESS

================================================================================

[3/3] Execute: pytest tests/
Command: pytest tests/
Shell: powershell
--------------------------------------------------------------------------------
Exit Code: 0

STDOUT:
======================== test session starts =========================
collected 15 items

tests/test_model.py .............                             [ 86%]
tests/test_utils.py ..                                        [100%]

======================== 15 passed in 2.34s ==========================

✓ SUCCESS

================================================================================
```

## Code Diff Output

Git-style diffs are generated for all code changes:

```
Code Changes - Iteration 2
================================================================================

--- src/model.py ---
--- a/src/model.py
+++ b/src/model.py
@@ -10,7 +10,10 @@ class Model:
     def train(self, data, epochs=10):
-        # Simple training loop
-        for epoch in range(epochs):
-            loss = self.compute_loss(data)
+        # Enhanced training loop with validation
+        for epoch in range(epochs):
+            train_loss = self.compute_loss(data['train'])
+            val_loss = self.compute_loss(data['val'])
+            print(f"Epoch {epoch}: train_loss={train_loss:.3f}, val_loss={val_loss:.3f}")
+            
+            if val_loss < self.best_loss:
+                self.save_checkpoint(f"best_model_epoch{epoch}.pkl")

--- src/utils.py ---
--- a/src/utils.py
+++ b/src/utils.py
@@ -5,0 +6,8 @@ import numpy as np
+
+def split_data(data, train_ratio=0.8):
+    """Split data into train and validation sets."""
+    split_idx = int(len(data) * train_ratio)
+    return {
+        'train': data[:split_idx],
+        'val': data[split_idx:]
+    }
```

## LLM Feedback Loop

### What LLM Sees (Iteration 2+)

```
================================================================================
PREVIOUS ITERATION EXECUTION RESULTS
================================================================================

## Command Execution Log

[1/2] Execute: python main.py
Command: python main.py
Shell: powershell
Exit Code: 1

STDERR:
Traceback (most recent call last):
  File "main.py", line 5, in <module>
    from utils import preprocess
ModuleNotFoundError: No module named 'utils'

✗ FAILED (exit code: 1)

================================================================================

## Code Changes (Git Diff Format)

[Previous iteration's code diffs...]

================================================================================
Please review the execution results above and:
1. Fix any errors or failures
2. Improve code based on actual output
3. Add new features or refinements
4. Provide new execution commands if needed
================================================================================

ALL-CODE MODE ENABLED
================================================================================

You can generate ANY code files (not restricted to simulation.py):

**Code File Format:**
```python path/to/file.py
# Your code here
```

[... instructions continue ...]
```

### What LLM Responds With

```
## REVIEW

The previous execution failed due to a missing module import. The code tried to import
from `utils` but the file was saved as `utils/helper.py`. I'll fix the import path and
also add proper error handling.

## REVISION DIFFS

File: main.py
```python
import sys
from pathlib import Path

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

# Corrected import
from code.utils.helper import preprocess

def main():
    try:
        data = preprocess("dataset.csv")
        print(f"Loaded {len(data)} samples")
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
```

File: code/utils/__init__.py
```python
# Make utils a package
from .helper import preprocess

__all__ = ['preprocess']
```

Execute: python main.py
```

## Example Use Cases

### 1. **Build a Python Library**

```bash
python main.py \
  --topic "Data Processing Library" \
  --field "Software Engineering" \
  --question "Create a high-performance data transformation library" \
  --all-code \
  --code-output-dir src \
  --max-iterations 6
```

**Result**: Complete library with:
- `src/core/transform.py` - Core transformations
- `src/io/readers.py` - Data readers
- `src/io/writers.py` - Data writers  
- `tests/test_transform.py` - Unit tests
- `setup.py` - Installation script
- `README.md` - Documentation

### 2. **Create a Web Application**

```bash
python main.py \
  --topic "Task Management Web App" \
  --field "Web Development" \
  --question "Build a React + FastAPI task tracker" \
  --all-code \
  --code-output-dir app \
  --max-iterations 8
```

**Result**: Full-stack application with:
- `app/backend/main.py` - FastAPI server
- `app/backend/models.py` - Database models
- `app/frontend/src/App.js` - React components
- `app/frontend/src/api.js` - API client
- `docker-compose.yml` - Deployment config
- Commands to run: `uvicorn main:app`, `npm start`

### 3. **Implement Research Algorithm**

```bash
python main.py \
  --topic "Novel Optimization Algorithm" \
  --field "Machine Learning" \
  --question "Implement adaptive gradient descent with momentum scheduling" \
  --all-code \
  --max-iterations 10
```

**Result**: Research implementation with:
- `code/optimizer.py` - Algorithm implementation
- `code/benchmarks.py` - Performance tests
- `code/visualize.py` - Result plotting
- `experiments/run_benchmarks.sh` - Experiment scripts
- Execution shows convergence curves and performance metrics

### 4. **Data Science Pipeline**

```bash
python main.py \
  --topic "Customer Churn Prediction" \
  --field "Data Science" \
  --question "End-to-end ML pipeline for churn prediction" \
  --all-code \
  --code-output-dir pipeline \
  --max-iterations 7
```

**Result**: ML pipeline with:
- `pipeline/data_loader.py` - Data ingestion
- `pipeline/feature_engineering.py` - Feature creation
- `pipeline/model_training.py` - Model training
- `pipeline/evaluation.py` - Metrics and validation
- `pipeline/deploy.py` - Model deployment
- Notebooks in `notebooks/exploratory_analysis.ipynb`

## Advanced Features

### Multi-Language Projects

LLM can mix languages:

```python
# Python backend
File: backend/server.py
```python
from fastapi import FastAPI
app = FastAPI()
```

# JavaScript frontend
File: frontend/app.js
```javascript
fetch('/api/data').then(r => r.json())
```

# Bash scripts
File: scripts/deploy.sh
```bash
#!/bin/bash
docker build -t myapp .
docker run -p 8000:8000 myapp
```
```

### Incremental Development

LLM adds features iteratively:

**Iteration 1**: Basic functionality
**Iteration 2**: Add error handling
**Iteration 3**: Add tests
**Iteration 4**: Add logging
**Iteration 5**: Add performance optimizations
**Iteration 6**: Add documentation

### Test-Driven Development

LLM can follow TDD:

1. **Write failing tests**
2. **Run tests** → See failures
3. **Implement features**
4. **Run tests** → See passes
5. **Refactor**
6. **Run tests** → Still passing

## Configuration

### In `config.json`

```json
{
  "all_code_mode": true,
  "code_output_dir": "src",
  "execution_log_file": "build.log",
  "execution_timeout": 600,
  "max_code_files": 100,
  "allowed_commands": ["python", "node", "npm", "cargo", "go", "pytest", "jest"]
}
```

### Security Considerations

**⚠️ WARNING**: All-code mode executes arbitrary commands. Use with caution!

**Mitigations**:
1. **Sandboxing**: Run in Docker container
2. **Command Whitelist**: Restrict allowed commands
3. **Timeout Limits**: Prevent infinite loops
4. **Resource Limits**: Set memory/CPU caps
5. **No sudo**: Never run with elevated privileges

### Command Whitelist Example

```python
ALLOWED_COMMANDS = [
    'python', 'python3',
    'node', 'npm', 'npx',
    'cargo', 'rustc',
    'go', 'go test',
    'pytest', 'jest',
    'make', 'cmake'
]

def is_command_allowed(cmd: str) -> bool:
    return any(cmd.startswith(allowed) for allowed in ALLOWED_COMMANDS)
```

## Troubleshooting

### Issue: Code files not extracted

**Solution**: Ensure LLM uses proper format:
```
```python filename.py
code here
```
```

### Issue: Commands not executed

**Solution**: Check execution log for errors. Ensure commands are formatted correctly:
```
Execute: your command here
```

### Issue: Execution timeout

**Solution**: Increase timeout:
```bash
python main.py --all-code --request-timeout 7200
```

### Issue: Permission denied

**Solution**: Check file permissions, run in user directory, not system directories.

## Best Practices

1. **Start Simple**: Begin with basic files, add complexity iteratively
2. **Clear Commands**: Use explicit commands like `Execute: python test.py`
3. **Check Logs**: Review `execution_log.txt` after each iteration
4. **Incremental**: Add 1-2 files per iteration, not 20 at once
5. **Test Early**: Have LLM write and run tests from iteration 2
6. **Use Diffs**: Review code diffs to track changes
7. **Sandbox**: Always run in isolated environment
8. **Backup**: Keep backups before running destructive commands

## Comparison: Standard vs All-Code Mode

| Feature | Standard Mode | All-Code Mode |
|---------|---------------|---------------|
| Code Files | `simulation.py` only | Unlimited files, any language |
| Execution | Manual | Automatic with feedback |
| Iteration | Review paper only | Review code + execution results |
| Use Case | Research papers | Full codebases |
| Output | LaTeX + 1 Python file | Complete project structure |
| Feedback Loop | None for code | Execution results → LLM |
| Languages | Python only | Python, JS, C++, Go, Rust, etc. |

## Future Enhancements

1. **Interactive Mode**: Real-time command input/output streaming
2. **Debugger Integration**: LLM can set breakpoints and inspect variables
3. **Git Integration**: Auto-commit after each iteration
4. **Dependency Management**: Auto-install requirements (pip, npm)
5. **Cloud Execution**: Run commands on remote servers
6. **Parallel Execution**: Run multiple commands simultaneously
7. **Performance Profiling**: Include profiling data in feedback
8. **Visual Debugging**: Screenshots of GUI apps

## Conclusion

All-code mode transforms AI-Scientist from a research paper generator into a complete software development assistant. By removing restrictions on code generation and adding iterative execution feedback, the LLM can build, test, debug, and refine real codebases across any programming language or framework.

**Start building today**:
```bash
python main.py --all-code --topic "Your Project" --field "Your Domain"
```
