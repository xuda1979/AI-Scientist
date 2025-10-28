# 🧪 Experimental Code Generation with All-Code Mode

## Overview

The AI-Scientist software can now **help write ALL kinds of experimental code** using the `--all-code` flag. This mode removes all restrictions on file types and enables the LLM to generate complete, executable codebases for any research or experimental purpose.

## ✅ What's Enabled

### Full Code Generation Freedom
- ✅ **Any Programming Language**: Python, C++, JavaScript, R, Julia, Go, Rust, etc.
- ✅ **Any File Type**: `.py`, `.cpp`, `.js`, `.r`, `.jl`, `.sh`, `.sql`, etc.
- ✅ **Any Project Structure**: Nested directories, modules, packages
- ✅ **No File Name Restrictions**: Not limited to `simulation.py` anymore

### Automatic Execution & Feedback
- ✅ **Command Extraction**: LLM specifies what commands to run
- ✅ **Auto-Execution**: System runs the commands automatically
- ✅ **Result Capture**: Stdout, stderr, and exit codes logged
- ✅ **Iterative Debugging**: LLM sees results and fixes errors

### Complete Project Development
- ✅ **Multiple Files**: Generate entire project structures
- ✅ **Dependencies**: Create `requirements.txt`, `package.json`, etc.
- ✅ **Tests**: Write unit tests, integration tests
- ✅ **Documentation**: Generate README files, docstrings
- ✅ **Build Scripts**: Makefiles, shell scripts, CI/CD configs

## 🚀 Quick Start

### Basic Usage

```bash
python main.py \
  --all-code \
  --topic "Your Experiment Topic" \
  --field "Research Field" \
  --question "What research question to explore?" \
  --max-iterations 5 \
  --output-dir output/my_experiment
```

### Example: Machine Learning Experiment

```bash
python main.py \
  --all-code \
  --topic "Deep Reinforcement Learning for Game AI" \
  --field "Machine Learning" \
  --question "How to implement DQN for Atari games?" \
  --max-iterations 6 \
  --output-dir output/dqn_atari
```

**This will generate:**
- `code/agent.py` - DQN agent implementation
- `code/network.py` - Neural network architecture
- `code/replay_buffer.py` - Experience replay buffer
- `code/train.py` - Training loop
- `code/evaluate.py` - Evaluation script
- `code/utils.py` - Helper functions
- `code/requirements.txt` - Dependencies
- `execution_log.txt` - Command execution results

## 📋 Types of Experimental Code Supported

### 1. Machine Learning Experiments

```bash
# PyTorch Model Training
python main.py --all-code \
  --topic "Image Classification with Vision Transformers" \
  --question "How to fine-tune ViT on custom dataset?"

# TensorFlow Implementation
python main.py --all-code \
  --topic "Seq2Seq Model with Attention" \
  --question "How to implement machine translation?"

# Custom Training Loops
python main.py --all-code \
  --topic "Meta-Learning with MAML" \
  --question "How to implement Model-Agnostic Meta-Learning?"
```

**Generates**: Model architectures, data loaders, training loops, evaluation scripts, visualization code

### 2. Scientific Simulations

```bash
# Physics Simulation
python main.py --all-code \
  --topic "N-Body Gravity Simulation" \
  --question "How to simulate planetary motion?"

# Chemistry Simulation
python main.py --all-code \
  --topic "Molecular Dynamics with Lennard-Jones Potential" \
  --question "How to simulate particle interactions?"

# Biological Simulation
python main.py --all-code \
  --topic "Epidemic Spread Model (SIR)" \
  --question "How to model disease propagation?"
```

**Generates**: Simulation engines, numerical solvers, visualization, parameter studies

### 3. Data Analysis Pipelines

```bash
# Statistical Analysis
python main.py --all-code \
  --topic "A/B Testing Framework" \
  --question "How to analyze experimental data with statistical tests?"

# Time Series Analysis
python main.py --all-code \
  --topic "Stock Price Forecasting with ARIMA" \
  --question "How to build time series prediction pipeline?"

# Big Data Processing
python main.py --all-code \
  --topic "Distributed Data Processing with Spark" \
  --question "How to analyze large-scale datasets?"
```

**Generates**: Data cleaning, feature engineering, statistical tests, visualization, reports

### 4. Algorithm Implementations

```bash
# Optimization Algorithms
python main.py --all-code \
  --topic "Genetic Algorithm for TSP" \
  --question "How to solve Traveling Salesman Problem?"

# Graph Algorithms
python main.py --all-code \
  --topic "Community Detection in Networks" \
  --question "How to implement Louvain algorithm?"

# Search Algorithms
python main.py --all-code \
  --topic "A* Pathfinding with Heuristics" \
  --question "How to implement efficient pathfinding?"
```

**Generates**: Algorithm implementations, benchmarking, test cases, performance analysis

### 5. Web Applications & APIs

```bash
# REST API
python main.py --all-code \
  --topic "FastAPI Server for ML Model" \
  --question "How to serve trained model as REST API?"

# Web Dashboard
python main.py --all-code \
  --topic "Real-time Data Visualization Dashboard" \
  --question "How to build interactive web dashboard?"

# Database Application
python main.py --all-code \
  --topic "PostgreSQL Data Management System" \
  --question "How to build CRUD application with SQL?"
```

**Generates**: API endpoints, database schemas, frontend code, testing, deployment configs

### 6. Research Prototypes

```bash
# Novel Architecture
python main.py --all-code \
  --topic "Hybrid CNN-Transformer Architecture" \
  --question "How to combine convolutional and attention layers?"

# Custom Loss Functions
python main.py --all-code \
  --topic "Perceptual Loss for Image Generation" \
  --question "How to implement custom differentiable losses?"

# Experimental Optimizers
python main.py --all-code \
  --topic "Adaptive Learning Rate Scheduler" \
  --question "How to implement novel optimization strategies?"
```

**Generates**: Research code, ablation studies, comparison baselines, experiment tracking

## 🔧 Advanced Features

### Multi-Language Projects

The LLM can generate projects mixing multiple languages:

```bash
python main.py --all-code \
  --topic "High-Performance Computing Pipeline" \
  --question "How to accelerate Python with C++ extensions?"
```

**Generates**:
- `code/core.cpp` - C++ implementation for speed-critical code
- `code/bindings.py` - Python bindings using pybind11
- `code/main.py` - Python interface
- `code/CMakeLists.txt` - Build configuration
- `code/setup.py` - Python package setup

### Test-Driven Development

```bash
python main.py --all-code \
  --topic "Robust Data Processing Library" \
  --question "How to build well-tested data pipeline?"
```

**The LLM will**:
1. Generate initial implementation
2. Create unit tests
3. Run tests and see failures
4. Fix bugs based on test output
5. Iterate until all tests pass

### Continuous Integration

```bash
python main.py --all-code \
  --topic "Production-Ready ML Pipeline" \
  --question "How to build deployable ML system?"
```

**Generates**:
- `code/.github/workflows/ci.yml` - GitHub Actions
- `code/Dockerfile` - Container configuration
- `code/docker-compose.yml` - Multi-service setup
- `code/Makefile` - Build automation
- `code/tests/` - Complete test suite

## 📊 Workflow Example

### Iteration-by-Iteration Breakdown

**Iteration 1: Initial Generation**
```
LLM generates:
  - model.py (neural network)
  - train.py (training script)
  - requirements.txt

System executes:
  $ pip install -r requirements.txt
  $ python train.py

Results:
  ModuleNotFoundError: No module named 'torch'
```

**Iteration 2: Fix Dependencies**
```
LLM sees error and updates:
  - requirements.txt (adds torch, numpy, matplotlib)

System executes:
  $ pip install -r requirements.txt
  $ python train.py

Results:
  FileNotFoundError: data/train.csv not found
```

**Iteration 3: Add Data Handling**
```
LLM generates:
  - data_loader.py (downloads and processes data)
  - train.py (updated to use data loader)

System executes:
  $ python data_loader.py
  $ python train.py

Results:
  Training epoch 1/10... Loss: 0.523
  Training complete! Best accuracy: 0.87
```

**Iteration 4: Add Evaluation**
```
LLM generates:
  - evaluate.py (test set evaluation)
  - visualize.py (plot training curves)

System executes:
  $ python evaluate.py
  $ python visualize.py

Results:
  Test Accuracy: 0.85
  Plots saved to results/training_curve.png
```

## 🎯 Use Cases

### Research Paper Experiments

When writing a research paper, you can use all-code mode to generate all experimental code:

```bash
python main.py --all-code \
  --modify-existing \
  --output-dir papers/my_paper \
  --topic "Novel Attention Mechanism for NLP" \
  --max-iterations 8
```

**Benefits**:
- Complete codebase for paper experiments
- Reproducible results with execution logs
- Automatic debugging and refinement
- Ready-to-share code repository

### Algorithm Development

Develop and test new algorithms:

```bash
python main.py --all-code \
  --topic "Adaptive Sampling for Bayesian Optimization" \
  --question "How to improve acquisition function efficiency?"
```

**Benefits**:
- Rapid prototyping
- Automatic benchmarking
- Comparison with baselines
- Performance profiling

### Educational Implementations

Learn by implementing concepts:

```bash
python main.py --all-code \
  --topic "Transformer Architecture from Scratch" \
  --question "How does self-attention work mathematically?"
```

**Benefits**:
- Step-by-step implementation
- Working examples
- Detailed comments
- Visualization of concepts

## ⚙️ Configuration Options

### Command-Line Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--all-code` | Enable unrestricted code generation | `False` |
| `--code-output-dir` | Subdirectory for code files | `"code"` |
| `--execution-log` | Log file for command execution | `"execution_log.txt"` |
| `--max-iterations` | Maximum refinement iterations | `3` |
| `--output-dir` | Project output directory | `output/<topic>` |
| `--model` | LLM model to use | `"gpt-4o"` |

### Example with Custom Settings

```bash
python main.py \
  --all-code \
  --code-output-dir src \
  --execution-log build.log \
  --max-iterations 10 \
  --output-dir experiments/exp_001 \
  --model gpt-4o
```

## 🔒 Security Considerations

### ⚠️ Warning

All-code mode executes arbitrary code generated by the LLM. Be cautious:

1. **Review generated code** before execution
2. **Use virtual environments** or containers
3. **Set resource limits** (CPU, memory, time)
4. **Avoid sensitive data** in project directories
5. **Monitor execution** for unexpected behavior

### Recommended Safety Measures

```bash
# Use Docker container for isolation
docker run -it --rm \
  -v $(pwd):/workspace \
  --memory="4g" \
  --cpus="2" \
  python:3.11 bash

# Inside container
cd /workspace
python main.py --all-code --topic "Your Topic"
```

### Timeout Protection

All commands have default 5-minute timeout. Modify in code if needed:

```python
# In utils/all_code_handler.py
execute_command(cmd, cwd, timeout=300)  # 5 minutes
```

## 📚 Examples Gallery

### Example 1: Reinforcement Learning

**Command:**
```bash
python main.py --all-code \
  --topic "PPO Algorithm for Continuous Control" \
  --max-iterations 5
```

**Generated Structure:**
```
output/ppo_algorithm_for_continuous_control/
├── code/
│   ├── ppo_agent.py          # Actor-critic agent
│   ├── policy_network.py     # Policy network
│   ├── value_network.py      # Value network
│   ├── environment.py        # Gym environment wrapper
│   ├── train.py              # Training loop
│   ├── evaluate.py           # Evaluation script
│   ├── utils.py              # Helper functions
│   └── requirements.txt      # torch, gym, numpy
├── execution_log.txt         # Command execution history
├── paper.tex                 # LaTeX paper
└── diffs/                    # Code change diffs
```

### Example 2: Computational Biology

**Command:**
```bash
python main.py --all-code \
  --topic "Protein Folding Energy Minimization" \
  --field "Computational Biology"
```

**Generated Files:**
- `code/protein_structure.py` - PDB file parser
- `code/energy_functions.py` - Force field calculations
- `code/minimizer.py` - Gradient descent optimizer
- `code/visualize_structure.py` - 3D structure plots
- `code/benchmark.py` - Compare with known structures

### Example 3: Computer Vision

**Command:**
```bash
python main.py --all-code \
  --topic "Object Detection with YOLO" \
  --max-iterations 6
```

**Generated Pipeline:**
- Data augmentation
- Model architecture
- Loss functions
- Training with logging
- Inference script
- Evaluation metrics (mAP)
- Visualization overlays

## 🆚 Comparison: Standard vs All-Code Mode

| Feature | Standard Mode | All-Code Mode |
|---------|--------------|---------------|
| **File Generation** | Only `simulation.py` | Any files, any languages |
| **Code Execution** | Manual | Automatic with logging |
| **Feedback Loop** | None | Iterative debugging |
| **Project Structure** | Flat | Nested directories |
| **Testing** | Manual | LLM-generated tests |
| **Documentation** | Paper-focused | Code + Paper |
| **Iteration Type** | Paper revisions | Code debugging + Paper |

## 🎓 Best Practices

### 1. Start with Clear Goals

```bash
# ✅ Good: Specific, measurable goal
--topic "Train 95% accurate MNIST classifier in <10 lines"

# ❌ Bad: Vague, unmeasurable
--topic "Do machine learning"
```

### 2. Use Adequate Iterations

- Simple scripts: `--max-iterations 3`
- Complex projects: `--max-iterations 6-10`
- Novel research: `--max-iterations 10+`

### 3. Specify Constraints

```bash
--question "How to implement X using only NumPy (no deep learning libraries)?"
--question "How to optimize Y to run in <1 second on CPU?"
```

### 4. Request Tests

```bash
--question "How to implement Z with comprehensive unit tests and >90% coverage?"
```

### 5. Ask for Documentation

```bash
--question "How to build W with detailed docstrings and usage examples?"
```

## 🐛 Troubleshooting

### Issue: Commands Not Executing

**Solution**: Check that LLM is providing commands in supported formats:
- `Execute: python script.py`
- `$ python script.py`
- Bash code blocks with `#!/bin/bash`

### Issue: Files Not Generated

**Solution**: Verify code block format in LLM response:
- ` ```python filename.py`
- `File: path/to/file.ext\n```python`

### Issue: Execution Timeout

**Solution**: Increase timeout in `utils/all_code_handler.py`:
```python
execute_command(cmd, cwd, timeout=600)  # 10 minutes
```

### Issue: Import Errors

**Solution**: LLM should generate `requirements.txt` and run:
```
Execute: pip install -r requirements.txt
```

## 📖 Further Reading

- **Complete Documentation**: `ALL_CODE_MODE_DOCUMENTATION.md`
- **Implementation Summary**: `ALL_CODE_MODE_SUMMARY.md`
- **Test Examples**: `test_all_code_example.py`
- **Code Reference**: `utils/all_code_handler.py`

## 🤝 Contributing

If you develop interesting experimental code with all-code mode, consider:

1. Sharing your workflow as an example
2. Reporting bugs or edge cases
3. Suggesting new features
4. Contributing code improvements

## ✨ Summary

**The AI-Scientist can now write ALL kinds of experimental code!**

✅ Any language, any file type, any project structure  
✅ Automatic execution with iterative debugging  
✅ Complete projects from idea to working code  
✅ Research experiments, algorithms, applications  
✅ Test-driven development with LLM feedback  

**Get started now:**

```bash
python main.py --all-code --topic "Your Experiment" --max-iterations 5
```

Happy experimenting! 🚀
