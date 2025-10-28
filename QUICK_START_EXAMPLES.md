# 🚀 Quick Start: All-Code Mode Examples

## Ready to Use Commands

Copy and paste these commands to start generating experimental code immediately!

---

## 🧪 Machine Learning Examples

### Example 1: Simple Neural Network
```bash
python main.py --all-code \
  --topic "Neural Network from Scratch" \
  --question "How to implement backpropagation in NumPy?" \
  --max-iterations 3 \
  --output-dir output/simple_nn
```

**What you'll get:**
- `code/network.py` - Neural network class
- `code/layers.py` - Dense layer implementation
- `code/activations.py` - Activation functions
- `code/train.py` - Training script with gradient descent
- `code/test.py` - Test on XOR problem
- Automatic execution and debugging

---

### Example 2: PyTorch Image Classifier
```bash
python main.py --all-code \
  --topic "CIFAR-10 Image Classification" \
  --question "How to train a CNN on CIFAR-10 with data augmentation?" \
  --max-iterations 5 \
  --output-dir output/cifar10_cnn \
  --model gpt-4o
```

**What you'll get:**
- `code/model.py` - CNN architecture
- `code/dataset.py` - CIFAR-10 loader with augmentation
- `code/train.py` - Training loop with logging
- `code/evaluate.py` - Test set evaluation
- `code/visualize.py` - Plot training curves
- `code/requirements.txt` - torch, torchvision, matplotlib

---

### Example 3: Reinforcement Learning Agent
```bash
python main.py --all-code \
  --topic "Deep Q-Network for CartPole" \
  --question "How to implement DQN with experience replay and target network?" \
  --max-iterations 6 \
  --output-dir output/dqn_cartpole
```

**What you'll get:**
- `code/dqn_agent.py` - DQN agent
- `code/replay_buffer.py` - Experience replay
- `code/network.py` - Q-network
- `code/train.py` - Training loop
- `code/play.py` - Visualize trained agent
- Automatic training and evaluation

---

## 🔬 Scientific Computing Examples

### Example 4: Physics Simulation
```bash
python main.py --all-code \
  --topic "N-Body Gravitational Simulation" \
  --question "How to simulate planetary motion with Verlet integration?" \
  --max-iterations 4 \
  --output-dir output/nbody_sim
```

**What you'll get:**
- `code/nbody.py` - N-body simulation engine
- `code/integrator.py` - Verlet integrator
- `code/visualize.py` - Animation of orbits
- `code/run_simulation.py` - Main simulation script
- Energy conservation plots

---

### Example 5: Monte Carlo Simulation
```bash
python main.py --all-code \
  --topic "Monte Carlo Option Pricing" \
  --question "How to price European options using Monte Carlo?" \
  --max-iterations 3 \
  --output-dir output/monte_carlo_options
```

**What you'll get:**
- `code/monte_carlo.py` - MC simulation
- `code/option_models.py` - Black-Scholes
- `code/pricing.py` - Price calculator
- `code/visualize_paths.py` - Stock path plots
- Statistical analysis of results

---

### Example 6: Differential Equations
```bash
python main.py --all-code \
  --topic "Lorenz Attractor Simulation" \
  --question "How to solve chaotic differential equations with RK4?" \
  --max-iterations 4 \
  --output-dir output/lorenz_chaos
```

**What you'll get:**
- `code/lorenz.py` - Lorenz system
- `code/rk4_solver.py` - Runge-Kutta integrator
- `code/visualize_3d.py` - 3D butterfly plot
- `code/analyze_chaos.py` - Lyapunov exponents
- Interactive visualization

---

## 📊 Data Science Examples

### Example 7: Time Series Forecasting
```bash
python main.py --all-code \
  --topic "Stock Price Prediction with LSTM" \
  --question "How to forecast time series using LSTM networks?" \
  --max-iterations 5 \
  --output-dir output/lstm_forecast
```

**What you'll get:**
- `code/data_loader.py` - Time series preprocessing
- `code/lstm_model.py` - LSTM architecture
- `code/train.py` - Training with early stopping
- `code/predict.py` - Future forecasting
- `code/visualize.py` - Prediction plots
- Automatic training on sample data

---

### Example 8: A/B Testing Framework
```bash
python main.py --all-code \
  --topic "Statistical A/B Testing System" \
  --question "How to build automated A/B test analysis with multiple metrics?" \
  --max-iterations 4 \
  --output-dir output/ab_testing
```

**What you'll get:**
- `code/ab_test.py` - Core testing framework
- `code/statistics.py` - T-test, chi-square, power analysis
- `code/sample_size.py` - Sample size calculator
- `code/visualization.py` - Result plots
- `code/example_usage.py` - Demo with synthetic data

---

### Example 9: Clustering Analysis
```bash
python main.py --all-code \
  --topic "Customer Segmentation with K-Means" \
  --question "How to perform clustering analysis with visualization?" \
  --max-iterations 3 \
  --output-dir output/clustering
```

**What you'll get:**
- `code/kmeans.py` - K-means from scratch
- `code/data_generation.py` - Generate synthetic data
- `code/elbow_method.py` - Find optimal K
- `code/visualize.py` - 2D/3D cluster plots
- Silhouette analysis

---

## 🧮 Algorithm Examples

### Example 10: Genetic Algorithm
```bash
python main.py --all-code \
  --topic "Genetic Algorithm for TSP" \
  --question "How to solve Traveling Salesman Problem with evolutionary search?" \
  --max-iterations 5 \
  --output-dir output/genetic_tsp
```

**What you'll get:**
- `code/genetic_algorithm.py` - GA framework
- `code/tsp_problem.py` - TSP representation
- `code/crossover.py` - Crossover operators
- `code/mutation.py` - Mutation operators
- `code/visualize_evolution.py` - Convergence plots
- Animated tour visualization

---

### Example 11: Graph Algorithms
```bash
python main.py --all-code \
  --topic "Network Analysis with Graph Algorithms" \
  --question "How to implement PageRank and community detection?" \
  --max-iterations 4 \
  --output-dir output/graph_analysis
```

**What you'll get:**
- `code/graph.py` - Graph data structure
- `code/pagerank.py` - PageRank algorithm
- `code/community.py` - Louvain algorithm
- `code/visualize_network.py` - Network plots
- Example on real network data

---

### Example 12: Search Algorithms
```bash
python main.py --all-code \
  --topic "A* Pathfinding with Visualization" \
  --question "How to implement A* search with different heuristics?" \
  --max-iterations 4 \
  --output-dir output/astar_pathfinding
```

**What you'll get:**
- `code/astar.py` - A* implementation
- `code/heuristics.py` - Manhattan, Euclidean, Chebyshev
- `code/grid.py` - Grid world representation
- `code/visualize_search.py` - Animated pathfinding
- Interactive maze solver

---

## 🌐 Web Application Examples

### Example 13: FastAPI ML Service
```bash
python main.py --all-code \
  --topic "Machine Learning Model API" \
  --question "How to serve ML model predictions via REST API?" \
  --max-iterations 5 \
  --output-dir output/ml_api
```

**What you'll get:**
- `code/main.py` - FastAPI application
- `code/models.py` - Pydantic request/response models
- `code/inference.py` - Model loading and prediction
- `code/train_model.py` - Train example model
- `code/test_api.py` - API tests
- `code/requirements.txt` - fastapi, uvicorn
- `code/Dockerfile` - Container setup

---

### Example 14: Real-time Dashboard
```bash
python main.py --all-code \
  --topic "Interactive Data Dashboard" \
  --question "How to build real-time visualization dashboard with Streamlit?" \
  --max-iterations 4 \
  --output-dir output/data_dashboard
```

**What you'll get:**
- `code/app.py` - Streamlit dashboard
- `code/data_fetcher.py` - Real-time data source
- `code/visualizations.py` - Chart components
- `code/metrics.py` - Calculate KPIs
- Live updating plots

---

## 🎮 Game Development Examples

### Example 15: Game AI
```bash
python main.py --all-code \
  --topic "Tic-Tac-Toe AI with Minimax" \
  --question "How to implement unbeatable game AI?" \
  --max-iterations 4 \
  --output-dir output/tictactoe_ai
```

**What you'll get:**
- `code/game.py` - Game logic
- `code/minimax.py` - Minimax with alpha-beta pruning
- `code/player.py` - Human and AI players
- `code/gui.py` - Simple GUI
- `code/benchmark.py` - Test AI strength

---

## 💡 Research Algorithm Examples

### Example 16: Novel Optimization Method
```bash
python main.py --all-code \
  --topic "Adaptive Particle Swarm Optimization" \
  --question "How to implement PSO with adaptive parameters?" \
  --max-iterations 6 \
  --output-dir output/adaptive_pso
```

**What you'll get:**
- `code/pso.py` - PSO algorithm
- `code/test_functions.py` - Benchmark functions
- `code/adaptive_params.py` - Parameter adaptation
- `code/compare_baselines.py` - Compare with standard PSO
- `code/visualize_convergence.py` - Convergence plots
- Performance analysis

---

### Example 17: Custom Attention Mechanism
```bash
python main.py --all-code \
  --topic "Novel Attention Mechanism for NLP" \
  --question "How to implement multi-head attention with positional encoding?" \
  --max-iterations 6 \
  --output-dir output/custom_attention
```

**What you'll get:**
- `code/attention.py` - Attention layers
- `code/positional_encoding.py` - Position embeddings
- `code/transformer_block.py` - Full transformer
- `code/train_language_model.py` - Training on text
- `code/visualize_attention.py` - Attention heatmaps
- Comparison with standard attention

---

## 🔧 Advanced Multi-File Projects

### Example 18: Complete ML Pipeline
```bash
python main.py --all-code \
  --topic "End-to-End Machine Learning Pipeline" \
  --question "How to build production ML system with data processing, training, evaluation, and serving?" \
  --max-iterations 8 \
  --output-dir output/ml_pipeline \
  --model gpt-4o
```

**What you'll get:**
- `code/data/` - Data loading and preprocessing
- `code/models/` - Model architectures
- `code/training/` - Training scripts
- `code/evaluation/` - Metrics and evaluation
- `code/serving/` - API and inference
- `code/utils/` - Helper functions
- `code/tests/` - Unit and integration tests
- `code/configs/` - Configuration files
- Complete production-ready pipeline

---

### Example 19: Research Codebase
```bash
python main.py --all-code \
  --topic "Novel Deep Learning Architecture Research" \
  --question "How to implement, train, and evaluate a new architecture for image classification?" \
  --max-iterations 10 \
  --output-dir output/research_architecture
```

**What you'll get:**
- Complete research codebase
- Baseline comparisons
- Ablation studies
- Visualization tools
- Experiment tracking
- Reproducible results
- Research paper (paper.tex)

---

## 📝 Tips for Best Results

### 1. Be Specific in Questions
```bash
# ✅ Good
--question "How to implement LSTM with attention for sequence-to-sequence translation?"

# ❌ Too vague
--question "How to do machine learning?"
```

### 2. Specify Constraints
```bash
--question "How to implement neural network using only NumPy (no frameworks)?"
--question "How to optimize code to run in under 1 second on CPU?"
--question "How to implement with 90%+ test coverage?"
```

### 3. Request Testing
```bash
--question "How to implement X with comprehensive unit tests and examples?"
```

### 4. Ask for Documentation
```bash
--question "How to build Y with detailed docstrings and README?"
```

### 5. Use More Iterations for Complex Projects
- Simple scripts: `--max-iterations 3`
- Medium complexity: `--max-iterations 5`
- Complex projects: `--max-iterations 8-10`

---

## 🎯 Quick Command Template

```bash
python main.py --all-code \
  --topic "YOUR_TOPIC_HERE" \
  --question "YOUR_SPECIFIC_QUESTION_HERE" \
  --max-iterations 5 \
  --output-dir output/YOUR_PROJECT_NAME
```

**Replace:**
- `YOUR_TOPIC_HERE` with your experimental topic
- `YOUR_SPECIFIC_QUESTION_HERE` with what you want to implement
- `YOUR_PROJECT_NAME` with desired output directory name

---

## 📚 Next Steps

1. **Choose an example** from above that matches your needs
2. **Copy the command** and modify for your use case
3. **Run the command** and watch the LLM generate code
4. **Review the output** in the output directory
5. **Iterate** by modifying and re-running if needed

For more information, see:
- `EXPERIMENTAL_CODE_GENERATION.md` - Detailed guide
- `ALL_CODE_MODE_DOCUMENTATION.md` - Complete documentation
- `ALL_CODE_MODE_READY.md` - Feature verification

**Start experimenting!** 🚀
