#!/usr/bin/env python3
"""
Demonstration: All-Code Mode with LLM Creating and Executing Complicated Code

This script shows how to use all-code mode where the LLM:
1. Creates complicated code in multiple languages
2. Generates execution commands
3. System automatically runs the commands
4. Results are fed back to LLM for iteration
"""

import sys
import subprocess
from pathlib import Path

def demo_simple_example():
    """Demo 1: Simple Python code generation and execution"""
    print("=" * 80)
    print("DEMO 1: Simple Neural Network Implementation")
    print("=" * 80)
    print()
    print("Command:")
    cmd = [
        sys.executable, "main.py",
        "--all-code",
        "--topic", "Simple Neural Network from Scratch",
        "--question", "How to implement backpropagation in pure Python?",
        "--max-iterations", "3",
        "--output-dir", "output/demo_neural_net",
        "--model", "gpt-4o"
    ]
    print(" ".join(cmd))
    print()
    print("What will happen:")
    print("  1. LLM creates: network.py, train.py, test.py")
    print("  2. LLM outputs: Execute: python train.py")
    print("  3. System runs the command automatically")
    print("  4. If errors occur, LLM sees them and fixes in next iteration")
    print()
    
    response = input("Run this demo? (y/n): ").strip().lower()
    if response == 'y':
        print("\nRunning...")
        subprocess.run(cmd)
    else:
        print("Skipped.")
    print()


def demo_multifile_project():
    """Demo 2: Complex multi-file project"""
    print("=" * 80)
    print("DEMO 2: Multi-File Physics Simulation")
    print("=" * 80)
    print()
    print("Command:")
    cmd = [
        sys.executable, "main.py",
        "--all-code",
        "--topic", "N-Body Gravitational Simulation",
        "--question", "How to simulate planetary motion with Verlet integration?",
        "--max-iterations", "5",
        "--output-dir", "output/demo_nbody",
        "--model", "gpt-5-pro"
    ]
    print(" ".join(cmd))
    print()
    print("What LLM will create:")
    print("  code/")
    print("    ├── src/")
    print("    │   ├── nbody.py         (simulation engine)")
    print("    │   ├── integrator.py    (Verlet integrator)")
    print("    │   └── forces.py        (gravity calculations)")
    print("    ├── visualize.py         (plotting)")
    print("    ├── run_simulation.py    (main script)")
    print("    └── requirements.txt     (dependencies)")
    print()
    print("What commands LLM will generate:")
    print("  Execute: pip install -r requirements.txt")
    print("  Execute: python run_simulation.py")
    print("  Execute: python visualize.py")
    print()
    print("System will automatically:")
    print("  ✓ Install dependencies")
    print("  ✓ Run simulation")
    print("  ✓ Generate plots")
    print("  ✓ Report results to LLM")
    print()
    
    response = input("Run this demo? (y/n): ").strip().lower()
    if response == 'y':
        print("\nRunning...")
        subprocess.run(cmd)
    else:
        print("Skipped.")
    print()


def demo_multilanguage():
    """Demo 3: Multi-language project (Python + C++)"""
    print("=" * 80)
    print("DEMO 3: Python-C++ Hybrid Performance Library")
    print("=" * 80)
    print()
    print("Command:")
    cmd = [
        sys.executable, "main.py",
        "--all-code",
        "--topic", "High-Performance Matrix Operations",
        "--question", "How to accelerate Python with C++ extensions using pybind11?",
        "--max-iterations", "6",
        "--output-dir", "output/demo_hybrid",
        "--model", "gpt-5-pro"
    ]
    print(" ".join(cmd))
    print()
    print("What LLM will create:")
    print("  code/")
    print("    ├── cpp/")
    print("    │   ├── matrix_ops.cpp   (C++ implementation)")
    print("    │   └── matrix_ops.h     (header file)")
    print("    ├── python/")
    print("    │   ├── __init__.py")
    print("    │   └── bindings.cpp     (pybind11 bindings)")
    print("    ├── tests/")
    print("    │   └── test_ops.py")
    print("    ├── CMakeLists.txt       (build configuration)")
    print("    ├── setup.py             (Python packaging)")
    print("    └── benchmark.py         (performance comparison)")
    print()
    print("What commands LLM will generate:")
    print("  Execute: mkdir build && cd build && cmake ..")
    print("  Execute: make")
    print("  Execute: python setup.py install")
    print("  Execute: python benchmark.py")
    print()
    
    response = input("Run this demo? (y/n): ").strip().lower()
    if response == 'y':
        print("\nRunning...")
        subprocess.run(cmd)
    else:
        print("Skipped.")
    print()


def demo_black_hole_paper():
    """Demo 4: Modify existing paper with complex code generation"""
    print("=" * 80)
    print("DEMO 4: Black Hole Paper with Quantum Simulation Code")
    print("=" * 80)
    print()
    print("Command:")
    cmd = [
        sys.executable, "main.py",
        "--all-code",
        "--modify-existing",
        "--output-dir", "output/black_hole",
        "--model", "gpt-5-pro",
        "--max-iterations", "5",
        "--user-prompt", "Add comprehensive quantum simulation code for the black hole thermodynamics model"
    ]
    print(" ".join(cmd))
    print()
    print("What LLM will do:")
    print("  1. Read existing black hole paper")
    print("  2. Create quantum simulation framework:")
    print("     - Hawking radiation calculator")
    print("     - Entropy evolution tracker")
    print("     - Page curve generator")
    print("     - Information scrambling simulator")
    print("  3. Generate execution commands:")
    print("     Execute: python src/hawking_radiation.py")
    print("     Execute: python src/page_curve.py --save-plot")
    print("  4. Run simulations automatically")
    print("  5. See results and iterate to fix any issues")
    print("  6. Update paper with new simulation results")
    print()
    
    response = input("Run this demo? (y/n): ").strip().lower()
    if response == 'y':
        print("\nRunning...")
        subprocess.run(cmd)
    else:
        print("Skipped.")
    print()


def demo_custom_complicated():
    """Demo 5: User's custom complicated code request"""
    print("=" * 80)
    print("DEMO 5: Custom Complicated Code Generation")
    print("=" * 80)
    print()
    print("This demo lets you specify YOUR OWN complicated code project.")
    print()
    
    topic = input("Enter your topic (e.g., 'Quantum Machine Learning Framework'): ").strip()
    if not topic:
        print("Skipped.")
        return
    
    question = input("Enter your question (e.g., 'How to implement variational quantum circuits?'): ").strip()
    if not question:
        question = f"How to implement {topic}?"
    
    iterations = input("Max iterations (default 5): ").strip()
    iterations = iterations if iterations else "5"
    
    model = input("Model (gpt-5-pro/gemini-2.5-pro, default gpt-5-pro): ").strip()
    model = model if model else "gpt-5-pro"
    
    print()
    cmd = [
        sys.executable, "main.py",
        "--all-code",
        "--topic", topic,
        "--question", question,
        "--max-iterations", iterations,
        "--output-dir", f"output/custom_{topic.replace(' ', '_').lower()[:20]}",
        "--model", model
    ]
    print("Command:")
    print(" ".join(cmd))
    print()
    print("The LLM will:")
    print("  ✓ Create complicated code for your topic")
    print("  ✓ Generate execution commands")
    print("  ✓ System will run them automatically")
    print("  ✓ Iterate to fix any errors")
    print()
    
    response = input("Run this custom demo? (y/n): ").strip().lower()
    if response == 'y':
        print("\nRunning...")
        subprocess.run(cmd)
    else:
        print("Skipped.")
    print()


def show_feature_summary():
    """Show summary of all-code mode features"""
    print("=" * 80)
    print("ALL-CODE MODE FEATURE SUMMARY")
    print("=" * 80)
    print()
    print("✅ What LLM Can Do:")
    print("   • Create code in ANY language (Python, C++, JavaScript, R, Julia, Go, Rust, etc.)")
    print("   • Generate COMPLEX multi-file projects")
    print("   • Create nested directory structures")
    print("   • Write tests, documentation, build scripts")
    print("   • Output execution commands")
    print()
    print("✅ What System Does Automatically:")
    print("   • Extract ALL code blocks from LLM response")
    print("   • Save files to proper locations")
    print("   • Parse execution commands from LLM")
    print("   • Run commands with timeout protection")
    print("   • Capture stdout, stderr, exit codes")
    print("   • Log everything to execution_log.txt")
    print("   • Send results back to LLM")
    print("   • Generate code diffs across iterations")
    print()
    print("✅ Supported Command Formats:")
    print("   Execute: python script.py")
    print("   Run: pytest tests/")
    print("   $ pip install numpy")
    print("   > powershell -Command 'Get-Date'")
    print("   ```bash")
    print("   python train.py")
    print("   ```")
    print()
    print("✅ Safety Features:")
    print("   • Command timeout (default: 5 minutes)")
    print("   • Error isolation (failures don't crash workflow)")
    print("   • Execution logging")
    print("   • Exit code tracking")
    print()
    print("✅ Documentation:")
    print("   • ALL_CODE_INDEX.md - Navigation hub")
    print("   • EXPERIMENTAL_CODE_GENERATION.md - Complete guide")
    print("   • QUICK_START_EXAMPLES.md - 19 ready examples")
    print("   • ALL_CODE_VERIFICATION.md - This verification")
    print()


def main():
    """Main demo menu"""
    print("\n" + "=" * 80)
    print("🧪 ALL-CODE MODE: LLM CREATES & EXECUTES COMPLICATED CODE")
    print("=" * 80)
    print()
    print("This demonstrates that the software ALREADY:")
    print("  ✅ Allows LLM to create complicated code")
    print("  ✅ Lets LLM generate execution commands")
    print("  ✅ Automatically executes those commands")
    print("  ✅ Captures results and feeds back to LLM")
    print()
    
    while True:
        print("\n" + "-" * 80)
        print("SELECT A DEMO:")
        print("-" * 80)
        print("  1. Simple Neural Network (3 iterations, quick)")
        print("  2. Multi-File Physics Simulation (5 iterations)")
        print("  3. Python-C++ Hybrid Library (6 iterations, advanced)")
        print("  4. Black Hole Paper + Quantum Code (modify existing)")
        print("  5. Custom Complicated Code (your own topic)")
        print()
        print("  0. Show Feature Summary")
        print("  q. Quit")
        print()
        
        choice = input("Enter choice: ").strip().lower()
        print()
        
        if choice == '1':
            demo_simple_example()
        elif choice == '2':
            demo_multifile_project()
        elif choice == '3':
            demo_multilanguage()
        elif choice == '4':
            demo_black_hole_paper()
        elif choice == '5':
            demo_custom_complicated()
        elif choice == '0':
            show_feature_summary()
        elif choice == 'q':
            print("Exiting demo.")
            break
        else:
            print("Invalid choice. Try again.")
    
    print("\n" + "=" * 80)
    print("Thank you for exploring All-Code Mode!")
    print("=" * 80)
    print()
    print("📚 Documentation:")
    print("   • ALL_CODE_VERIFICATION.md - Feature verification")
    print("   • EXPERIMENTAL_CODE_GENERATION.md - Complete guide")
    print("   • QUICK_START_EXAMPLES.md - 19 examples")
    print()
    print("🚀 Quick Start:")
    print("   python main.py --all-code --topic 'Your Topic' --max-iterations 5")
    print()


if __name__ == "__main__":
    main()
