#!/usr/bin/env python3
"""
Demo script showing custom prompt examples
"""

def show_custom_prompt_examples():
    """Display various custom prompt examples for different use cases."""
    
    print("🎯 CUSTOM PROMPT EXAMPLES")
    print("="*60)
    print()
    
    examples = [
        {
            "category": "📐 Mathematical Focus",
            "prompt": "Emphasize mathematical rigor with formal proofs, lemmas, and theorems. Include detailed complexity analysis and mathematical notation.",
            "best_for": "Theoretical papers, algorithm papers, formal methods"
        },
        {
            "category": "🔬 Experimental Focus", 
            "prompt": "Prioritize empirical validation with extensive experiments, statistical analysis, and reproducible results. Include detailed methodology and error analysis.",
            "best_for": "Experimental papers, machine learning, systems evaluation"
        },
        {
            "category": "🌍 Practical Applications",
            "prompt": "Focus on real-world applications, case studies, and practical implications. Include industry relevance and deployment considerations.",
            "best_for": "Systems papers, applied research, industry collaboration"
        },
        {
            "category": "📚 Survey Style",
            "prompt": "Structure as a comprehensive survey with systematic classification, comparative analysis, and future research directions.",
            "best_for": "Review papers, state-of-the-art surveys, literature reviews"
        },
        {
            "category": "🔒 Security Focus",
            "prompt": "Include detailed threat models, security analysis, attack scenarios, and defense mechanisms. Emphasize formal security proofs where applicable.",
            "best_for": "Cybersecurity papers, cryptography, privacy research"
        },
        {
            "category": "🏥 Medical/Clinical",
            "prompt": "Follow clinical research standards with appropriate validation protocols, ethical considerations, and statistical significance testing.",
            "best_for": "Medical informatics, healthcare AI, clinical studies"
        },
        {
            "category": "💬 Accessible Writing",
            "prompt": "Use clear, accessible language suitable for interdisciplinary audiences while maintaining technical accuracy. Include intuitive explanations.",
            "best_for": "Interdisciplinary research, broad impact papers"
        },
        {
            "category": "⚡ Performance Focus",
            "prompt": "Emphasize scalability, performance optimization, and efficiency analysis. Include detailed benchmarks and performance comparisons.",
            "best_for": "Systems papers, high-performance computing, optimization"
        }
    ]
    
    for i, example in enumerate(examples, 1):
        print(f"{i}. {example['category']}")
        print(f"   Prompt: \"{example['prompt']}\"")
        print(f"   Best for: {example['best_for']}")
        print()
    
    print("💡 USAGE TIPS")
    print("-" * 30)
    print("• Be specific about what you want to emphasize")
    print("• Mention writing style preferences")
    print("• Specify evaluation methodologies if relevant")
    print("• Include field-specific requirements")
    print("• Balance user preferences with paper quality")
    print()
    
    print("🚀 COMMAND LINE EXAMPLES")
    print("-" * 30)
    print("# Mathematical focus:")
    print("python sciresearch_workflow.py --topic 'Algorithm Design' \\")
    print("  --user-prompt 'Emphasize mathematical rigor with formal complexity analysis'")
    print()
    print("# Practical applications:")
    print("python sciresearch_workflow.py --topic 'ML Security' \\")
    print("  --user-prompt 'Focus on real-world applications with industry case studies'")
    print()
    print("# Survey style:")
    print("python sciresearch_workflow.py --topic 'Blockchain Scalability' \\")
    print("  --user-prompt 'Structure as comprehensive survey with comparative analysis'")

if __name__ == "__main__":
    show_custom_prompt_examples()
