"""
Example integration of quality enhancement system with AI-Scientist main workflow.

This demonstrates how to integrate the research quality validators and enhanced prompts
into the existing AI-Scientist paper generation process.
"""

import os
import sys
from pathlib import Path

# Add utils to path
utils_path = Path(__file__).parent / 'utils'
sys.path.insert(0, str(utils_path))

from utils.quality_enhanced_workflow import QualityEnhancedWorkflow
from utils.research_robustness_validators import ComprehensiveResearchValidator
from utils.enhanced_research_prompts import get_enhanced_paper_prompt

def enhanced_ai_scientist_main(
    idea: str = None,
    paper_type: str = "theoretical",
    output_dir: str = "output",
    model: str = "gpt-4",
    enable_quality_assurance: bool = True,
    quality_threshold: float = 0.6,
    max_quality_iterations: int = 3
):
    """
    Enhanced AI-Scientist main function with integrated quality assurance.
    
    Args:
        idea: Research idea to develop
        paper_type: Type of paper ("theoretical", "computational", "experimental") 
        output_dir: Output directory for generated paper
        model: LLM model to use
        enable_quality_assurance: Whether to use quality validation system
        quality_threshold: Minimum quality score required (0-1)
        max_quality_iterations: Maximum revision iterations for quality improvement
    """
    
    print("=" * 60)
    print("AI-SCIENTIST WITH RESEARCH QUALITY ENHANCEMENT")
    print("=" * 60)
    
    if not idea:
        idea = input("Enter your research idea: ")
    
    print(f"\nGenerating {paper_type} paper for idea: {idea}")
    print(f"Output directory: {output_dir}")
    print(f"Quality assurance: {'Enabled' if enable_quality_assurance else 'Disabled'}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    if enable_quality_assurance:
        # Use quality-enhanced workflow
        print("\n" + "="*50)
        print("USING QUALITY-ENHANCED WORKFLOW")
        print("="*50)
        
        workflow = QualityEnhancedWorkflow(
            quality_threshold=quality_threshold,
            max_iterations=max_quality_iterations
        )
        
        # Generate paper with quality assurance
        success, validation_results, paper_path = workflow.generate_paper_with_quality_assurance(
            idea=idea,
            paper_type=paper_type,
            output_dir=output_dir,
            model=model
        )
        
        if success:
            print(f"\n✅ HIGH-QUALITY PAPER GENERATED SUCCESSFULLY!")
            print(f"📄 Paper saved to: {paper_path}")
            print(f"📊 Quality score: {sum(r.score for r in validation_results.values()) / len(validation_results):.2f}/1.00")
        else:
            print(f"\n⚠️  Paper generated but quality threshold not met")
            print(f"📄 Paper saved to: {paper_path}")
            print("📋 Check quality_validation_report.txt for improvement suggestions")
            
        # Print quality summary
        print(f"\n📈 QUALITY BREAKDOWN:")
        for name, result in validation_results.items():
            status = "✅ PASS" if result.passed else "❌ FAIL"
            print(f"  {name.replace('_', ' ').title()}: {result.score:.2f}/1.00 {status}")
    
    else:
        # Use standard workflow (placeholder - replace with actual implementation)
        print("\n" + "="*50)
        print("USING STANDARD WORKFLOW")  
        print("="*50)
        print("⚠️  Standard workflow not implemented in this example")
        print("💡 Set enable_quality_assurance=True to use enhanced system")
        
        # Generate enhanced prompt even for standard workflow
        enhanced_prompt = get_enhanced_paper_prompt(paper_type)
        print(f"\n📝 Enhanced prompt generated ({len(enhanced_prompt)} characters)")
        
        # Save enhanced prompt for manual use
        prompt_path = os.path.join(output_dir, "enhanced_prompt.txt")
        with open(prompt_path, 'w') as f:
            f.write(f"Research Idea: {idea}\n\n{enhanced_prompt}")
        print(f"📄 Enhanced prompt saved to: {prompt_path}")

def validate_existing_paper_example(paper_path: str):
    """
    Example of validating an existing paper and generating improvement suggestions.
    """
    
    print("=" * 60)
    print("VALIDATING EXISTING PAPER")
    print("=" * 60)
    
    if not os.path.exists(paper_path):
        print(f"❌ Paper not found: {paper_path}")
        return
    
    # Initialize validator
    workflow = QualityEnhancedWorkflow()
    
    # Validate paper
    print(f"📄 Analyzing paper: {paper_path}")
    validation_results, report = workflow.validate_existing_paper(paper_path)
    
    if not validation_results:
        print("❌ Validation failed")
        return
    
    # Calculate overall score
    overall_score = sum(result.score for result in validation_results.values()) / len(validation_results)
    
    print(f"\n📊 OVERALL QUALITY SCORE: {overall_score:.2f}/1.00")
    
    # Show summary
    print(f"\n📈 DETAILED BREAKDOWN:")
    for name, result in validation_results.items():
        status = "✅ PASS" if result.passed else "❌ FAIL"
        severity = f"({result.severity.upper()})" if not result.passed else ""
        print(f"  {name.replace('_', ' ').title()}: {result.score:.2f}/1.00 {status} {severity}")
        
        if result.issues:
            print(f"    Issues: {len(result.issues)} found")
            for issue in result.issues[:2]:  # Show first 2 issues
                print(f"    • {issue}")
            if len(result.issues) > 2:
                print(f"    • ... and {len(result.issues) - 2} more")
    
    # Show recommendations
    needs_improvement = overall_score < 0.6
    if needs_improvement:
        print(f"\n🔧 IMPROVEMENT NEEDED")
        print("Top recommendations:")
        all_suggestions = []
        for result in validation_results.values():
            all_suggestions.extend(result.suggestions)
        
        for i, suggestion in enumerate(all_suggestions[:5], 1):
            print(f"  {i}. {suggestion}")
        
        print(f"\n📋 Full report saved to: {paper_path.replace('.tex', '_quality_report.txt')}")
    else:
        print(f"\n✅ PAPER MEETS QUALITY STANDARDS")
        print("Minor improvements may still be beneficial - check full report")

def main():
    """Main entry point with examples"""
    
    import argparse
    
    parser = argparse.ArgumentParser(description="AI-Scientist with Research Quality Enhancement")
    parser.add_argument("--mode", choices=["generate", "validate"], default="validate",
                      help="Mode: generate new paper or validate existing paper")
    parser.add_argument("--paper-path", type=str, default="output/sample_paper/paper.tex",
                      help="Path to paper for validation mode")
    parser.add_argument("--idea", type=str, help="Research idea for generation mode")
    parser.add_argument("--paper-type", choices=["theoretical", "computational", "experimental"], 
                      default="theoretical", help="Type of paper to generate")
    parser.add_argument("--output-dir", type=str, default="output/enhanced_paper",
                      help="Output directory for generation mode")
    parser.add_argument("--quality-threshold", type=float, default=0.6,
                      help="Quality threshold for acceptance (0-1)")
    
    args = parser.parse_args()
    
    if args.mode == "validate":
        # Validate existing paper
        validate_existing_paper_example(args.paper_path)
        
    elif args.mode == "generate":
        # Generate new paper
        enhanced_ai_scientist_main(
            idea=args.idea,
            paper_type=args.paper_type,
            output_dir=args.output_dir,
            quality_threshold=args.quality_threshold
        )

if __name__ == "__main__":
    main()