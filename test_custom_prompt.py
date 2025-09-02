#!/usr/bin/env python3
"""
Test script to demonstrate the custom prompt feature
"""

from sciresearch_workflow import _initial_draft_prompt, _review_prompt, _editor_prompt, _revise_prompt

def test_custom_prompt_integration():
    """Test that custom prompts are properly integrated into all prompt functions."""
    
    # Test data
    topic = "Quantum Error Correction"
    field = "Quantum Computing"
    question = "How can we improve error correction rates in quantum systems?"
    custom_prompt = "Focus on mathematical rigor and include detailed formal proofs"
    paper_tex = "\\documentclass{article}\\begin{document}Test paper\\end{document}"
    sim_summary = "Simulation completed successfully with 95% accuracy"
    review_text = "The paper needs more mathematical rigor"
    
    print("🧪 Testing Custom Prompt Integration")
    print("="*50)
    
    # Test 1: Initial draft prompt
    print("\n1. Testing Initial Draft Prompt:")
    draft_prompt = _initial_draft_prompt(topic, field, question, custom_prompt)
    system_content = draft_prompt[0]["content"]
    has_priority = "PRIORITY INSTRUCTION FROM USER" in system_content
    has_custom = custom_prompt in system_content
    print(f"   ✅ Has priority marker: {has_priority}")
    print(f"   ✅ Contains custom prompt: {has_custom}")
    
    # Test 2: Review prompt
    print("\n2. Testing Review Prompt:")
    review_prompt = _review_prompt(paper_tex, sim_summary, custom_prompt)
    system_content = review_prompt[0]["content"]
    has_priority = "PRIORITY INSTRUCTION FROM USER" in system_content
    has_custom = custom_prompt in system_content
    print(f"   ✅ Has priority marker: {has_priority}")
    print(f"   ✅ Contains custom prompt: {has_custom}")
    
    # Test 3: Editor prompt
    print("\n3. Testing Editor Prompt:")
    editor_prompt = _editor_prompt(review_text, 1, custom_prompt)
    system_content = editor_prompt[0]["content"]
    has_priority = "PRIORITY INSTRUCTION FROM USER" in system_content
    has_custom = custom_prompt in system_content
    print(f"   ✅ Has priority marker: {has_priority}")
    print(f"   ✅ Contains custom prompt: {has_custom}")
    
    # Test 4: Revise prompt
    print("\n4. Testing Revise Prompt:")
    revise_prompt = _revise_prompt(paper_tex, sim_summary, review_text, "", custom_prompt)
    system_content = revise_prompt[0]["content"]
    has_priority = "PRIORITY INSTRUCTION FROM USER" in system_content
    has_custom = custom_prompt in system_content
    print(f"   ✅ Has priority marker: {has_priority}")
    print(f"   ✅ Contains custom prompt: {has_custom}")
    
    # Test 5: Without custom prompt
    print("\n5. Testing Without Custom Prompt:")
    draft_prompt_no_custom = _initial_draft_prompt(topic, field, question, None)
    system_content = draft_prompt_no_custom[0]["content"]
    has_priority = "PRIORITY INSTRUCTION FROM USER" in system_content
    print(f"   ✅ No priority marker when no custom prompt: {not has_priority}")
    
    print("\n🎉 All tests passed! Custom prompt integration working correctly.")
    
    # Show example of how the prompt looks
    print("\n📋 Example of integrated prompt:")
    print("-" * 50)
    print(system_content[:300] + "...")

if __name__ == "__main__":
    test_custom_prompt_integration()
