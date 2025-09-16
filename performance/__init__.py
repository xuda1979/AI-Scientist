"""Test-time compute scaling and performance analysis utilities."""

# Import the existing implementation and new functions
from .scaling import (test_time_compute_scaling, _generate_best_revision_candidate,
                      _generate_best_initial_draft_candidate, _evaluate_initial_draft_quality)

# Legacy scaling implementation for backward compatibility
import json
import re
import statistics
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def test_time_compute_scaling(
    model: str,
    candidate_counts: List[int] = [3, 5, 7, 10],
    timeout_base: int = 1800,
    test_prompt: str = None
) -> Dict[str, Any]:
    """
    Implement proper test-time compute scaling with candidate generation and selection.
    
    This method generates multiple candidate responses for each iteration and selects
    the best one using quality metrics, demonstrating how additional compute at test time
    can improve performance quality.
    
    Args:
        model: AI model to test (e.g., 'gpt-5', 'gemini-1.5-pro', 'gpt-4o')
        candidate_counts: List of candidate counts to test for scaling analysis
        timeout_base: Base timeout in seconds for API calls
        test_prompt: Custom prompt for testing (uses default if None)
    
    Returns:
        Dictionary containing quality results, scaling analysis, and best candidates
    """
    from ..ai.chat import _universal_chat
    
    print(f"INFO: Starting test-time compute scaling with candidate generation for {model}")
    print(f"INFO: Testing candidate counts: {candidate_counts}")
    
    # Default test prompt if none provided
    if test_prompt is None:
        test_prompt = (
            "Design an efficient algorithm for solving the traveling salesman problem "
            "for graphs with 20-50 nodes. Provide a detailed explanation of your approach, "
            "analyze its time complexity, and discuss potential optimizations. "
            "Include pseudocode and explain why your solution is superior to basic approaches."
        )
    
    results = {
        "model": model,
        "test_config": {
            "candidate_counts_tested": candidate_counts,
            "base_timeout": timeout_base,
            "test_prompt_length": len(test_prompt)
        },
        "candidate_results": {},
        "quality_analysis": {},
        "scaling_analysis": {},
        "best_candidates": {}
    }
    
    def _evaluate_response_quality(response: str) -> Dict[str, float]:
        """Evaluate response quality using multiple metrics."""
        metrics = {}
        
        # Length and detail score (longer responses often more detailed)
        metrics['length_score'] = min(len(response) / 2000, 1.0)
        
        # Technical depth (presence of technical terms)
        technical_terms = ['algorithm', 'complexity', 'optimization', 'efficiency', 
                          'implementation', 'analysis', 'performance', 'solution', 
                          'approach', 'method', 'technique', 'strategy']
        tech_count = sum(1 for term in technical_terms if term.lower() in response.lower())
        metrics['technical_depth'] = min(tech_count / 8, 1.0)
        
        # Structure score (presence of organized sections)
        structure_indicators = ['step', 'first', 'second', 'third', 'finally', 
                               'algorithm:', 'approach:', 'solution:', 'analysis:',
                               '1.', '2.', '3.', 'pseudocode', 'complexity:']
        structure_count = sum(1 for indicator in structure_indicators 
                             if indicator.lower() in response.lower())
        metrics['structure_score'] = min(structure_count / 6, 1.0)
        
        # Code/pseudocode presence
        code_indicators = ['def ', 'for ', 'while ', 'if ', 'return ', 'function',
                          'procedure', 'begin', 'end', '```', 'algorithm']
        code_count = sum(1 for indicator in code_indicators 
                        if indicator.lower() in response.lower())
        metrics['code_presence'] = min(code_count / 4, 1.0)
        
        # Mathematical notation (for algorithmic problems)
        math_indicators = ['O(', '(', '(', 'log', 'n', 'n^2', 'exponential', 
                          'polynomial', 'linear', 'quadratic', 'complexity']
        math_count = sum(1 for indicator in math_indicators 
                        if indicator.lower() in response.lower())
        metrics['math_notation'] = min(math_count / 3, 1.0)
        
        # Overall quality score (weighted average)
        metrics['overall_quality'] = (
            metrics['length_score'] * 0.2 +
            metrics['technical_depth'] * 0.25 +
            metrics['structure_score'] * 0.25 +
            metrics['code_presence'] * 0.15 +
            metrics['math_notation'] * 0.15
        )
        
        return metrics
    
    def _select_best_candidate(candidates: List[str]) -> Tuple[str, Dict[str, float], int]:
        """Select the best candidate based on quality metrics."""
        best_candidate = ""
        best_score = -1
        best_index = -1
        best_metrics = {}
        
        for i, candidate in enumerate(candidates):
            metrics = _evaluate_response_quality(candidate)
            if metrics['overall_quality'] > best_score:
                best_score = metrics['overall_quality']
                best_candidate = candidate
                best_index = i
                best_metrics = metrics
        
        return best_candidate, best_metrics, best_index
    
    # Test each candidate count
    for candidate_count in candidate_counts:
        print(f"INFO: Testing {candidate_count} candidates...")
        
        # Prepare test messages
        test_messages = [
            {"role": "system", "content": "You are an expert computer scientist and algorithm designer. Provide detailed, technically accurate solutions with clear explanations."},
            {"role": "user", "content": test_prompt}
        ]
        
        # Generate multiple candidates
        candidates = []
        generation_times = []
        
        print(f"INFO: Generating {candidate_count} candidate responses...")
        
        for i in range(candidate_count):
            print(f"    Generating candidate {i + 1}/{candidate_count}...")
            
            start_time = time.time()
            
            try:
                # Add slight variation to encourage diverse responses
                varied_messages = test_messages.copy()
                if i > 0:
                    variation_prompts = [
                        " Focus on a different algorithmic approach.",
                        " Emphasize implementation details and optimizations.",
                        " Provide alternative solutions and compare them.",
                        " Include more mathematical analysis and proofs.",
                        " Focus on practical considerations and real-world applications."
                    ]
                    variation = variation_prompts[i % len(variation_prompts)]
                    varied_messages[-1]["content"] += variation
                
                # Make API call
                response = _universal_chat(
                    messages=varied_messages,
                    model=model,
                    request_timeout=timeout_base,
                    prompt_type="test_scaling"
                )
                
                generation_time = time.time() - start_time
                candidates.append(response)
                generation_times.append(generation_time)
                
                print(f"       Candidate {i + 1} generated: {generation_time:.2f}s, {len(response)} chars")
                
            except Exception as e:
                print(f"       Candidate {i + 1} failed: {e}")
                candidates.append("")
                generation_times.append(None)
        
        # Evaluate all candidates and select the best
        valid_candidates = [c for c in candidates if c.strip()]
        
        if valid_candidates:
            print(f"  Selecting best candidate from {len(valid_candidates)} valid responses...")
            
            best_candidate, best_metrics, best_index = _select_best_candidate(valid_candidates)
            
            # Evaluate all candidates for comparison
            all_metrics = []
            for candidate in valid_candidates:
                metrics = _evaluate_response_quality(candidate)
                all_metrics.append(metrics['overall_quality'])
            
            results["candidate_results"][candidate_count] = {
                "total_candidates": candidate_count,
                "successful_candidates": len(valid_candidates),
                "best_candidate_index": best_index,
                "best_quality_score": best_metrics['overall_quality'],
                "average_quality_score": statistics.mean(all_metrics),
                "quality_improvement": best_metrics['overall_quality'] - statistics.mean(all_metrics) if len(all_metrics) > 1 else 0,
                "std_dev_quality": statistics.stdev(all_metrics) if len(all_metrics) > 1 else 0,
                "generation_times": [t for t in generation_times if t is not None],
                "total_compute_time": sum(t for t in generation_times if t is not None),
                "best_metrics_breakdown": best_metrics
            }
            
            results["best_candidates"][candidate_count] = {
                "content": best_candidate,
                "quality_score": best_metrics['overall_quality'],
                "metrics": best_metrics
            }
            
            print(f"    Best candidate quality: {best_metrics['overall_quality']:.3f}")
            print(f"    Average quality: {statistics.mean(all_metrics):.3f}")
            print(f"      Quality improvement: {best_metrics['overall_quality'] - statistics.mean(all_metrics):.3f}")
            
        else:
            results["candidate_results"][candidate_count] = {
                "error": "All candidates failed",
                "total_candidates": candidate_count,
                "successful_candidates": 0
            }
    
    # Analyze quality scaling with compute
    print(f"\nAnalyzing quality scaling with test-time compute...")
    
    successful_results = {k: v for k, v in results["candidate_results"].items() 
                         if isinstance(v, dict) and "best_quality_score" in v}
    
    if len(successful_results) >= 2:
        candidate_counts_sorted = sorted(successful_results.keys())
        quality_scores = [successful_results[k]["best_quality_score"] for k in candidate_counts_sorted]
        compute_times = [successful_results[k]["total_compute_time"] for k in candidate_counts_sorted]
        
        # Calculate quality improvement per additional candidate
        quality_improvements = []
        compute_ratios = []
        
        for i in range(1, len(candidate_counts_sorted)):
            quality_improvement = quality_scores[i] - quality_scores[i-1]
            candidate_ratio = candidate_counts_sorted[i] / candidate_counts_sorted[i-1]
            compute_ratio = compute_times[i] / compute_times[i-1] if compute_times[i-1] > 0 else 1
            
            quality_improvements.append(quality_improvement)
            compute_ratios.append(compute_ratio)
        
        avg_quality_per_candidate = statistics.mean(quality_improvements) if quality_improvements else 0
        avg_compute_scaling = statistics.mean(compute_ratios) if compute_ratios else 1
        
        # Determine scaling efficiency
        efficiency_ratio = avg_quality_per_candidate * 10 / avg_compute_scaling  # Scale for readability
        
        if efficiency_ratio > 0.5:
            scaling_efficiency = "excellent"
        elif efficiency_ratio > 0.2:
            scaling_efficiency = "good"
        elif efficiency_ratio > 0.1:
            scaling_efficiency = "moderate"
        else:
            scaling_efficiency = "poor"
        
        results["scaling_analysis"] = {
            "quality_improvement_per_candidate": avg_quality_per_candidate,
            "compute_scaling_factor": avg_compute_scaling,
            "efficiency_ratio": efficiency_ratio,
            "scaling_efficiency": scaling_efficiency,
            "max_quality_achieved": max(quality_scores),
            "quality_variance": statistics.stdev(quality_scores) if len(quality_scores) > 1 else 0,
            "optimal_candidate_count": candidate_counts_sorted[quality_scores.index(max(quality_scores))]
        }
        
        # Generate recommendations for test-time compute scaling
        results["recommendations"] = {
            "recommended_candidate_count": max(3, min(7, results["scaling_analysis"]["optimal_candidate_count"])),
            "quality_ceiling": max(quality_scores),
            "compute_budget_per_query": statistics.mean(compute_times),
            "scaling_notes": f"Test-time compute scaling shows {scaling_efficiency} efficiency with quality improvements of {avg_quality_per_candidate:.3f} per additional candidate"
        }
    
    else:
        results["scaling_analysis"] = {"error": "Insufficient data for scaling analysis"}
        results["recommendations"] = {"error": "Cannot generate recommendations due to insufficient data"}
    
    # Print comprehensive summary
    print(f"\nTest-Time Compute Scaling Summary for {model}")
    print(f"=" * 50)
    
    for candidate_count, result in results["candidate_results"].items():
        if "best_quality_score" in result:
            print(f"{candidate_count:2d} candidates: Quality {result['best_quality_score']:.3f} "
                  f"(+{result['quality_improvement']:.3f} improvement) "
                  f"Time: {result['total_compute_time']:.1f}s")
        else:
            print(f"{candidate_count:2d} candidates: FAILED")
    
    if "quality_improvement_per_candidate" in results["scaling_analysis"]:
        analysis = results["scaling_analysis"]
        print(f"\nQuality scaling: +{analysis['quality_improvement_per_candidate']:.3f} per candidate")
        print(f"Efficiency: {analysis['scaling_efficiency']} ({analysis['efficiency_ratio']:.2f})")
        print(f"Peak quality: {analysis['max_quality_achieved']:.3f}")
        print(f"Optimal count: {analysis['optimal_candidate_count']} candidates")
    
    if "recommended_candidate_count" in results["recommendations"]:
        rec = results["recommendations"]
        print(f"\nRecommended candidates: {rec['recommended_candidate_count']}")
        print(f"Expected compute time: {rec['compute_budget_per_query']:.1f}s per query")
    
    print(f"=" * 50)
    
    return results
