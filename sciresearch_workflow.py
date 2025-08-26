#!/usr/bin/env python3
import os
import argparse
import openai
from pathlib import Path
import difflib
import re


def call_openai(model, prompt):
    """Call the OpenAI ChatCompletion API with a single user prompt."""
    response = openai.ChatCompletion.create(
        model=model,
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content.strip()


def generate_idea(topic: str, model: str) -> str:
    """Generate a high-value research idea for the given topic."""
    prompt = (
        f"Provide a high-value, innovative and practical research idea for the topic: {topic}. "
        "Ensure the idea lends itself to rigorous methodology and clear exposition. "
        "Respond with a concise description."
    )
    return call_openai(model, prompt)


def write_paper(topic: str, idea: str, model: str) -> str:
    """Draft a full research paper given a topic and research idea."""
    prompt = (
        "You are an expert researcher. Write a full research paper based on the following topic and idea.\n"
        f"Topic: {topic}\n"
        f"Idea: {idea}\n\n"
        "The paper should include the following sections: Abstract, Introduction, Related Work, Methodology, "
        "Experiments, Results, Discussion, Conclusion, References.\n"
        "Use LaTeX formatting for a full-length paper, employing appropriate section commands. \n"
        "If code is necessary for the experiments, include it using the LaTeX lstlisting environment with "
        "language=Python.\n"
        "Ensure rigorous methodology and clear exposition throughout."
    )
    return call_openai(model, prompt)


def review_paper(paper_content: str, model: str) -> str:
    """Ask the model to review the paper as a top journal peer reviewer."""
    prompt = (
        "You are a top journal peer reviewer. Please review the following research paper and provide "
        "constructive feedback on its novelty, clarity of exposition, rigor of methodology, and significance. "
        "Identify any issues that need to be addressed before publication.\n\n"
        f"Paper:\n{paper_content}"
    )
    return call_openai(model, prompt)


def evaluate_paper(paper_content: str, feedback: str, model: str) -> str:
    """Determine if the paper is ready for submission based on the review feedback."""
    prompt = (
        "You are a journal editor evaluating a paper and the peer review feedback. Decide whether the paper "
        "is of top quality and can be submitted to a top journal without any modifications. Respond with 'YES' "
        "if it is ready, otherwise respond with 'NO' and briefly explain why.\n\n"
        f"Paper:\n{paper_content}\n\n"
        f"Review Feedback:\n{feedback}"
    )
    return call_openai(model, prompt)


def revise_paper(paper_content: str, feedback: str, model: str) -> str:
    """Revise the paper based on review feedback."""
    prompt = (
        "Based on the reviewer feedback provided, revise the research paper to address all issues and improve its "
        "quality. Ensure rigorous methodology and clear exposition. Provide the complete revised paper in LaTeX "
        "format. Do not include any explanations, only the revised paper.\n\n"
        f"Original Paper:\n{paper_content}\n\n"
        f"Review Feedback:\n{feedback}"
    )
    return call_openai(model, prompt)


def save_paper_and_code(paper_content: str, output_dir: str) -> Path:
    """Save the paper and any Python code blocks to the output directory."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    paper_path = output_path / "paper.tex"
    paper_path.write_text(paper_content, encoding="utf-8")
    # extract python code blocks from LaTeX lstlisting or Markdown fences
    code_blocks = re.findall(r"\\begin{lstlisting}.*?\n(.*?)\\end{lstlisting}", paper_content, re.DOTALL)
    if not code_blocks:
        code_blocks = re.findall(r"```python(.*?)```", paper_content, re.DOTALL)
    for idx, code in enumerate(code_blocks, 1):
        code_file = output_path / f"code_{idx}.py"
        code_file.write_text(code.strip(), encoding="utf-8")
    return paper_path


def apply_diff_and_save(original_path: Path, new_content: str) -> str:
    """Apply modifications to the original file and return a unified diff."""
    original = original_path.read_text(encoding="utf-8")
    diff = difflib.unified_diff(
        original.splitlines(),
        new_content.splitlines(),
        fromfile=str(original_path),
        tofile="revised_" + original_path.name,
        lineterm=""
    )
    diff_text = "\n".join(diff)
    # write revised content
    original_path.write_text(new_content, encoding="utf-8")
    return diff_text


def main():
    parser = argparse.ArgumentParser(
        description="Run a simple sci research workflow using OpenAI models."
    )
    parser.add_argument("--topic", required=True, help="Research topic.")
    parser.add_argument(
        "--output-dir", default="output", help="Directory to store the generated paper and code."
    )
    parser.add_argument(
        "--model", default="gpt-5", help="OpenAI model to use (default: gpt-5)."
    )
    parser.add_argument(
        "--max-iters",
        type=int,
        default=3,
        help="Maximum number of review/evaluation/revision cycles (default: 3)",
    )
    args = parser.parse_args()

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError("OPENAI_API_KEY environment variable is not set.")
    openai.api_key = api_key

    # Step 1: Generate research idea
    idea = generate_idea(args.topic, args.model)
    print(f"Generated Idea:\n{idea}\n")

    # Step 2: Write research paper
    paper_content = write_paper(args.topic, idea, args.model)
    paper_path = save_paper_and_code(paper_content, args.output_dir)
    print(f"Initial paper saved to {paper_path}")

    for iteration in range(1, args.max_iters + 1):
        print(f"\n--- Review Cycle {iteration} ---")
        feedback = review_paper(paper_content, args.model)
        print(f"Reviewer Feedback:\n{feedback}\n")

        decision = evaluate_paper(paper_content, feedback, args.model)
        print(f"Editor Decision: {decision}\n")
        if decision.strip().upper().startswith("YES"):
            print("Paper is ready for submission. Workflow completed.")
            break

        revised_content = revise_paper(paper_content, feedback, args.model)
        diff_text = apply_diff_and_save(paper_path, revised_content)
        save_paper_and_code(revised_content, args.output_dir)
        print("Paper revised and saved. Diff between versions:\n")
        print(diff_text)
        paper_content = revised_content
    else:
        print("Maximum review iterations reached without editor approval.")


if __name__ == "__main__":
    main()
