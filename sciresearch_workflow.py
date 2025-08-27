#!/usr/bin/env python3
import os
import argparse
from pathlib import Path
import difflib
import re
from datetime import datetime
import time

from openai import OpenAI


client: OpenAI | None = None


def call_openai(
    model: str,
    prompt: str,
    *,
    timeout: int | None = 3600,
    max_retries: int = 3,
):
    """Call the OpenAI chat completions API with retry and timeout.

    Parameters
    ----------
    model: str
        Name of the model to query.
    prompt: str
        The content to send as a user message.
    timeout: int | None, optional
        Maximum time to wait for a response in seconds (default: ``3600``).
        Pass ``None`` to wait indefinitely.
    max_retries: int, optional
        Number of times to retry the request on failure (default: 3).
    """
    if client is None:
        raise RuntimeError("OpenAI client has not been initialized")

    last_exception: Exception | None = None
    for attempt in range(1, max_retries + 1):
        try:
            wait_msg = (
                f"Waiting up to {timeout} seconds" if timeout is not None else "Waiting indefinitely"
            )
            print(
                f"Calling OpenAI (attempt {attempt}/{max_retries})...\n{wait_msg} for {model} to respond."
            )
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                timeout=timeout,
            )
            return response.choices[0].message.content.strip()
        except Exception as exc:  # broad except to catch networking/timeout errors
            last_exception = exc
            if attempt < max_retries:
                sleep_time = 2 ** (attempt - 1)
                print(
                    f"OpenAI request failed: {exc}. Retrying in {sleep_time} seconds..."
                )
                time.sleep(sleep_time)
    # If we reach here, all retries failed
    raise RuntimeError(
        f"OpenAI request failed after {max_retries} attempts: {last_exception}"
    ) from last_exception


def generate_idea(
    topic: str,
    field: str,
    question: str,
    model: str,
    *,
    timeout: int | None,
    max_retries: int,
) -> str:
    """Generate a high-value research idea for the given topic, field and question."""
    prompt = (

        f"Provide a high-value, innovative and practical research idea for the topic: {topic}. "
        "Ensure the idea lends itself to rigorous methodology and clear exposition. "
        "Respond with a concise description."

    )
    return call_openai(model, prompt, timeout=timeout, max_retries=max_retries)


def write_paper(
    topic: str,
    field: str,
    question: str,
    idea: str,
    model: str,
    *,
    timeout: int | None,
    max_retries: int,
) -> str:
    """Draft a full research paper given a topic, field, research question and idea."""
    prompt = (
        "You are an expert researcher. Write a full research paper based on the following information.\n"
        f"Field: {field}\n"
        f"Topic: {topic}\n"
        f"Research Question: {question}\n"
        f"Idea: {idea}\n\n"
 
        "The paper should include the following sections: Abstract, Introduction, Related Work, Methodology, "
        "Experiments, Results, Discussion, Conclusion, References.\n"
        "Use LaTeX formatting for a full-length paper, employing appropriate section commands. \n"
        "If code is necessary for the experiments, include it using the LaTeX lstlisting environment with "
        "language=Python.\n"
        "Ensure rigorous methodology and clear exposition throughout."
 
    )
    return call_openai(model, prompt, timeout=timeout, max_retries=max_retries)


def review_paper(
    paper_content: str,
    model: str,
    *,
    timeout: int | None,
    max_retries: int,
) -> str:
    """Ask the model to review the paper and give constructive feedback.

    The reviewer must not recommend rejection; instead it should provide
    actionable suggestions that help the authors improve the work to meet
    top‑journal standards.
    """
    prompt = (
        "You are a top journal peer reviewer. Review the following research paper and provide "
        "constructive, actionable feedback on its novelty, clarity of exposition, rigor of methodology, and significance. "
        "Do not recommend rejection; instead, offer specific suggestions to improve the work so it can meet top-journal standards.\n\n"
        f"Paper:\n{paper_content}"
    )
    return call_openai(model, prompt, timeout=timeout, max_retries=max_retries)


def evaluate_paper(
    paper_content: str,
    feedback: str,
    model: str,
    *,
    timeout: int | None,
    max_retries: int,
) -> str:
    """Decide if the paper is ready for submission.

    An editor response of ``NO`` must include clear guidance on how to
    improve the paper rather than rejecting it outright.
    """
    prompt = (
        "You are a journal editor evaluating a paper and the peer review feedback. Decide whether the paper "
        "is of top quality and can be submitted to a top journal without any modifications. Respond with 'YES' "
        "if it is ready. If it is not ready, respond with 'NO' and provide specific suggestions to improve the paper rather than rejecting it.\n\n"
        f"Paper:\n{paper_content}\n\n"
        f"Review Feedback:\n{feedback}"
    )
    return call_openai(model, prompt, timeout=timeout, max_retries=max_retries)


def revise_paper(
    paper_content: str,
    feedback: str,
    model: str,
    *,
    timeout: int | None,
    max_retries: int,
) -> str:
    """Revise the paper based on review feedback."""
    prompt = (
        "Based on the reviewer feedback provided, revise the research paper to address all issues and improve its "
 
        "quality. Provide the complete revised paper as a LaTeX document using \\documentclass{article}. Use the "
        "lstlisting environment for any Python code blocks. Do not include any explanations, only the revised paper.\n\n"
 
        f"Original Paper:\n{paper_content}\n\n"
        f"Review Feedback:\n{feedback}"
    )
    return call_openai(model, prompt, timeout=timeout, max_retries=max_retries)


def save_paper_and_code(paper_content: str, output_dir: Path | str) -> Path:
    """Save the paper and Python snippets to a unique subdirectory.

    A timestamped subfolder is created inside ``output_dir`` so multiple
    papers can coexist. If ``output_dir`` already points to an existing paper
    directory (containing ``paper.tex``), files are updated in place. Python
    code blocks contained in LaTeX ``lstlisting`` environments are extracted to
    ``code_<n>.py`` files. The directory where ``paper.tex`` and code files are
    written is returned for subsequent steps.
    """
    base_path = Path(output_dir)

    if (base_path / "paper.tex").exists():
        paper_dir = base_path
    else:
        base_path.mkdir(parents=True, exist_ok=True)
        paper_dir = base_path / datetime.now().strftime("%Y%m%d_%H%M%S")
        paper_dir.mkdir(parents=True, exist_ok=True)

    paper_path = paper_dir / "paper.tex"

    paper_path.write_text(paper_content, encoding="utf-8")

    # extract python code blocks from LaTeX lstlisting environments
    code_blocks = re.findall(
        r"\\begin{lstlisting}(?:\[[^\]]*\])?\s*(.*?)\s*\\end{lstlisting}",
        paper_content,
        re.DOTALL,
    )

    for idx, code in enumerate(code_blocks, 1):
        code_file = paper_dir / f"code_{idx}.py"
        code_file.write_text(code.strip(), encoding="utf-8")
    return paper_dir


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
        description="Run a simple sci research workflow using OpenAI models.",
    )
    parser.add_argument("--topic", required=True, help="Research topic.")
    parser.add_argument(
        "--output-dir", default="output", help="Directory to store the generated paper and code.",
    )
    parser.add_argument(
        "--model", default="gpt-5", help="OpenAI model to use (default: gpt-5).",
    )
 
    parser.add_argument("--field", help="Research field.")
    parser.add_argument("--question", help="Specific research question to address.")
 
    parser.add_argument(
        "--max-iters",
        type=int,
        default=3,
        help="Maximum number of review/evaluation/revision cycles (default: 3)",
    )

    parser.add_argument(
        "--request-timeout",
        type=int,
        default=3600,
        help="Timeout in seconds for each OpenAI request (default: 3600; use 0 to wait indefinitely)",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=3,
        help="Number of times to retry a failed OpenAI request (default: 3)",
    )

    args = parser.parse_args()

    if args.request_timeout == 0:
        args.request_timeout = None

    if not args.field:
        args.field = input("Enter research field: ").strip()
    if not args.question:
        args.question = input("Enter research question: ").strip()

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError("OPENAI_API_KEY environment variable is not set.")
    global client
    client_kwargs = {
        "api_key": api_key,
        "max_retries": args.max_retries,
    }
    if args.request_timeout is not None:
        client_kwargs["timeout"] = args.request_timeout
    client = OpenAI(**client_kwargs)

 
    try:
        # Step 1: Generate research idea
        idea = generate_idea(
            args.topic,
            args.field,
            args.question,
            args.model,
            timeout=args.request_timeout,
            max_retries=args.max_retries,
        )
        print(f"Generated Idea:\n{idea}\n")

        # Step 2: Write research paper
        paper_content = write_paper(
            args.topic,
            args.field,
            args.question,
            idea,
            args.model,
            timeout=args.request_timeout,
            max_retries=args.max_retries,
        )
        paper_dir = save_paper_and_code(paper_content, args.output_dir)
        paper_path = paper_dir / "paper.tex"
        print(f"Initial paper saved to {paper_path}")

        for iteration in range(1, args.max_iters + 1):
            print(f"\n--- Review Cycle {iteration} ---")
            feedback = review_paper(
                paper_content,
                args.model,
                timeout=args.request_timeout,
                max_retries=args.max_retries,
            )
            print(f"Reviewer Feedback:\n{feedback}\n")

            decision = evaluate_paper(
                paper_content,
                feedback,
                args.model,
                timeout=args.request_timeout,
                max_retries=args.max_retries,
            )
            print(f"Editor Decision: {decision}\n")
            if decision.strip().upper().startswith("YES"):
                print("Paper is ready for submission. Workflow completed.")
                break

            combined_feedback = feedback + "\n\nEditor Suggestions:\n" + decision
            revised_content = revise_paper(
                paper_content,
                combined_feedback,
                args.model,
                timeout=args.request_timeout,
                max_retries=args.max_retries,
            )
            diff_text = apply_diff_and_save(paper_path, revised_content)
            save_paper_and_code(revised_content, paper_dir)
            print("Paper revised and saved. Diff between versions:\n")
            print(diff_text)
            paper_content = revised_content
        else:
            print("Maximum review iterations reached without editor approval.")
    except KeyboardInterrupt:
        print("Workflow interrupted by user.")
 
 


if __name__ == "__main__":
    main()
