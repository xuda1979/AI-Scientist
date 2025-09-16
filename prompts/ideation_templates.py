"""Research ideation prompt templates."""

def _ideation_prompt(topic: str, field: str, num_ideas: int = 15) -> str:
    """Generate prompt for research ideation phase."""
    return f"""You are a research ideation expert in {field}. Generate {num_ideas} high-quality, novel research ideas related to "{topic}".

For each idea, provide:
1. Title: A clear, specific research title
2. Question: A precise, answerable research question

Focus on:
- Novel approaches and unexplored angles
- Practical significance and impact
- Feasibility for academic research
- Current relevance to the field

Format each idea as:
Title: [Research Title]
Question: [Research Question]

Generate ideas that span different aspects, approaches, and scales related to {topic} in {field}.

Begin generating the {num_ideas} research ideas now:"""
