"""Prompt templates for various workflow steps."""
from .templates import (_initial_draft_prompt, _review_prompt, _revise_prompt, _editor_prompt,
                        _combined_review_edit_revise_prompt)

__all__ = ['_initial_draft_prompt', '_review_prompt', '_revise_prompt', '_editor_prompt',
           '_combined_review_edit_revise_prompt']
