# Prompt Template Files

Prompt templates used by the AI workflow are stored as plain-text files in the top-level `prompts/` directory. Each template corresponds to a `.txt` file whose filename matches the helper name in `src/ai/prompts.py` (for example, `initial_draft` uses `prompts/initial_draft.txt`).

## File Format

* Files must be UTF-8 encoded text.
* Use standard Markdown formatting when helpful (headings, lists, fenced code blocks, etc.).
* Runtime parameters are inserted with Python's `str.format` syntax:
  * Wrap placeholder names in curly braces, e.g. `{topic}` or `{num_ideas}`.
  * Literal curly braces must be escaped by doubling them (e.g. `{{example}}`).
* Optional sections can be represented by placeholders whose value may be an empty string. For instance, `initial_draft.txt` ends with `{additional_requirements}`, which either expands to a blank string or to the contributor's additional instructions preceded by leading newlines.
* Avoid trailing spaces and keep indentation consistent—what you write is exactly what will be sent to the language model.

## Adding or Updating Templates

1. Create or edit the appropriate `.txt` file in `prompts/`.
2. Ensure that every placeholder used in the file is provided by `PromptTemplates` in `src/ai/prompts.py`.
3. When introducing a new template function, add a matching helper in `PromptTemplates` and reuse `_load_template` to benefit from on-disk caching.
4. Run tests or linting as required by the contribution to confirm that the new template renders correctly.

This approach keeps long-form prompt text out of the Python codebase, simplifies maintenance, and allows contributors to tweak prompt copy without touching application logic.
