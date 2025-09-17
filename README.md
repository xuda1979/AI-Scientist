# AI-Scientist

## Citation

If you use AI-Scientist in your research, please cite this software as follows:

```
@software{ai_scientist,
  author = {AI-Scientist Developers},
  title = {AI-Scientist},
  year = {2025},
  url = {https://github.com/AI-Scientist/AI-Scientist}
}
```

Please replace the URL or version number with the appropriate release or commit hash you used in your work.

## Google Gemini Configuration

The workflow can call Google Gemini models via the Google AI SDK. To enable these calls:

1. Obtain an API key from the [Google AI Studio](https://ai.google.dev/).
2. Provide the key to the workflow using one of the following options:
   - Set the `GOOGLE_API_KEY` (or `GEMINI_API_KEY`) environment variable before running the workflow.
   - Add a `google_api_key` entry to your workflow configuration JSON (see `config_example.json`).
3. If your environment requires a proxy for outbound HTTPS traffic, set it with the `GOOGLE_API_PROXY` environment variable or the `google_api_proxy` configuration field.

When a Gemini model is selected without a configured API key, the workflow now raises a clear error explaining how to supply the key.
