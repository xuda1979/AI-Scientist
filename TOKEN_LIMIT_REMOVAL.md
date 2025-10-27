# Token Limit Removal - Paper Truncation Fix

## Problem Identified

The AI Scientist workflow was truncating papers during GPT-5-Pro API calls due to restrictive token limits:

1. **Responses API**: Had `max_output_tokens=16000` limit
2. **Chat Completions API**: Had `max_tokens=4000` or `max_completion_tokens=4000` limits

### Impact
When the GPT-5-Pro model was processing the deliberative_compute paper (which had ~771 lines), the response was cut off at approximately line 770, resulting in:
- Missing sections: "Taxonomy and classification of algorithms", "Complexity and compute footprint", "Synthetic evaluation and results"
- Incomplete final proposition
- No proper conclusion, bibliography, or document ending
- Paper compilation errors due to missing `\end{document}`

## Root Cause

The truncation occurred in `sciresearch_workflow.py` at lines:
- **Line 500**: `max_output_tokens=16000` for Responses API (with_options path)
- **Line 506**: `max_output_tokens=16000` for Responses API (direct path)
- **Line 520**: `max_completion_tokens=4000` for gpt-5/o1 models
- **Line 523**: `max_tokens=4000` for other models

## Solution Implemented

### Changes Made to `sciresearch_workflow.py`

1. **Removed `max_output_tokens` from Responses API calls** (lines 500, 506)
   - Now allows model to generate complete responses without artificial limits
   
2. **Removed `max_completion_tokens` from gpt-5/o1 models** (line 520)
   - Allows full paper generation for advanced models
   
3. **Removed `max_tokens` from other models** (line 523)
   - Ensures all models can generate complete papers

### Code Changes

```python
# BEFORE (Responses API):
resp = responses_client.create(
    model=model,
    input=_convert_messages_to_responses_input(processed_messages),
    max_output_tokens=16000,  # This was limiting output
)

# AFTER (Responses API):
resp = responses_client.create(
    model=model,
    input=_convert_messages_to_responses_input(processed_messages),
    # No max_output_tokens limit - let model generate full response
)

# BEFORE (Chat Completions):
if model.startswith("gpt-5") or model.startswith("o1"):
    completion_kwargs["max_completion_tokens"] = 4000
else:
    completion_kwargs["temperature"] = temp
    completion_kwargs["max_tokens"] = 4000

# AFTER (Chat Completions):
if model.startswith("gpt-5") or model.startswith("o1"):
    # No token limit for gpt-5 and o1 models - allow full paper generation
    pass
else:
    completion_kwargs["temperature"] = temp
    # No max_tokens limit - let model generate full response
```

## Benefits

1. **Complete Papers**: Models can now generate complete academic papers without truncation
2. **Natural Endings**: Papers will have proper conclusions, bibliographies, and document structure
3. **Better Quality**: No artificial limits on paper length or complexity
4. **Flexible Output**: Different papers can have different lengths based on content needs

## Considerations

1. **API Costs**: Removing limits may increase token usage and API costs
2. **Processing Time**: Longer responses may take more time to generate
3. **Model Defaults**: Models will use their default maximum context/output limits
4. **Timeout Protection**: The existing timeout settings (3600s for Responses API) still provide protection against runaway generation

## Testing Recommendations

1. Test with a complete paper to verify no truncation occurs
2. Monitor API usage and costs
3. Verify LaTeX compilation succeeds with complete output
4. Check that bibliography and references are properly included

## Date Applied
October 26, 2025

## Files Modified
- `c:\Users\Lenovo\software\AI-Scientist\sciresearch_workflow.py`
