# GUI vs Command-Line Feature Parity Analysis

## Current Status

The GUI (ui/gui_app.py) is **ALMOST** feature-complete but is missing ONE command-line option.

## Missing Feature in GUI

### 1. Blueprint Planning Control ❌
**Command-line argument:** `--disable-blueprint-planning`
- **Type:** Boolean flag (action="store_true")
- **Default:** False (planning is enabled by default)
- **Description:** Skip the research blueprint planning step before drafting
- **Location in CLI:** Line 4505-4508

**Required GUI Changes:**
- Add checkbox in Execution Settings or Advanced Options frame
- Add to `self.vars` dictionary
- Add to `_gather_parameters()` method
- Pass to `run_workflow()` function

## Complete Feature Comparison

| Feature | Command-Line | GUI | Status |
|---------|--------------|-----|--------|
| **Project Details** |
| Topic | ✅ --topic | ✅ topic | ✅ |
| Field | ✅ --field | ✅ field | ✅ |
| Research Question | ✅ --question | ✅ question | ✅ |
| Document Type | ✅ --document-type | ✅ document_type | ✅ |
| Output Directory | ✅ --output-dir | ✅ output_dir | ✅ |
| Model | ✅ --model | ✅ model | ✅ |
| **Execution Settings** |
| Request Timeout | ✅ --request-timeout | ✅ request_timeout | ✅ |
| Max Retries | ✅ --max-retries | ✅ max_retries | ✅ |
| Max Iterations | ✅ --max-iterations | ✅ max_iterations | ✅ |
| No Early Stopping | ✅ --no-early-stopping | ✅ no_early_stopping | ✅ |
| Modify Existing | ✅ --modify-existing | ✅ modify_existing | ✅ |
| Strict Singletons | ✅ --strict-singletons | ✅ strict_singletons | ✅ |
| Python Executable | ✅ --python-exec | ✅ python_exec | ✅ |
| Config File | ✅ --config | ✅ config_path | ✅ |
| Save Config | ✅ --save-config | ✅ save_config_path | ✅ |
| **Blueprint Planning** |
| Disable Blueprint Planning | ✅ --disable-blueprint-planning | ❌ **MISSING** | ❌ |
| **Quality & Validation** |
| Quality Threshold | ✅ --quality-threshold | ✅ quality_threshold | ✅ |
| Check References | ✅ --check-references | ✅ check_references | ✅ |
| Skip Reference Check | ✅ --skip-reference-check | ✅ skip_reference_check | ✅ |
| Validate Figures | ✅ --validate-figures | ✅ validate_figures | ✅ |
| Skip Figure Validation | ✅ --skip-figure-validation | ✅ skip_figure_validation | ✅ |
| Enable PDF Review | ✅ --enable-pdf-review | ✅ enable_pdf_review | ✅ |
| Disable PDF Review | ✅ --disable-pdf-review | ✅ disable_pdf_review | ✅ |
| Enable Ideation | ✅ --enable-ideation | ✅ enable_ideation | ✅ |
| Skip Ideation | ✅ --skip-ideation | ✅ skip_ideation | ✅ |
| Specify Idea | ✅ --specify-idea | ✅ specify_idea | ✅ |
| Number of Ideas | ✅ --num-ideas | ✅ num_ideas | ✅ |
| **User Customization** |
| Custom User Prompt | ✅ --user-prompt | ✅ user_prompt (text box) | ✅ |
| **Output & Tracking** |
| Save Output Diffs | ✅ --output-diffs | ✅ output_diffs | ✅ |
| Disable Output Diffs | ✅ --no-output-diffs | ✅ no_output_diffs | ✅ |
| **Content Protection** |
| Disable Content Protection | ✅ --disable-content-protection | ✅ disable_content_protection | ✅ |
| Auto-Approve Changes | ✅ --auto-approve-changes | ✅ auto_approve_changes | ✅ |
| Content Protection Threshold | ✅ --content-protection-threshold | ✅ content_protection_threshold | ✅ |
| **Test-Time Scaling** |
| Enable Test Scaling Mode | ✅ --test-scaling | ✅ test_scaling | ✅ |
| Scaling Prompt | ✅ --scaling-prompt | ✅ scaling_prompt | ✅ |
| Scaling Candidates | ✅ --scaling-candidates | ✅ scaling_candidates | ✅ |
| Scaling Timeout | ✅ --scaling-timeout | ✅ scaling_timeout | ✅ |
| Use Test-Time Scaling | ✅ --use-test-time-scaling | ✅ use_test_time_scaling | ✅ |
| Revision Candidates | ✅ --revision-candidates | ✅ revision_candidates | ✅ |
| Draft Candidates | ✅ --draft-candidates | ✅ draft_candidates | ✅ |

## Summary

- **Total Command-Line Features:** 43
- **GUI Features Implemented:** 42
- **Missing in GUI:** 1 (--disable-blueprint-planning)
- **Completion Rate:** 97.7%

## Implementation Plan

### Step 1: Add GUI Control for Blueprint Planning
Add checkbox to Execution Settings frame:
```python
self._add_check(frame, "Disable Blueprint Planning", "disable_blueprint_planning", default=False, row=9)
```

### Step 2: Pass to Config
Update `_prepare_config()` to handle the new parameter:
```python
config.disable_blueprint_planning = bool(params["disable_blueprint_planning"])
```

### Step 3: Update _gather_parameters()
Add to the parameters dictionary:
```python
"disable_blueprint_planning": bool(self.vars["disable_blueprint_planning"].get()),
```

### Step 4: Pass to run_workflow()
Ensure the parameter is passed when calling `run_workflow()`:
```python
result_dir = run_workflow(
    ...
    disable_blueprint_planning=bool(params["disable_blueprint_planning"]),
    ...
)
```

## Testing Checklist

After implementation:
- [ ] GUI checkbox appears in Execution Settings
- [ ] Checkbox state is correctly read
- [ ] Parameter is passed to run_workflow()
- [ ] Blueprint planning is skipped when checkbox is enabled
- [ ] Workflow completes successfully with blueprint planning disabled
- [ ] Workflow completes successfully with blueprint planning enabled (default)

## Verification

Once implemented, verify 100% parity by:
1. Running workflow from GUI with blueprint planning disabled
2. Running same workflow from command-line with --disable-blueprint-planning
3. Comparing outputs - should be identical
