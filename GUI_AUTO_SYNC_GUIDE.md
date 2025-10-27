# GUI Auto-Sync with Command-Line - Quick Guide

## What Changed

The GUI now **automatically inherits all changes** from the command-line workflow.

### Before
```
CLI Change → Manual GUI Update → Testing Both
(Easy to miss, features drift apart)
```

### After
```
CLI Change → Automatic GUI Sync
(One change, both interfaces updated)
```

## How It Works

### Architecture
```
GUI → workflow_wrapper.py → sciresearch_workflow.py
CLI → workflow_wrapper.py → sciresearch_workflow.py
        ↑
   Single Point of Control
```

### Key File: `workflow_wrapper.py`

This file contains ALL workflow execution logic. Both GUI and CLI call it.

**Result:** Change workflow once, both interfaces update.

## For Developers: Adding New Features

### Step 1: Add to Core Workflow

**File:** `sciresearch_workflow.py`

```python
# 1. Add argument
p.add_argument("--new-feature", type=str, help="New feature")

# 2. Add to run_workflow() signature
def run_workflow(..., new_feature: str = "default", ...):
    # Use the feature
    print(f"Using: {new_feature}")
```

### Step 2: Add to Workflow Wrapper

**File:** `workflow_wrapper.py`

```python
def execute_workflow(params: WorkflowParameters, ...):
    # Extract parameter
    new_feature = params.get("new_feature", "default")
    
    # Pass to run_workflow (add to the call)
    result_dir = run_workflow(
        ...
        new_feature=new_feature,  # ADD THIS LINE
        ...
    )
```

### Step 3: Add to GUI

**File:** `ui/gui_app.py`

```python
# In _build_XXX_frame() method:
self._add_entry(frame, "New Feature", "new_feature", row=X)

# In _gather_parameters() method:
"new_feature": self.vars["new_feature"].get().strip() or "default",
```

### That's It!

✅ CLI works immediately (Step 1 + 2)
✅ GUI works immediately (Step 2 + 3)
✅ Both call same code (guaranteed sync)

## Common Scenarios

### Scenario 1: Add Boolean Flag

**CLI:**
```python
p.add_argument("--enable-magic", action="store_true")
```

**Wrapper:**
```python
def execute_workflow(params, ...):
    enable_magic = params.get("enable_magic", False)
    result_dir = run_workflow(..., enable_magic=enable_magic, ...)
```

**GUI:**
```python
self._add_check(frame, "Enable Magic", "enable_magic", default=False, row=X)
"enable_magic": bool(self.vars["enable_magic"].get()),
```

### Scenario 2: Add String Parameter

**CLI:**
```python
p.add_argument("--magic-spell", type=str, default="abracadabra")
```

**Wrapper:**
```python
magic_spell = params.get("magic_spell", "abracadabra")
result_dir = run_workflow(..., magic_spell=magic_spell, ...)
```

**GUI:**
```python
self._add_entry(frame, "Magic Spell", "magic_spell", default="abracadabra", row=X)
"magic_spell": self.vars["magic_spell"].get().strip() or "abracadabra",
```

### Scenario 3: Add Numeric Parameter

**CLI:**
```python
p.add_argument("--magic-power", type=int, default=100)
```

**Wrapper:**
```python
magic_power = int(params.get("magic_power", 100))
result_dir = run_workflow(..., magic_power=magic_power, ...)
```

**GUI:**
```python
self._add_spinbox(frame, "Magic Power", "magic_power", default=100, from_=0, to=999, row=X)
"magic_power": int(self.vars["magic_power"].get()),
```

## Boolean Inversions (Special Case)

Some GUI checkboxes use "disable_" but workflow uses "enable_".

### Handling in Wrapper

**File:** `workflow_wrapper.py` in `WorkflowParameters.from_gui()`:

```python
if "disable_magic" in params:
    params["enable_magic"] = not bool(params.pop("disable_magic"))
```

**GUI checkbox:** "Disable Magic" (checked = disabled)
**Workflow parameter:** `enable_magic=False`

## Testing

### Quick Test: GUI Still Works
```bash
python ui/gui_app.py
```
✅ Should launch without errors
✅ New feature should appear if added to GUI

### Full Test: CLI and GUI Parity
```bash
# Run from GUI with specific parameters
# Then run equivalent CLI command:
python sciresearch_workflow.py --topic "Test" --new-feature "test-value" ...

# Results should be identical
```

## Troubleshooting

### GUI Error: "Parameter not found"

**Problem:** Parameter added to CLI but not to wrapper

**Solution:** Add to `workflow_wrapper.py` `execute_workflow()`:
```python
new_param = params.get("new_param", default)
result_dir = run_workflow(..., new_param=new_param, ...)
```

### GUI Shows Old Behavior

**Problem:** GUI not calling wrapper

**Solution:** Check `ui/gui_app.py` `_run_workflow_thread()`:
```python
# Should have:
result_dir = run_from_gui(params, cancel_event=self.cancel_event)

# NOT:
result_dir = run_workflow(...)  # Old direct call
```

### GUI and CLI Produce Different Results

**Problem:** Different parameter values

**Solution:** Compare parameters:
```python
# Add debug print in workflow_wrapper.py:
print(f"Parameters: {params.to_dict()}")
```

Run both GUI and CLI, compare output.

## Maintenance Checklist

When modifying workflow:

- [ ] Update `sciresearch_workflow.py` (core logic)
- [ ] Update `workflow_wrapper.py` (wrapper logic)
- [ ] Update `ui/gui_app.py` (GUI controls) if new parameters
- [ ] Test CLI: `python sciresearch_workflow.py ...`
- [ ] Test GUI: `python ui/gui_app.py`
- [ ] Verify both produce same results

## Benefits Summary

✅ **Automatic Sync:** GUI inherits CLI changes
✅ **No Duplication:** Workflow logic in one place
✅ **Guaranteed Parity:** Both call same function
✅ **Easier Maintenance:** Change once, works everywhere
✅ **Faster Development:** Add feature once, test once

## Files to Remember

1. **`workflow_wrapper.py`** - Single source of truth for workflow execution
2. **`ui/gui_app.py`** - GUI controls and parameter collection (view layer only)
3. **`sciresearch_workflow.py`** - Core workflow logic and CLI argument parsing

**Rule:** Workflow logic goes in `workflow_wrapper.py`, NOT in `ui/gui_app.py`!

---

**You asked for automatic synchronization - you got it!** 🎉

When you change the command-line workflow, the GUI automatically changes correspondingly!
