# Physics Hook Implementation: Summary of Changes

## Completed Changes (2025-11-20)

Successfully implemented a clean separation between the **statistical toy model** (Haar scrambler) and the **PT–MPO infrastructure** in `simulation.py`, with a hook to plug in first-principles microscopic dynamics.

---

## Modified Files

### `simulation.py`

**Lines 1–34**: Updated module header documentation
- Clearly labels the default dynamics as a "statistical toy model"
- Explains the toy model is for stress-testing structural results, not solving 4D GR+QFT
- Documents the `set_physical_step_update` hook for first-principles models

**Lines 43–73**: Added `set_physical_step_update(fn)` hook
- Global variable `_PhysicalStepUpdate` to store the callback
- Function signature: `fn(S_prev, target_chi, rng) -> s_vals`
- Comprehensive docstring explaining usage and expectations

**Lines 168–199**: Modified `run_tensor_step(S_prev, target_chi, rng)`
- Check if `_PhysicalStepUpdate` is registered
- If registered: delegate to callback, enforce capacity constraint, normalize
- If not registered: fall back to Haar scrambler (default toy model)
- Clearly separated with comments: "Default: statistical toy model"

---

## New Files Created

### `example_physical_step.py`
- Three example callbacks (passthrough, Schwarzschild stub, JT/Schwarzian stub)
- Usage documentation
- Demonstrates the callback interface

### `test_physical_callback.py`
- Comprehensive test suite with 4 tests:
  1. Default toy model works
  2. Custom callback replaces toy model
  3. Capacity constraint enforcement
  4. End-to-end Page curve with callback
- All tests pass ✓

### `PHYSICS_HOOK_README.md`
- Complete documentation of the changes
- Usage guide with examples
- Implementation checklist for real microscopic models
- Technical notes on capacity constraints, normalization, randomness

---

## Verification

### ✓ Module loads correctly
```bash
python -c "import simulation; print('OK')"
```

### ✓ Hook function available
```bash
python -c "import simulation; print(hasattr(simulation, 'set_physical_step_update'))"
# Output: True
```

### ✓ All tests pass
```bash
python test_physical_callback.py
# All 4 tests pass
```

### ✓ Full simulation works
```bash
python simulation.py --outdir . --steps 6 --num-runs 10
# Successfully generates all datatables
```

---

## Key Features

### 1. Backward Compatibility
- Default behavior unchanged: Haar scrambler runs if no callback registered
- All existing scripts/workflows continue to work

### 2. Clean Separation
- Toy model clearly isolated in `else` branch of `run_tensor_step`
- Infrastructure (truncation, normalization, Page curves) independent of physics

### 3. Robust Callback Interface
- Automatic capacity constraint enforcement
- Automatic normalization
- Clear error messages for malformed callbacks

### 4. Extensibility
- Simple one-function interface to swap in real physics
- All downstream machinery (error budgets, convergence checks) automatically uses the new physics

---

## Usage Example

```python
import numpy as np
from simulation import set_physical_step_update

def my_physics(S_prev, target_chi, rng):
    """Your 4D gravity / QFT model here."""
    # TODO: Compute one-step evolution from influence functional
    # For now, a simple decay model:
    s = S_prev * 0.95
    if len(s) < target_chi:
        s = np.append(s, 0.1 * rng.rand())
    return s

# Register the callback
set_physical_step_update(my_physics)

# Run simulation - will use my_physics instead of Haar scrambler
# python simulation.py --outdir . --steps 12 --num-runs 50
```

---

## What This Is / Is Not

### ✓ What This IS:
- A **clean hook** to plug in microscopic dynamics
- A **separation** of toy model from infrastructure
- A **framework** for first-principles PT–MPO simulations

### ✗ What This Is NOT:
- A direct 4D GR+QFT solver (would require full research codebase)
- A complete implementation of any specific microscopic model
- A replacement for the influence functional / kernel derivations

---

## Next Steps for a Non-Toy Implementation

1. **Choose your model**: 4D SK horizon EFT, JT gravity, lattice QFT, etc.

2. **Derive the one-step map**: From eq. 4.14 / App. F or your model's action

3. **Compute entanglement spectrum**: Extract singular values after one step

4. **Implement callback**: `(S_prev, target_chi, rng) -> s_vals`

5. **Register and run**: `set_physical_step_update(callback)`

The PT–MPO infrastructure (truncation, Page curves, error budgets) will handle the rest.

---

## Diff Summary

```diff
simulation.py:
  Header (lines 1-34):      Rewrote to clarify toy vs. infrastructure
  Hook (lines 43-73):       Added set_physical_step_update(fn)
  run_tensor_step (168+):   Added callback check before Haar scrambler

New files:
  example_physical_step.py:     Example callbacks
  test_physical_callback.py:    Test suite (4 tests, all pass)
  PHYSICS_HOOK_README.md:       Complete documentation
  PHYSICS_HOOK_PATCH_SUMMARY.md: This file
```

---

## Testing Record

| Test | Status | Notes |
|------|--------|-------|
| Module import | ✓ Pass | No errors, hook function available |
| Default toy model | ✓ Pass | Haar scrambler works as before |
| Custom callback | ✓ Pass | Properly replaces toy model |
| Capacity constraint | ✓ Pass | Enforced correctly |
| Normalization | ✓ Pass | Always maintained |
| End-to-end simulation | ✓ Pass | Full datatables generated |

---

## Conclusion

The PT–MPO engine in `simulation.py` is now a **generic framework** that can consume microscopic dynamics from any model via the `set_physical_step_update` hook. The toy Haar scrambler is cleanly isolated and can be replaced without modifying the core infrastructure.

This is as far as you can push the separation in a clean way. Implementing actual 4D gravity / QFT dynamics is a separate research task, but now you have a well-defined interface to plug it in when ready.
