# PT–MPO Physics Hook: Separating Toy Model from Infrastructure

## Summary

The PT–MPO engine in `simulation.py` has been refactored to cleanly separate the **statistical toy model** (Haar scrambler with capacity constraint) from the **tensor network infrastructure** (PT–MPO contraction, Page curve tracking, error budgets).

This allows you to plug in a **first-principles microscopic model** (4D gravity EFT, JT/Schwarzian, lattice QFT, etc.) without rewriting the entire simulation framework.

---

## What Changed

### 1. Updated Documentation (`simulation.py` header)

The module header now explicitly states:

> **NOTE ON PT-MPO:**
> * The default local dynamics are a **statistical toy model**: each step draws a Haar-random unitary on (memory ⊗ vacuum) and enforces the shrinking-capacity constraint implied by S_BH(u). This is meant to stress-test structural results (Page envelopes, finite-memory truncation, error budgets), not to directly solve 4D Einstein–Hilbert + QFT.
>
> * For a first-principles simulation of a specific microscopic model (4D gravity EFT, JT/Schwarzian, lattice QFT, …), register a callback via `set_physical_step_update`. That callback must compute the one-step singular values from your UV model; the PT–MPO machinery here then handles truncation, Page-curve construction, and convergence/error accounting.

### 2. New Hook: `set_physical_step_update(fn)`

A module-level function that registers a custom microscopic step-update rule:

```python
def set_physical_step_update(fn):
    """
    Register a microscopic step-update rule for the PT–MPO engine.

    The callback should implement a single HMC update step for your chosen
    4D gravity / QFT model:

        fn(S_prev: np.ndarray, target_chi: int, rng: np.random.RandomState)
            -> np.ndarray

    where
      * S_prev     : 1D array of Schmidt coefficients on the memory cut at step n−1
      * target_chi : capacity ceiling implied by S_BH(u_n) (P0)
      * rng        : NumPy RandomState for any stochastic subroutines

    The function must return a 1D array of (possibly unnormalized) singular values
    for the updated cut. This helper will enforce the capacity ceiling and normalize
    so that sum_i s_i^2 = 1.
    """
```

### 3. Modified `run_tensor_step(S_prev, target_chi, rng)`

The function now checks if a callback is registered:

* **If registered**: delegates the entire update to your microscopic model, then enforces capacity constraint and normalization.
* **If not registered**: falls back to the default Haar scrambler (toy model).

This isolates the "toy physics" in the `else` branch and makes it easy to swap in real UV dynamics.

---

## How to Use

### Option A: Default Toy Model (Unchanged Behavior)

If you don't register a callback, `simulation.py` behaves exactly as before—using the Haar scrambler:

```bash
python simulation.py --outdir . --steps 12 --num-runs 50
```

### Option B: Custom Microscopic Model

1. **Implement your callback** in a separate module (e.g., `my_physics.py`):

   ```python
   import numpy as np
   from simulation import set_physical_step_update

   def my_qft_step(S_prev, target_chi, rng):
       """
       Compute one-step evolution from your 4D gravity / QFT model.
       
       TODO: Replace this stub with actual physics:
         - Load or compute the influence functional kernel K(u-u')
           from your 4D matching (eq. 4.14, App. F).
         - Build the process tensor for this time step Δu.
         - Apply it to the current state (S_prev) to get the updated
           entanglement spectrum across the memory cut.
       """
       # Example placeholder: simple decay
       s = np.asarray(S_prev, dtype=float) * 0.95
       if len(s) < target_chi:
           s = np.append(s, 0.1 * rng.rand())
       return s

   # Register the callback
   set_physical_step_update(my_qft_step)
   ```

2. **Import your module before running the simulation**:

   ```python
   import my_physics  # This registers the callback
   import simulation

   # Now run the simulation as usual
   # All PT-MPO steps will use your callback instead of the toy model
   ```

   Or from the command line:

   ```bash
   python -c "import my_physics; import simulation; simulation.main()" --outdir . --steps 12 --num-runs 50
   ```

---

## Example Files

### `example_physical_step.py`

Demonstrates three example callbacks:

1. **`trivial_passthrough`**: No evolution (debugging).
2. **`schwarzschild_kernel_step`**: Stub for 4D Schwarzschild kernel (eq. 4.14).
3. **`jt_schwarzian_step`**: Stub for JT gravity / Schwarzian dynamics.

Run it for usage examples:

```bash
python example_physical_step.py
```

### `test_physical_callback.py`

Comprehensive test suite verifying:

* Default toy model works.
* Custom callbacks properly replace the toy model.
* Capacity constraints are enforced.
* Normalization is maintained.
* End-to-end Page curve generation with a callback.

Run the tests:

```bash
python test_physical_callback.py
```

---

## What This Does NOT Do

This patch **does not turn `simulation.py` into a direct 4D GR+QFT solver**. That would require:

* A full implementation of the Einstein equations or effective field theory.
* Quantum field theory on curved spacetime (e.g., lattice QFT, Schwarzian mode matching).
* Numerical integration of the influence functional / process tensor for your specific model.

What this patch **does** provide:

* A **clean separation** between "toy statistical model" and "PT–MPO bookkeeping."
* A **well-defined hook** so you (or a separate research code) can plug in the microscopic physics.
* All the downstream machinery (truncation, Page curves, error budgets, convergence checks) automatically works with your physical model.

---

## Implementation Checklist

To implement a non-toy microscopic model, you need to:

1. **Choose your model**: 4D SK horizon EFT, JT gravity, lattice QFT, etc.

2. **Compute the one-step map**: From the influence functional (eq. 4.14, App. F) or your model's action, derive how the state evolves over one time step Δu.

3. **Extract the entanglement spectrum**: After applying your one-step map, compute the singular values across the memory cut.

4. **Implement the callback**: Write a function with signature `(S_prev, target_chi, rng) -> s_vals` that does steps 2–3.

5. **Register and run**: Call `set_physical_step_update(your_callback)` before running `simulation.py`.

---

## Technical Notes

### Capacity Constraint (P0)

The PT–MPO engine enforces a capacity ceiling `target_chi = floor(e^{S_BH(u)})` derived from the semiclassical black hole entropy. Even if your callback returns more modes, the engine will truncate to `target_chi` and renormalize.

This ensures the simulation respects the area law / generalized second law throughout the evaporation.

### Normalization

Your callback should return singular values (possibly unnormalized). The engine will:

1. Truncate to `target_chi` if needed.
2. Normalize so `sum_i s_i^2 = 1`.

This makes it easier to implement callbacks—you don't have to worry about perfect normalization.

### Randomness

The `rng` parameter is a `np.random.RandomState` instance, seeded deterministically by the main simulation loop. Use it for any stochastic operations in your callback to maintain reproducibility.

---

## Testing

Run the included test suite to verify the mechanism works:

```bash
python test_physical_callback.py
```

Expected output:

```
============================================================
Testing Physical Step-Update Callback Mechanism
============================================================

Test 1: Default toy model (Haar scrambler)
  ✓ Default toy model works

Test 2: Custom callback (trivial passthrough)
  ✓ Custom callback works

Test 3: Capacity constraint enforcement
  ✓ Capacity constraint enforced

Test 4: End-to-end mini Page curve with callback
  ✓ End-to-end test passed

============================================================
All tests passed! ✓
============================================================
```

---

## Files Modified/Created

### Modified:
- **`simulation.py`**:
  - Updated header documentation (lines 1–34).
  - Added `set_physical_step_update(fn)` hook (lines 43–73).
  - Modified `run_tensor_step(...)` to check for registered callback (lines 168–199).

### Created:
- **`example_physical_step.py`**: Example callbacks and usage guide.
- **`test_physical_callback.py`**: Comprehensive test suite.
- **`PHYSICS_HOOK_README.md`**: This documentation.

---

## Next Steps

1. **Implement your microscopic model** using the influence functional / kernel from your chosen theory (4D SK, JT, lattice QFT, etc.).

2. **Register the callback** via `set_physical_step_update(your_callback)`.

3. **Run the simulation** as usual—all PT–MPO machinery (truncation, Page curves, error budgets) will use your first-principles dynamics.

4. **Compare results** to the toy model to understand how much the microscopic details affect the structural predictions (Page time, final entropy, convergence rates, etc.).

---

## References

For the microscopic physics (kernel, influence functional, etc.), see:

* **Eq. 4.14 and Appendix F** of the manuscript: Schwarzschild kernel Ξ_R(ω) and noise kernel N(ω) from 4D→2D matching.
* **Sec. 4**: JT gravity / Schwarzian effective action and boundary correlators.
* **Sec. 5**: PT–MPO formalism and structural results (independent of microscopic details).

The current toy model (Haar scrambler) is used to stress-test the structural PT–MPO machinery. Replacing it with first-principles physics is a straightforward extension via the hook documented here.
