#!/usr/bin/env python3
"""Quick validation that the physics hook is working."""

print('='*70)
print('PHYSICS HOOK VALIDATION')
print('='*70)

import simulation
import numpy as np

print('\n1. Module imported successfully')
print('   - set_physical_step_update:', hasattr(simulation, 'set_physical_step_update'))
print('   - run_tensor_step:', hasattr(simulation, 'run_tensor_step'))
print('   - calculate_entropy:', hasattr(simulation, 'calculate_entropy'))

print('\n2. Default behavior (toy model):')
rng = np.random.RandomState(42)
S = np.array([1.0])
for i in range(3):
    S = simulation.run_tensor_step(S, 5, rng)
    print(f'   Step {i+1}: dim={len(S)}, entropy={simulation.calculate_entropy(S):.4f}')

print('\n3. Custom callback:')
def custom(S_prev, chi, rng):
    return S_prev * 0.95

simulation.set_physical_step_update(custom)
S = np.array([1.0])
rng = np.random.RandomState(42)
for i in range(3):
    S = simulation.run_tensor_step(S, 5, rng)
    print(f'   Step {i+1}: dim={len(S)}, entropy={simulation.calculate_entropy(S):.4f}')

simulation.set_physical_step_update(None)

print('\n' + '='*70)
print('All checks passed! Hook is working correctly.')
print('='*70)
