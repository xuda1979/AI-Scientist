from utils.sim_runner import summarize_simulation_outputs

# Test the updated function
test_code = """import numpy as np
import matplotlib.pyplot as plt

def test_simulation():
    data = np.random.randn(100)
    print("Mean:", np.mean(data))
    return np.mean(data)

if __name__ == "__main__":
    result = test_simulation()
    print("Final result:", result)"""

test_outputs = {
    'stdout': 'Mean: 0.123\nFinal result: 0.123',
    'stderr': 'Warning: some deprecation warning'
}

summary = summarize_simulation_outputs(test_outputs, test_code)
print('=== SUMMARY INCLUDES BOTH CODE AND OUTPUTS ===')
print(summary[:800])
print('\n=== VERIFICATION ===')
print('CONTAINS CODE:', 'import numpy' in summary)
print('CONTAINS STDOUT:', 'Mean: 0.123' in summary)
print('CONTAINS STDERR:', 'Warning:' in summary)
print('LENGTH:', len(summary), 'characters')
