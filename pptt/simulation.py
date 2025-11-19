import numpy as np
import matplotlib.pyplot as plt

def simulate_experiment():
    np.random.seed(42)
    data = np.random.normal(loc=0, scale=1, size=1000)
    return data

def analyze_data(data):
    mean = np.mean(data)
    std_dev = np.std(data)
    return mean, std_dev

def plot_results(data, mean, std_dev):
    plt.figure(figsize=(10, 6))
    plt.hist(data, bins=30, alpha=0.7, label='Data')
    plt.axvline(mean, color='r', linestyle='dashed', linewidth=2, label=f'Mean: {mean:.2f}')
    plt.axvline(mean + std_dev, color='g', linestyle='dashed', linewidth=2, label=f'Std Dev: {std_dev:.2f}')
    plt.axvline(mean - std_dev, color='g', linestyle='dashed', linewidth=2)
    plt.title('Simulation Results')
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.legend()
    plt.show()

if __name__ == '__main__':
    data = simulate_experiment()
    mean, std_dev = analyze_data(data)
    plot_results(data, mean, std_dev)