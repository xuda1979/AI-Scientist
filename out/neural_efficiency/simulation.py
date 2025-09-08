
import numpy as np
import matplotlib.pyplot as plt

# Default simulation for the paper
def main():
    print("Running simulation...")
    
    # Generate some sample data
    x = np.linspace(0, 10, 100)
    y = np.sin(x) + 0.1 * np.random.randn(100)
    
    # Create a simple plot
    plt.figure(figsize=(8, 6))
    plt.plot(x, y, 'b-', label='Data')
    plt.xlabel('X values')
    plt.ylabel('Y values')
    plt.title('Sample Results')
    plt.legend()
    plt.grid(True)
    plt.savefig('results_plot.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Save results
    with open('results.txt', 'w') as f:
        f.write(f"Sample Results\n")
        f.write(f"Mean: {np.mean(y):.3f}\n")
        f.write(f"Std: {np.std(y):.3f}\n")
        f.write(f"Min: {np.min(y):.3f}\n")
        f.write(f"Max: {np.max(y):.3f}\n")
    
    print("Simulation completed. Results saved to results.txt")

if __name__ == "__main__":
    main()
