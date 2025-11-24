import numpy as np
import matplotlib.pyplot as plt
import csv
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit.library import UnitaryGate
from qiskit.quantum_info import Statevector, partial_trace, entropy
from scipy.stats import unitary_group

class BlackHoleComb:
    """
    Implements the Horizon Memory Comb (HMC) simulation on a quantum circuit.
    Based on the 'Minimal Comb' architecture.
    
    Axioms implemented:
    1. Finite Memory (defined by n_memory)
    2. Fast Scrambling (Random Unitaries on M+I)
    3. Adiabatic Emission (Partial Swaps)
    """

    def __init__(self, n_memory, n_interior, n_steps, theta=np.pi/3):
        """
        Initialize the Black Hole Comb simulation.

        Args:
            n_memory (int): Number of qubits representing the horizon memory (M).
            n_interior (int): Number of qubits representing the interior (I).
            n_steps (int): Number of evaporation steps (time windows).
            theta (float): Partial swap angle. pi/2 is full swap (fast), small is adiabatic.
        """
        self.n_mem = n_memory
        self.n_int = n_interior
        self.n_steps = n_steps
        self.theta = theta
        
        # Registers
        self.qr_mem = QuantumRegister(n_memory, 'mem')
        self.qr_int = QuantumRegister(n_interior, 'int')
        # Radiation qubits are allocated for the entire history
        self.qr_rad = QuantumRegister(n_steps, 'rad')
        
        self.circuit = QuantumCircuit(self.qr_mem, self.qr_int, self.qr_rad)
        
        # Data storage for analysis
        self.entropies = []

    def _apply_scrambling(self):
        """
        Applies a random unitary to M + I registers.
        Implements Axiom 3: Fast Scrambling.
        """
        # Total qubits involved in scrambling
        total_qubits = self.n_mem + self.n_int
        dim = 2**total_qubits
        
        # Generate random unitary (Haar random proxy for 2-design)
        u_matrix = unitary_group.rvs(dim)
        u_gate = UnitaryGate(u_matrix, label='U_{scr}')
        
        # Apply to Memory and Interior
        qubits = list(self.qr_mem) + list(self.qr_int)
        self.circuit.append(u_gate, qubits)
        self.circuit.barrier()

    def _apply_emission(self, step_index):
        """
        Applies the emission isometry (Unitary Dilation P4).
        Uses a partial SWAP via Heisenberg coupling (RXX+RYY+RZZ).
        """
        # Target radiation qubit for this step
        rad_qubit = self.qr_rad[step_index]
        
        # Source memory qubit (The "Edge" mode)
        mem_qubit = self.qr_mem[0] 
        
        # Implementing partial swap via RXX, RYY, RZZ decomposition
        # This unitary couples the Horizon Memory to the Radiation field
        self.circuit.rxx(self.theta, mem_qubit, rad_qubit)
        self.circuit.ryy(self.theta, mem_qubit, rad_qubit)
        self.circuit.rzz(self.theta, mem_qubit, rad_qubit)
        self.circuit.barrier()

    def run_simulation(self):
        """
        Executes the time evolution step-by-step and calculates entropy.
        """
        # 1. Formation: Initial Scrambling to create BH state
        self._apply_scrambling() 
        
        print(f"Starting simulation: {self.n_mem} Mem, {self.n_int} Int, {self.n_steps} Steps, Theta={self.theta:.2f}")

        for step in range(self.n_steps):
            # A. Scramble the Black Hole (M + I)
            self._apply_scrambling()
            
            # B. Emit Radiation (Interaction M_edge -> R_step)
            self._apply_emission(step)
            
            # C. Calculate Entropy of Radiation R_{<=step}
            current_state = Statevector.from_instruction(self.circuit)
            
            # Indices to TRACE OUT: Memory + Interior + Future Radiation
            # Memory indices: 0 to n_mem-1
            # Interior indices: n_mem to n_mem + n_int - 1
            # Future Rad indices: n_mem + n_int + step + 1 to end
            
            # Construct list of indices to keep (The active Radiation)
            start_rad_index = self.n_mem + self.n_int
            active_rad_indices = list(range(start_rad_index, start_rad_index + step + 1))
            
            # Get reduced density matrix of just the active radiation
            rho_rad = partial_trace(current_state, [i for i in range(current_state.num_qubits) if i not in active_rad_indices])
            
            # Von Neumann Entropy (base 2)
            S_rad = entropy(rho_rad, base=2)
            self.entropies.append(S_rad)

    def plot_page_curve(self, filename='page_curve_simulation.png'):
        """
        Plots the accumulated radiation entropy vs time steps.
        """
        steps = np.arange(1, self.n_steps + 1)
        
        plt.figure(figsize=(10, 6))
        plt.plot(steps, self.entropies, 'o-', label='HMC Entropy $S(R)$', linewidth=2, color='blue')
        
        # Theoretical Page Time marker (Heuristic)
        # Page time ~ d_BH / 2. Here we approximate based on qubit count.
        page_time_heuristic = (self.n_mem + self.n_int) * 1.3 
        plt.axvline(x=page_time_heuristic, color='red', linestyle='--', label='Approx. Page Time')
        
        plt.xlabel('Evaporation Step (Time)')
        plt.ylabel('Entanglement Entropy $S(R)$')
        plt.title(f'Page Curve: Horizon Memory Comb (M={self.n_mem}, I={self.n_int})')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(filename)
        print(f"Plot saved to {filename}")

    def save_data(self, filename='entropy_data.csv'):
        """Saves entropy data to CSV for paper inclusion."""
        with open(filename, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Step', 'Entropy'])
            for i, s in enumerate(self.entropies):
                writer.writerow([i+1, s])
        print(f"Data saved to {filename}")

    def draw_circuit(self, filename='circuit_diagram.png'):
        """Draws the circuit to a file."""
        self.circuit.draw('mpl', filename=filename)
        print(f"Circuit diagram saved to {filename}")

if __name__ == "__main__":
    # Minimal Comb parameters from Appendix N
    # 2 Memory + 2 Interior = 4 Qubit BH.
    # 12 Steps ensures Radiation (12Q) > BH (4Q), forcing turnover.
    sim = BlackHoleComb(n_memory=2, n_interior=2, n_steps=12, theta=np.pi/3)
    
    sim.run_simulation()
    sim.plot_page_curve()
    sim.save_data()
    # sim.draw_circuit() # Requires matplotlib interaction or latex installation

    print("\n--- Simulation Results ---")
    for i, s in enumerate(sim.entropies):
        print(f"Step {i+1:02d}: {s:.4f}")