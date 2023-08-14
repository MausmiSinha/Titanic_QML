from qiskit import QuantumCircuit, execute, Aer
from qiskit.visualization import plot_histogram
import matplotlib.pyplot as plt

# Create a quantum circuit with one qubit
qc = QuantumCircuit(1)

# Define Initial State as |1>
initial_state = [0,1]

# Apply initialization operation to qubit at position 0
print(qc.initialize(initial_state, 0))

dontshow=True

# Tell qiskit how to stimulate our circuit
backend = Aer.get_backend('statevector_simulator')

# Do the simulation, returning the result
result = execute(qc, backend).result()

# Lets get the probability distribution
counts = result.get_counts()

# Show the histogram
plot_histogram(counts)