from qskit import QuantumCircuit, Aer, execute
from qiskit.visulaisation import plot_histogram
import matplotlib.pyplot as plt

def run_circuit(qc, simulator='statevector_simulator', shots=1, hist=True):
    backend = Aer.get_backend(simulator)
    results = execute(qc, backend, shots=shots).result().get_counts()
    return plot_histogram(results, figsize=(18,4)) if hist else results

qc = QuantumCircuit(4)
qc.h([1,2,3,4])
run_circuit(qc)

# A single Hadamard Gate
qc.h(0)
run_circuit(qc)