from qiskit import QuantumCircuit, execute, Aer
from qiskit.visualization import plot_histogram
import matplotlib.pyplot as plt
from math import sqrt,pi, cos, sin
from sklearn.metrics import recall_score, precision_score, confusion_matrix
import numpy as np

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

# Define state psi
initial_state = [1/sqrt(2), 1/sqrt(2)]

# Redine the quantum circuit 
qc = QuantumCircuit(1)

# Initialize the 0th qubit in the state 'initial_state'
qc.initialize(initial_state,0)

# Execute the qc 
result = execute(qc, backend).result().get_counts()

# Plot the result
plot_histogram(result)

# The qubit with a probability of 0.25 to result in zero 
initial_state = [1/2, sqrt(3)/2]
qc = QuantumCircuit(1)
qc.initialize(initial_state, 0)
result = execute(qc, backend).result().get_counts()
plot_histogram(result)

# Using theta to define quantum state vector 
def get_state(theta):
    return [ cos(theta/2), sin(theta/2)]

# Defining value of theta for getting the initial state 
theta = -pi/2

qc = QuantumCircuit(1)
qc.initialize(get_state(theta), 0)
backend = Aer.get_backend('statevector_simulator')
result = execute(qc, backend).result().get_counts()

plot_histogram(counts)

# Quantum Circuit provides the draw function that renders an image of the circuit 
# diagram. We provide output = text as a named parameter to get an ASCII art version
# of image 
print(qc.draw(output='text'))

# Building a simple Parametrized Quantum Circuit(PQC) Classifier
def pqc_classify(backend, passenger_state):
    '''backend -- a qiskit backend to run the quantum circuit at 
       passenger_state -- a valid quantum state vector'''
    
    qc = QuantumCircuit(1)
    qc.initialize(passenger_state, 0)
    # Measure the qubit
    qc.measure_all()

    # run the quantum circuit and get the counts, these are either {'0':1} or {'1':1}
    result = execute(qc, backend).result().get_counts(qc)

    # get the bit 0 or 1
    return int(list(map(lambda item: item[0], counts.items()))[0])

with open('train.npy', 'rb') as f:
    train_input = np.load

