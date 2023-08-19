from qiskit import QuantumCircuit, Aer, execute, ClassicalRegister, QuantumRegister
from qiskit.visualization import plot_histogram
import matplotlib.pyplot as plt 
from math import asin, sqrt
from math import pi

import pandas as pd
train = pd.read_csv('train.csv')

# Convert psi value to theta value
def prob_to_angle(prob):
    return 2*asin(sqrt(prob))

theta = pi/2

def ccnot(qc):
    # Apply the first half of the rotatione
    qc.cry(theta, 1,2)

    # This sequence has no effect if both control qubits
    # are in state |1>
    qc.cx(0,1)
    qc.cry(-theta,1,2)
    qc.cx(0,1)

    # Apply the second half of the rotation
    qc.cry(theta, 0,2)

    # execute the qc
    return execute(qc,Aer.get_backend('statevector_simulator')).result().get_counts()

max_child_age = 8

# Prob. of being a child
population_child = train[train.Age.le(max_child_age)]
p_child = len(population_child)/len(train)

# Prob of being female
population_female = train[train.Sex.eq('female')]
p_female = len(population_female)/len(train)

# Initialize the quantum circuit
qc = QuantumCircuit(3)

# Set qubit 0 to p_child
qc.ry(prob_to_angle(p_child),0)

# Set qubit 1 to p_female
qc.ry(prob_to_angle(p_female),1)

# Defining the CCRY‐gate
def ccry(qc, theta, control1, control2, controlled):
    qc.cry(theta/2, control2, controlled)
    qc.cx(control1, control2)
    qc.cry(-theta/2, control2, controlled)
    qc.cx(control1, control2)
    qc.cry(theta/2, control1, controlled)

# Female children
population_female = train[train.Sex.eq("female")]
population_f_c = population_female[population_female.Age.le(max_child_age)]
surv_f_c = population_f_c[population_f_c.Survived.eq(1)]
p_surv_f_c = len(surv_f_c)/len(population_f_c)

# female adults
population_f_a = population_female[population_female.Age.gt(max_child_age)]
surv_f_a = population_f_a[population_f_a.Survived.eq(1)]
p_surv_f_a = len(surv_f_a)/len(population_f_a)

# Male children
population_male = train[train.Sex.eq("male")]
population_m_c = population_male[population_male.Age.le(max_child_age)]
surv_m_c = population_m_c[population_m_c.Survived.eq(1)]
p_surv_m_c = len(surv_m_c)/len(population_m_c)

# male adults
population_m_a = population_male[population_male.Age.gt(max_child_age)]
surv_m_a = population_m_a[population_m_a.Survived.eq(1)]
p_surv_m_a = len(surv_m_a)/len(population_m_a)

# Quantum circuit with classical register
qr = QuantumRegister(3)
cr = ClassicalRegister(1)
qc = QuantumCircuit(qr, cr)

# Set state |00> to conditional probability of male adults
qc.x(0)
qc.x(1)
ccry(qc, prob_to_angle(p_surv_m_a),0,1,2)
qc.x(0)
qc.x(1)

# Set state |01> to conditional prob. of male children
qc.x(0)
ccry(qc, prob_to_angle(p_surv_m_c),0 ,1 ,2)
qc.x(0)

# Set state |10> to conditional prob. of female adult
qc.x(1)
ccry(qc, prob_to_angle(p_surv_f_a),0,1,2)
qc.x(1)

# Set state |11> to conditional prob. of female children
ccry(qc, prob_to_angle(p_surv_f_c),0,1,2)

qc.measure(qr[2], cr[0])
# execute the qc
results = execute(qc,Aer.get_backend('qasm_simulator'), shots=1000).result().get_counts()
plot_histogram(results)

