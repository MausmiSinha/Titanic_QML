# Importing Libraries
from qiskit import QuantumCircuit, Aer, execute
from qiskit import ClassicalRegister, QuantumRegister
from qiskit.visualization import plot_histogram
import matplotlib.pyplot as plt
from math import asin, sqrt
from math import log

# Dataset with missing value
Data = [(1, 1), (1, 1), (0, 0), (0, 0), (0, 0), (0, None), (0, 1), (1, 0)]

def log_liklihood(Data, prob_a_b, prob_a_nb, prob_na_b, prob_na_nb):
    def get_prob(point):
        if point[0] == 1 and point[1] == 1:
            return log(prob_a_b)
        elif point[0] == 1 and point[1] == 0:
            return log(prob_a_nb)
        elif point[0] == 0 and point[1] == 1:
            return log(prob_na_b)
        elif point[0] == 0 and point[1] == 0:
            return log(prob_na_nb)
        else:
            return log(prob_na_b+prob_na_nb)
        
    return sum(map(get_prob, Data))

def prob_to_angle(prob):
    return 2*asin(sqrt(prob))

def as_pqc(cnt_quantum, with_qc, cnt_classical=1, shots=1, hist=False, measure=False):
    # Prepare the circuit with qubits and a classical bit to hold measurement
    qr = QuantumRegister(cnt_quantum)
    cr = ClassicalRegister(cnt_classical)
    qc = QuantumCircuit(qr, cr) if measure else QuantumCircuit(qr)

    with_qc(qc, qr=qr, cr=cr)

    results = execute(
        qc,
        Aer.get_backend('statevetor') if measure is False else Aer.get_backend('qasm_simulator')
        shots = shots
    ). result().get_counts()

    return plot_histogram(results, figsize=(12,4)) if hist else results

def qbn(Data, hist=True):
    def circuit(qc, qr=None, qc=None):
        list_a = list(filter(lambda item: item[0] == 1, Data))
        list_na = list(filter(lambda item: item[0] == 0, Data))

        # set the marginal probability of A
        qc.ry(prob_to_angle(
            len(list_a)/len(Data)
        ),0)

        # set the conditional probability of not A and (B/not B)
        qc.x(0)
        qc.cry(prob_to_angle(sum(list(map(lambda item: item[1], list_na)))/ len(list_na)),0,1)
        qc.x(0)

        # set the conditional probability of A and (B/not B)
        qc.cry(prob_to_angle(sum(list(map(lambda item: item[1], list_a)))/ len(list_a)),0,1)

    return as_pqc(2, circuit, hist=hist)

qbn(list(filter(lambda item: item[1] is not None, Data)))
