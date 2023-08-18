import pandas as pd
from functools import reduce
from qiskit import QuantumCircuit,Aer, execute, ClassicalRegister, QuantumRegister
from math import asin, sqrt, ceil
from qiskit.visualization import plot_histogram
import matplotlib.pyplot as plt
train = pd.read_csv('train.csv')

# Size
cnt_all = len(train)

# List of all Survivors
survivors = train[train.Survived.eq(1)]
cnt_survivors = len(survivors)

# Calculate the prior prob
prob_survival = len(survivors)/cnt_all

print("The prior prob. of survived is: ", round(prob_survival,2))

# Get the modifier given the passenger's pclass
def get_modifier_pclass(pclass):
    # number of passengers with the same pclass
    cnt_survive_pclass = len(survivors[survivors.Pclass.eq(pclass)])

    # backward probability
    p_cl_surv = cnt_survive_pclass/cnt_survivors

    # Probability of the evidence 
    p_cl = len(train[train.Pclass.eq(pclass)])/cnt_all

    return p_cl_surv/p_cl

# get the modifier given the passenger's pclass
def get_modifier_sex(sex):
    # number of passengers with the same pclass 
    cnt_surv_sex = len(survivors[survivors.Sex.eq(sex)])

    # backward probability
    p_sex_surv = cnt_surv_sex/cnt_survivors

    # probability of the evidence
    p_sex = len(train[train.Sex.eq(sex)])/cnt_all

    return p_sex_surv/p_sex

def pre_process(passenger):
    '''
    passenger -- The pandas dataframe-row of the passenger
    returns a list of modifiers, like this [modifier a, modifier b...]
    '''
    return [
        get_modifier_pclass(passenger["Pclass"]),
        get_modifier_sex(passenger["Sex"]),
    ]
    
# a female passenger with 1st class ticket
print(pre_process(train.iloc[52]))

# a male passengerwith third class ticket 
print(pre_process(train.iloc[26]))

# Convert psi value to theta value
def prob_to_angle(prob):
    return 2*asin(sqrt(prob))

def pqc(backend, prior, modifier, shots=1, hist=False, measure = False):
    # prepare circuit with QUBIT and a classical bit to hold measurement
    qr = QuantumRegister(7)
    cr = ClassicalRegister(1)
    qc = QuantumCircuit(qr, cr) if measure else QuantumCircuit(qr)

    # insert quantum circuit
    qc = QuantumCircuit(4)

    # Measure qubit only if we want to measure 
    if measure:
        qc.measure(qr[0], cr[0])
    results = execute(qc,backend, shots=shots).result().get_counts()
    return plot_histogram(results, figsize=(12,4)) if hist else results

# caption the basic function
qc=QuantumCircuit(7)