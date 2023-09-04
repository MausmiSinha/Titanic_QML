import pandas as pd
from functools import reduce
from qiskit import QuantumCircuit,Aer, execute, ClassicalRegister, QuantumRegister
from math import asin, sqrt, ceil
from qiskit.visualization import plot_histogram
from sklearn.metrics import recall_score, precision_score, confusion_matrix
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

# Parametrized Quantum Circuit(PQC)
def pqc(backend, prior, modifier, shots=1, hist=False, measure = False):
    # prepare circuit with QUBIT and a classical bit to hold measurement
    qr = QuantumRegister(7)
    cr = ClassicalRegister(1)
    qc = QuantumCircuit(qr, cr) if measure else QuantumCircuit(qr)


    # The qubit positions
    trunks = 3
    aux = trunks + 1
    aux_half = trunks + 1
    aux_full = trunks + 2
    target = trunks + 3

    # Apply prior to qubit to the target qubit
    qc.ry(prob_to_angle(prior), target)

    # Work with the remainder
    qc.x(target)

    # Apply prior to the full auxilary qubit
    qc.cry(prob_to_angle(prior/(1-prior)), target, aux_full)

    # Work with the remainder
    qc.cx(aux_full, target)

    # Apply 0.5*prior to qubit 1
    qc.cry(prob_to_angle(0.5*prior/(1-(2*prior))), target, aux_half)

    # Rearrange states to seperate qubits
    qc.x(target)
    qc.cx(aux_full, target)

    sorted_modifiers = sorted(modifier)

    # Calculating the posterior probability for a modifier smaller than 1.0
    for step in range(0, len(modifier)):
        if sorted_modifiers[step] > 1:
            qc.cry(prob_to_angle(min(1, sorted_modifiers[step]-1)), aux_full, target)
            
            # Seperate the aux_full and the target qubit 
            qc.ccx(target, aux_full, 0)
            qc.ccx(target, 0, aux_full)

            if step == 0:
                # Equalize what we transferred to the target (*2) and increase the aux_full to reflect the modifier (*2)
                qc.cry(prob_to_angle(min(1, (sorted_modifiers[step]-1)*2*2)), aux_half, aux_full)
        
        else:
            # apply the modifier to the target qubit
            qc.cry(prob_to_angle(1-sorted_modifiers[step]), target, step*2)
            qc.cx(step*2, target)

            if step == 0:
                # apply modifier to full auxilary qubit
                qc.cry(prob_to_angle(1-sorted_modifiers[step]), aux_full, step*2+1)

                # Unentangle the full auxilary from trunk
                qc.cx(step*2+1, aux_full)

    
    # Measure qubit only if we want to measure 
    if measure:
        qc.measure(qr[0], cr[0])
    results = execute(qc,backend, shots=shots).result().get_counts()
    return plot_histogram(results, figsize=(12,4)) if hist else results

# Post processig function
def post_process(counts):
    """
    counts -- the result of the quantum circuit execute
    returns the prediction
    """
    return int(list(map(lambda item: item[0], counts.items()))[0])

# Defining the run function
def run(f_classify, data):
    return [f_classify(data.iloc[i]) for i in range(0, len(data))]

# Specifing re-usable backend
backend = Aer.get_backend('qasm_simulator')

# Specificity
def specificity(matrix):
    return matrix[0][0]/(matrix[0][0]+ matrix[0][1]) if (matrix[0][0]+matrix[0][1]>0) else 0

# Negative Predictive Value(NPV)
def npv(matrix):
    return matrix[0][0]/(matrix[0][1]+ matrix[1][0]) if (matrix[0][1]+ matrix[1][0] > 0) else 0

# Creating Classifier Report function
def classifier_report(name, run, classify, input, labels):
    print(name)
    cr_prediction = run(classify, input)
    cr_cm = confusion_matrix(labels, cr_prediction)

    cr_precision = precision_score(labels, cr_prediction)
    cr_recall = recall_score(labels, cr_prediction)
    cr_specificity = specificity(cr_cm)
    cr_npv = npv(cr_cm)
    cr_level = 0.25 * (cr_precision + cr_recall + cr_specificity + cr_npv)

    print('The precision score of the {} classifier is {:.2f}'
        .format(name, cr_precision))
    print('The recall score of the {} classifier is {:.2f}'
        .format(name, cr_recall))
    print('The specificity score of the {} classifier is {:.2f}'
        .format(name, cr_specificity))
    print('The npv score of the {} classifier is {:.2f}'
        .format(name, cr_npv))
    print('The information level is: {:.2f}'
        .format(cr_level))
    
classifier_report("Quantum Naive Bayes Classifier",
                  run,
                  lambda passenger: post_process(pqc(backend, prob_survival, pre_process(passenger), measure=True, hist=False)),
                  train,
                  train['Survived']
                  )