# Importing Libraries
from qiskit import QuantumCircuit, Aer, execute
from qiskit import ClassicalRegister, QuantumRegister
from qiskit.visualization import plot_histogram
import matplotlib.pyplot as plt
from math import asin, sqrt
from math import log
import pandas as pd

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

# Defining the CCRY‐gate
def ccry(qc, theta, control1, control2, controlled):
    qc.cry(theta/2, control2, controlled)
    qc.cx(control1, control2)
    qc.cry(-theta/2, control2, controlled)
    qc.cx(control1, control2)
    qc.cry(theta/2, control1, controlled)

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
        Aer.get_backend('statevector_simulator') if measure is False else Aer.get_backend('qasm_simulator'),
        shots = shots
    ). result().get_counts()

    return plot_histogram(results, figsize=(12,4)) if hist else results

def qbn(Data, hist=True):
    def circuit(qc, qr=None, cr=None):
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

# Updated Eval_qbn function 
def eval_qbn(model, prepare_data, data):
    results = model(prepare_data(data), hist=False)
    return (
                round(log_liklihood(data,
                                results['11'], # prob_a_b
                                results['01'], # prob_na_b
                                results['10'], # prob_a_nb
                                results['00']  # prob_na_nb
                                ), 3),
                results['10'] / (results['10']+ results['00'])
            )

# Evaluating the guess
print(eval_qbn(qbn, lambda dataset: list(map(lambda item: item if item[1] is not None else (item[0], 0.5) ,dataset)), Data))

# Refining the model
print(eval_qbn(qbn, lambda dataset: list(map(lambda item: item if item[1] is not None else (item[0], 0.3) ,dataset)), Data))

# Further Refining the model
print(eval_qbn(qbn, lambda dataset: list(map(lambda item: item if item[1] is not None else (item[0], 0.252) ,dataset)), Data))

# Another iteration
print(eval_qbn(qbn, lambda dataset: list(map(lambda item: item if item[1] is not None else (item[0], 0.252) ,dataset)), Data))

train = pd.read_csv('train.csv')

# Maximum age of passenger we consider as a child
max_child_age = 8

# Probability of being a child
population_child = train[train.Age.le(max_child_age)]
p_child = len(population_child)/len(train)

# Probability of being female
population_female = train[train.Sex.eq("female")]
p_female = len(population_female)/len(train)

# Positions of the qubits
QPOS_ISCHILD = 0
QPOS_SEX = 1

def apply_ischild_sex(qc):
    # set marginal probability of IsChild
    qc.ry(prob_to_angle(p_child), QPOS_ISCHILD)

    # set marginal probability of sex
    qc.ry(prob_to_angle(p_female), QPOS_SEX)

# REPRESENTING THE NORM
# position of the qubit representing the norm
QPOS_NORM = 2

def apply_norm(qc, norm_params):
    """
    norm_params = {
        'p_norm_am': 0.25,
        'p_norm_af': 0.35,
        'p_norm_cm': 0.45,
        'p_norm_cf': 0.55
    }
    """
    # Set the conditional Probability of norm given adult/male
    qc.x(QPOS_ISCHILD)
    qc.x(QPOS_SEX)
    ccry(qc, prob_to_angle(
        norm_params['p_norm_am']
    ),QPOS_ISCHILD, QPOS_SEX, QPOS_NORM)
    qc.x(QPOS_ISCHILD)
    qc.x(QPOS_SEX)

    # Set the conditional probabilty of norm given adult/female
    qc.x(QPOS_ISCHILD)
    ccry(qc, prob_to_angle(
        norm_params['p_norm_af']
    ),QPOS_ISCHILD, QPOS_SEX, QPOS_NORM)
    qc.x(QPOS_ISCHILD)

    # set the conditional probability of Norm given child/male
    qc.x(QPOS_SEX)
    ccry(qc, prob_to_angle(
        norm_params['p_norm_cm']
    ),QPOS_ISCHILD, QPOS_SEX, QPOS_NORM)
    qc.x(QPOS_SEX)

    # set the conditional probability of Norm given child/female
    ccry(qc, prob_to_angle(
        norm_params['p_norm_cf']
    ),QPOS_ISCHILD, QPOS_SEX, QPOS_NORM)

# Calculating the Probabilities Related to the Ticket Class
pop_first = train[train.Pclass.eq(1)]
surv_first = round(len(pop_first[pop_first.Survived.eq(1)])/len(pop_first), 2)
p_first = round(len(pop_first)/len(train),2)

pop_second = train[train.Pclass.eq(2)]
surv_second = round(len(pop_second[pop_second.Survived.eq(1)])/len(pop_second), 2)
p_second = round(len(pop_second)/len(train),2)

pop_third = train[train.Pclass.eq(3)]
surv_third = round(len(pop_third[pop_third.Survived.eq(1)])/len(pop_third), 2)
p_third = round(len(pop_third)/len(train),2)

print("First class: {} of the passengers, survived: {}".format(p_first,surv_first))
print("Second class: {} of the passengers, survived: {}".format(p_second,surv_second))
print("Third class: {} of the passengers, survived: {}".format(p_third,surv_third))

# Representing the ticket‐class
# positions of the qubits
QPOS_FIRST = 3
QPOS_SECOND = 4
QPOS_THIRD = 5

def apply_class(qc):
    # set the marginal probability of pclass = 1st
    qc.ry(prob_to_angle(p_first), QPOS_FIRST)

    qc.x(QPOS_FIRST)
    # Set the marginal probability of pclass=2nd
    qc.cry(prob_to_angle(p_second/(1-p_first)), QPOS_FIRST, QPOS_SECOND)

    # Set the marginal probability of pclass=3rd
    qc.x(QPOS_SECOND)
    ccry(qc, prob_to_angle(p_third/(1-p_first-p_second)), QPOS_FIRST, QPOS_SECOND, QPOS_THIRD)
    qc.x(QPOS_SECOND)
    qc.x(QPOS_FIRST)

# Listing Represent survival
# position of the qubit
QPOS_SURV = 6

def apply_survival(qc, surv_params):    
    """
    surv_params = {
        'p_surv_f1': 0.3,
        'p_surv_f2': 0.4,
        'p_surv_f3': 0.5,
        'p_surv_u1': 0.6,
        'p_surv_u2': 0.7,
        'p_surv_u3': 0.8
    }
    """

    # set the conditional probability of Survival given unfavored by norm
    qc.x(QPOS_NORM)
    ccry(qc, prob_to_angle(
        surv_params['p_surv_u1']
    ),QPOS_NORM, QPOS_FIRST, QPOS_SURV)

    ccry(qc, prob_to_angle(
        surv_params['p_surv_u2']
    ),QPOS_NORM, QPOS_SECOND, QPOS_SURV)

    ccry(qc, prob_to_angle(
        surv_params['p_surv_u3']
    ),QPOS_NORM, QPOS_THIRD, QPOS_SURV)
    qc.x(QPOS_NORM)

    # set the conditional probability of Survival given favored by norm
    ccry(qc, prob_to_angle(
        surv_params['p_surv_f1']
    ),QPOS_NORM, QPOS_FIRST, QPOS_SURV)

    ccry(qc, prob_to_angle(
        surv_params['p_surv_f2']
    ),QPOS_NORM, QPOS_SECOND, QPOS_SURV)

    ccry(qc, prob_to_angle(
        surv_params['p_surv_f3']
    ),QPOS_NORM, QPOS_THIRD, QPOS_SURV)

# The quantum Bayesian network
QUBITS = 7

def qbn_titanic(norm_params, surv_params, hist=True, measure=False, shots = 1):
    def circuit(qc, qr= None, cr = None):
        apply_ischild_sex(qc)
        apply_norm(qc, norm_params)
        apply_class(qc)
        apply_survival(qc, surv_params)

    return as_pqc(QUBITS, circuit, hist= hist, measure=measure, shots=shots)

# Trying the QBN
norm_params = {
    'p_norm_am': 0.25,
    'p_norm_af': 0.35,
    'p_norm_cm': 0.45,
    'p_norm_cf': 0.55
}

surv_params = {
    'p_surv_f1': 0.3,
    'p_surv_f2': 0.4,
    'p_surv_f3': 0.5,
    'p_surv_u1': 0.6,
    'p_surv_u2': 0.7,
    'p_surv_u3': 0.8
}

qbn_titanic(norm_params, surv_params, hist=True)