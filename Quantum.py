from qiskit import QuantumCircuit, execute, Aer
from qiskit.visualization import plot_histogram
import matplotlib.pyplot as plt
from math import sqrt,pi, cos, sin
from sklearn.metrics import recall_score, precision_score, confusion_matrix
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split

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

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

# Removing missing value records in Embarked Attribute
train = train.dropna(subset=["Embarked"])
test = test.dropna(subset=["Embarked"])
test = test.dropna(subset=["Fare"])

# Cabin attribute does not gives any signifacant insights so drop it
train = train.drop("Cabin", axis = 1)
test = test.drop("Cabin", axis = 1)

# Now since age is important for predicting the survival we we not remove the null values
# instead we will fill the null values of age by mean age
mean = train['Age'].mean()
mean1 = test['Age'].mean()
train['Age'] = train['Age'].fillna(mean)
test['Age'] = test['Age'].fillna(mean1)

# Now we will drop the identifiers since we have no redundant records
train = train.drop("PassengerId", axis=1)
train = train.drop("Name", axis=1)
train = train.drop("Ticket", axis=1)

# Encoding Textual and Categorical Data
le = LabelEncoder()

for col in ['Sex', 'Embarked']:
    le.fit( train[col])
    train[col] = le.transform(train[col])

# Feature Scaling
'''
A common way to cope with data of different scales is min-max-scaling, 
which is also known as normalization. This process shifts and rescales 
values so that they end up ranging from 0 to 1. It subtracts the minimum 
value from each value and divides it by the maximum minus the minimum value.

The scaler returns a NumPy-array instead of a Pandas DataFrame.
'''

scaler = MinMaxScaler()
scaler.fit(train)
train = scaler.transform(train)

# Splitting between train and test
input_data = train[:, 1:8]
labels = train[:, 0]

train_input, test_input, train_labels, test_labels = train_test_split(input_data, labels, test_size=0.2)

#Classification Runner
def run(f_classify, x):
    return list(map(f_classify, x))

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
    cr_cm = confusion_matrix(labels, input)

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
    
# Specify the quantum state that results in either 0 or 1
initial_state = [1/sqrt(2), 1/sqrt(2)] 

classifier_report("Random PQC",
                  run,
                  lambda passenger: pqc_classify(backend, initial_state),
                  train_input,
                  train_labels)
