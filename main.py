# 30 July, 2023
import pandas as pd
import numpy as np
import random 
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, precision_score, recall_score

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

# print('Number of Row {} and Number of Columns {} in Train dataset'.format(*train.shape))
# print('Number of Row {} and Number of Columns {} in Test dataset'.format(*test.shape))

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

# Removing Redundance, the passengerId in the data should be unique for each record
# If we have duplicate entry having same passengerId it means we have redundancy
print("There are {} unique Passengers in the data".format(train['PassengerId'].nunique()))
print("There are {} unique Names in the data".format(train['Name'].nunique()))
print("There are {} unique Ticket Number in the data".format(train['Ticket'].nunique()))

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

print("The minimum value is {} and the maximum value is {}".format(train.min(), train.max()))

# Splitting between train and test
input_data = train[:, 1:8]
labels = train[:, 0]

train_input, test_input, train_labels, test_labels = train_test_split(input_data, labels, test_size=0.2)

print("There are {} training records and {} testing records".format(train_input.shape[0], test_input.shape[0]))
print("There are {} input columns".format(train_input.shape[1]))

# using random classifier
random.seed(a=None, version=2)

def classify(passenger):
    return random.randint(0,1)

#Classification Runner
def run(f_classify, x):
    return list(map(f_classify, x))

#Run the classifier
random_predictions = run(classify, train_input)
random_cm = confusion_matrix(train_labels, random_predictions)

# Creating a evaluator function
def evaluate(prediction, actual):
    correct = list(filter(
        lambda item: item[0] == item[1],
        list(zip(prediction, actual)) 
    ))
    return '{} correct predictions out of {} predictions. Accuracy {:.0f}%' \
    .format(len(correct), len(actual), 100*len(correct)/len(actual))

# print(evaluate(result, train_labels))
# print(evaluate(run(classify, train_input), train_labels))

# Always predict a passenger died
def predict_death(item):
    return 0

# Creating a confusion matrix for our predictions
pred = run(predict_death, train_input)
print(confusion_matrix(train_labels, pred))

# Precision Score
print("The precision score of the predict_death classifier is {}".format(precision_score(train_labels, pred, zero_division=0)))

# Recall Score
print("The recall score of predict_death classifier is {}".format(recall_score(train_labels, pred)))

# Specificity
def specificity(matrix):
    return matrix[0][0]/(matrix[0][0]+ matrix[0][1]) if (matrix[0][0]+matrix[0][1]>0) else 0

cm = confusion_matrix(train_labels, pred)
print("The specificity score of predict_death classifier is {:.2f}".format(specificity(cm)))

# Negative Predictive Value(NPV)
def npv(matrix):
    return matrix[0][0]/(matrix[0][1]+ matrix[1][0]) if (matrix[0][1]+ matrix[1][0] > 0) else 0

print("The NPV Score of predict_death classifier is {:.2f}".format(npv(cm)))

# Now Evaluating Random Classifier
print("\nThe precision score of the random classifier is {:.2f}".format(precision_score(train_labels, random_predictions)))
print("The recall score of random classifier is {:.2f}".format(recall_score(train_labels, random_predictions)))
print("The specificity score of the random classifier is {:.2f}".format(specificity(random_cm)))
print("The npv score of the random classifier is {:.2f}".format(npv(random_cm)))

# Defining Hypocrite Classifier
def hypocrite(passenger, weight):
    return round(min(1, max(0,weight*0.5+random.uniform(0,1))))

# Creating Hypocrite Classifier
w_predictions = run(lambda passenger: hypocrite(passenger, -0.5), train_input)
w_cm = confusion_matrix(train_labels, w_predictions)

# Now Evaluating hypocrite Classifier
print("\nThe precision score of the hypocrite classifier is {:.2f}".format(precision_score(train_labels, w_predictions)))
print("The recall score of hypocrite classifier is {:.2f}".format(recall_score(train_labels, w_predictions)))
print("The specificity score of the hypocrite classifier is {:.2f}".format(specificity(w_cm)))
print("The npv score of the hypocrite classifier is {:.2f}".format(npv(w_cm)))

# Reusable Function to unmask hypocrite classifier
