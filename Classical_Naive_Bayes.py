import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split

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

