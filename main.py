#!/usr/bin/env python3

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

'''Data Preparation'''
# Load Data
df = pd.read_csv('https://raw.githubusercontent.com/dataprofessor/data/refs/heads/master/delaney_solubility_with_descriptors.csv')

# Separate x and y
y = df['logS']
x = df.drop('logS', axis=1)
print(f'X: \n{x}')

# Data splitting
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=100)

print(f'X Train: \n{x_train}')
print(f'X Test: \n{x_test}')

'''Model building'''
# Training the model
lr = LinearRegression()
lr.fit(x_train, y_train)

# Applying the model to make predictions
y_lr_train_prediction = lr.predict(x_train)
y_lr_test_prediction = lr.predict(x_test)
print(y_lr_train_prediction, y_lr_test_prediction)

# Evaluate model perfomance
