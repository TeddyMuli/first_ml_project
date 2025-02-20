#!/usr/bin/env python3

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

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
'''Linear Regression'''
print('Linear Regression')
# Training the model
lr = LinearRegression()
lr.fit(x_train, y_train)

# Applying the model to make predictions
y_lr_train_prediction = lr.predict(x_train)
y_lr_test_prediction = lr.predict(x_test)
#print(y_lr_train_prediction, y_lr_test_prediction)

# Evaluate model perfomance
lr_train_mse = mean_squared_error(y_train, y_lr_train_prediction)
lr_train_r2 = r2_score(y_train, y_lr_train_prediction)

lr_test_mse = mean_squared_error(y_test, y_lr_test_prediction)
lr_test_r2 = r2_score(y_test, y_lr_test_prediction)

print('LR MSE(TRAIN)', lr_train_mse)
print('LR R2(TRAIN)', lr_train_r2)
print('LR MSE(TEST)', lr_test_mse)
print('LR R2(TEST)', lr_test_r2)
print()
lr_results = pd.DataFrame(['Linear Regression', lr_train_mse, lr_train_r2, lr_test_mse, lr_test_r2]).transpose()
lr_results.columns = ['Method', 'Training MSE', 'Training R2', 'Test MSE', 'Test R2']
print(lr_results)
