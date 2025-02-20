#!/usr/bin/env python3

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
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

print()
lr_results = pd.DataFrame(['Linear Regression', lr_train_mse, lr_train_r2, lr_test_mse, lr_test_r2]).transpose()
lr_results.columns = ['Method', 'Training MSE', 'Training R2', 'Test MSE', 'Test R2']
print(lr_results)

'''Random Forest'''
print('Random Forest')
# Training the model
rf = RandomForestRegressor(max_depth=2, random_state=100)
rf.fit(x_train, y_train)

# Applying the model to make predictions
y_rf_train_prediction = rf.predict(x_train)
y_rf_test_prediction = rf.predict(x_test)

# Evaluate model perfomance
rf_train_mse = mean_squared_error(y_train, y_rf_train_prediction)
rf_train_r2 = r2_score(y_train, y_rf_train_prediction)

rf_test_mse = mean_squared_error(y_test, y_rf_test_prediction)
rf_test_r2 = r2_score(y_test, y_rf_test_prediction)

print()
rf_results = pd.DataFrame(['Random Forest', rf_train_mse, rf_train_r2, rf_test_mse, rf_test_r2]).transpose()
rf_results.columns = ['Method', 'Training MSE', 'Training R2', 'Test MSE', 'Test R2']
print(rf_results)

'''Model Comparison'''
df_models = pd.concat([lr_results, rf_results], axis=0)
print(df_models.reset_index(drop=True))

'''Data Visualization'''
plt.scatter(x=y_train, y=y_lr_train_prediction)
print(plt.plot())
