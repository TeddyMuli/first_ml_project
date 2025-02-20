#!/usr/bin/env python3

import pandas as pd
'''Data Preparation'''
# Load Data
df = pd.read_csv('https://raw.githubusercontent.com/dataprofessor/data/refs/heads/master/delaney_solubility_with_descriptors.csv')

# Separate x and y
y = df['logS']
x = df.drop('logS', axis=1)
print(x)

# Data splitting
