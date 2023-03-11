import time

import pandas as pd
import numpy as np

from models import LogisticRegressionModel, SVMModel

# Read the challenge data
challenges = pd.read_csv('./data/APUF_XOR_Challenge_Parity_64_500000.csv', header=None)
challenges = challenges.iloc[:, :65] #grab the 64 bits of the challenge

# Read the response data
xor_2 = pd.read_csv('./data/2-xorpuf.csv', header=None)

# Select NUM_POINTS points
NUM_POINTS = 20000
x = challenges.iloc[:NUM_POINTS].to_numpy()
y = np.ndarray.flatten(xor_2.iloc[:NUM_POINTS].to_numpy())


def train_and_test_model(model_type: str, x, y):
    if model_type.upper() == 'LR':
        model = LogisticRegressionModel()
    elif model_type.upper() == 'SVM':
        model = SVMModel()
    
    model.train(x, y)
    model.score(x, y)

train_and_test_model('LR', x, y)
train_and_test_model('SVM', x, y)