import numpy as np


def sigmoid(x):
    return 1/(1+np.exp(-x))

def sigmoid_derivative(x):
    sm = sigmoid(x)
    return sm*(1-sm)