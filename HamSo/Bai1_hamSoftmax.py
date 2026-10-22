import numpy as np


def softmax(scores):
    maxs = np.max(scores)
    
    exp_s = np.exp(scores - maxs)
    
    s = np.sum(exp_s)
    
    xs = exp_s / s
    
    return xs