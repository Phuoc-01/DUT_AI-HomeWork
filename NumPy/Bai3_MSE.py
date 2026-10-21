import numpy as np

def mse_loss(y_true, y_pred):
    s = (y_true - y_pred) ** 2
    loss = np.mean(s)
    
    return loss