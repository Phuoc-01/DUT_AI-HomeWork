import numpy as np

def stable_softmax(logits):
    m = np.max(logits, axis = 1, keepdims = True)
    
    tu = np.exp(logits - m)
    
    mau = np.sum(tu, axis = 1, keepdims = True)
    
    soft = tu / mau
    
    return soft