import numpy as np

def cosine_similarity(A, B):
    dp = np.sum(a*b,axis = 1)
    nA = np.sqrt(np.sum(A**2,axis = 1))
    nB = np.sqrt(np.sum(B**2,axis = 1))
    
    cs = dp/(nA + nB + 1e-8)
    
    return cs