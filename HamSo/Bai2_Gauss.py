import numpy as np
import matplotlib.pyplot as plt

def gaussian(x,mean,variance):
    
    #dlc =  can 2 cua phuong sai
    c = 1.0 / np.sqrt(2 * np.pi * variance)
    # mu
    e = -((x-mean)**2)/ (2*variance)
    
    return c*e

def log_gaussian(x, mean, variance):
    # so hang 1
    t1 = -0.5 * np.log(2 * np.pi * variance)
    # so hang 2
    t2 = -((x - mean)**2) / (2 * variance)
    
    return t1 + t2