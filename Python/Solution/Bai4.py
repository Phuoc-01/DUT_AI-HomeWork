import math
import random

def evaluate_regression_loss():
    num_s_in = input("Nhap so luong samples: ")
    
    if not num_s_in.isnumeric():
        print("number of samples must be an integer number")
        return
    
    num_s = int(num_s_in)
    
    lossname = input("nhap loss name: ").strip().upper()
    
    for i in range(num_s):
        t = random.uniform(1,10)
        p = random.uniform(1,10)
        
        if lossname == 'MAE':
            loss = abs(t - p)
        elif lossname == 'MSE':
            loss = (t - p)**2
        elif lossname == 'RMSE':
            loss = math.sqrt((t - p)**2)
        
        print(f"sample-{i}, pred: {p:.4f}, target: {t:.4f}, loss: {loss:.4f}")
        
        
