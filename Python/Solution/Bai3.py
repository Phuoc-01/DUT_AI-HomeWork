import math

def is_number ( n ):
    try:
        float(n)
    except ValueError:
        return False
    return True

def Tinh_activation_func():
    
    x_i = input("Nhap gia tri x: ")
    activation_name = input("nhap ten activation ").strip().lower()
    
    if not is_number(x):
        print('x must be number')
        return
    
    if activation_name not in ['binary', 'sigmoid', 'elu']:
        print(f"{activation_name} is not supported")
        return
    
    x = float(x_i)
    e = math.e
    alp = 0.01
    
    if activation_name == 'binary':
        if x < 0:
            fx = 0
        else:
            fx = 1
            
    elif activation_name == 'sigmoid':
        fx = 1 / (1 + e**(-x))
        
    else:
        if x < 0:
            fx = alp*(e**x - 1)
        else:
            fx = x
            
    print(f"{activation_name}: f({x:g})={fx:g}")
    
