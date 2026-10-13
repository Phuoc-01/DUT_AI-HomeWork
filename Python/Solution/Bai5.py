import math     


def pascal_algo(level):
    if level <= 0:
        print("Level must be bigger than 0")
        return
    
    tri = []
    
    for i in range(level):
        row = [1] * (i+1)
        
        for j in range(1,i):
            row[j] = tri[i-1][j-1] + tri[i-1][j]
            
        tri.append(row)
        
        print(" ".join(str(num) for num in row))
        
def fibo_algo(len):
    if len <= 0:
        print("length must be bigger than 0")
        return
    
    fib = []
    
    for i in range(len):
        if i == 0:
            fib.append(0)
        elif i == 1:
            fib.append(1)
        else:
            next = fib[i-1] + fib[i-2]
            fib.apend(next)
            
    print(", ".join(str(num) for num in fib))
    

