import math

def GiaiThua(n):
    if n == 0 or n == 1:
        return 1
    res = 1
    for i in range(2,n+1):
        res*=i
    return res

def sin_x(x,n):
    res = 0
    for i in range(n+1):
        temp = ((-1)**i) * (x**(2*i+1))/ GiaiThua(2*i+1)
        res += temp
    return res

def cos_x(x,n):
    res = 0
    for i in range(n+1):
        temp = ((-1)**i) * (x**(2*i))/ GiaiThua(2*i)
        res += temp
    return res

def sinh_x(x,n):
    res = 0
    for i in range(n+1):
        temp = (x**(2*i+1))/ GiaiThua(2*i+1)
        res += temp
    return res

def cosh_x(x,n):
    res = 0
    for i in range(n+1):
        temp = (x**(2*i))/ GiaiThua(2*i)
        res += temp
    return res

