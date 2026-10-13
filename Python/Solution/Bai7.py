import math

def reverse(n):
    res = 0
    while n > 0:
        carry = n%10
        res = res*10 + carry
        n//=10
    return res