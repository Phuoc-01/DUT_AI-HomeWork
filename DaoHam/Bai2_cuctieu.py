import numpy as np

def f(x):
    return 3*x**4 - 4*x**2 - 6*x - 3

def central_derivative(func, x, epsilon=1e-5):
    f_p = func(x + epsilon / 2.0)
    f_m = func(x - epsilon / 2.0)
    
    # cong thuc dao ham trung tam
    return (f_p - f_m) / epsilon

def find_minimum(func, x, num_iteration, step):
    for _ in range(num_iteration):
        grad = central_derivative(func, x)
        
        # cap nhat x
        if grad > 0:
            x -= step
        elif grad < 0:
            x += step
            
    return x

# kiem tra chuong trinh
assert np.isclose(central_derivative(f, 0.0), -6.0, atol=1e-5)
assert np.isclose(find_minimum(f, 3.0, 1, 0.1), 2.9)
assert np.isclose(find_minimum(f, 3.0, 5, 0.1), 2.5)

for x0 in [-3.0, 0.0, 3.0]:
    x_min = find_minimum(f, x0, num_iteration=100, step=0.1)
    print(f"x start={x0:+.1f} | x last={x_min:.3f} | f(x)={f(x_min):.3f}")