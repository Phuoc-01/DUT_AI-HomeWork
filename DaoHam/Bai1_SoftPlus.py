import numpy as np
import matplotlib.pyplot as plt


def softplus(x):
    # dung logaddexp xu ly tran so
    return np.logaddexp(0,x)

def softplus_derivative(x):
    x = np.asarray(x, dtype = float) # xu ly don va mang
    #xu ly ket qua re nhanh
    return np.where (x >= 0, 1.0 / (1.0 + np.exp(-x)), np.exp(x) / (1.0 + np.exp(x)))

x = np.linspace(-6, 6, 401)
y = softplus(x)
dy = softplus_derivative(x)

# kich thuoc
assert np.shape(y) == np.shape(x) and np.shape(dy) == np.shape(x)
#xet tai x = 0
assert np.isclose(softplus(0.0), np.log(2), atol=1e-8)
assert np.isclose(softplus_derivative(0.0), 0.5, atol=1e-8)
# dao ham trong khoang (0,1)
assert np.all((dy > 0) & (dy < 1))

# ve do thi
plt.plot(x, y, label="Softplus f(x)")
plt.plot(x, dy, label="Đạo hàm f'(x)")
plt.xlabel("x")
plt.grid(alpha=0.3)
plt.legend()
plt.show()