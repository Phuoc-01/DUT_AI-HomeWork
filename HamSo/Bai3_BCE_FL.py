import numpy as np

def binary_crossentropy(y, y_hat):
    y = np.asarray(y, dtype=float)
    y_hat = np.asarray(y_hat, dtype=float)
    if y.shape != y_hat.shape:
        raise ValueError("y và y_hat must same Length")
    eps = 1e-12
    # dung np.clip() va np.log() tra ve mang loss co kich thuoc = y
    y_hat_c = np.clip(y_hat, eps, 1.0 - eps)
    
    loss = -(y*np.log(y_hat_c) + (1 - y) * (np.log(1  - y_hat_c)))
    
    return loss
    
    
def focal_loss(y, y_hat, gamma):
    y = np.asarray(y, dtype=float)
    y_hat = np.asarray(y_hat, dtype=float)
    
    if y.shape != y_hat.shape:
        raise ValueError("y và y_hat must same Length")
    if gamma < 0:
        raise ValueError("gamma > 0")
    eps = 1e-12
    # y = 1: -((1 - y_hat) ** gamma) * log(y_hat)
    # y = 0: -(y_hat ** gamma) * log(1 - y_hat)
    
    y_hat_c = np.clip(y_hat, eps, 1.0 - eps)
    p = y * y_hat_c + (1 - y) * (1 - y_hat_c)
    
    mod = (1 - p) ** gamma
    bce_loss = -(y*np.log(y_hat_c) + (1 - y) * (np.log(1  - y_hat_c)))
    
    res = bce_loss * mod
    return res
    
y = np.array([0, 0, 1, 1])
y_hat = np.array([0.1, 0.8, 0.9, 0.2])

bce = binary_crossentropy(y, y_hat)
focal_0 = focal_loss(y, y_hat, gamma=0)
focal_2 = focal_loss(y, y_hat, gamma=2)

assert bce.shape == y.shape
assert focal_2.shape == y.shape
assert np.allclose(focal_0, bce)  # gamma = 0 thì focal loss bằng BCE
assert np.all(np.isfinite(binary_crossentropy(y, [0, 1, 1, 0])))

print("BCE từng mẫu:       ", bce)
print("Focal loss từng mẫu:", focal_2)
print("BCE trung bình:     ", np.mean(bce))