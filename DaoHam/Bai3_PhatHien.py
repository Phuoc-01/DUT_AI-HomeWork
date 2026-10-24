import numpy as np
import matplotlib.pyplot as plt

def compute_x_derivative(image):
    
    # Chuyển sang float để tránh tràn số 
    image = np.asarray(image, dtype=float)
    gx = np.zeros_like(image)
    
    # Cột j chạy từ 1 đến width-2
    # image[:, 2:] tương ứng với I[i, j+1] (dịch sang phải 1)
    # image[:, :-2] tương ứng với I[i, j-1] (dịch sang trái 1)
    gx[:, 1:-1] = image[:, 2:] - image[:, :-2]
    
    return gx

def compute_y_derivative(image):
    image = np.asarray(image, dtype=float)
    gy = np.zeros_like(image)
    
    # hàng từ 1 đến height-2
    # image[2:, :] tương ứng với I[i+1, j] (dịch xuống 1)
    # image[:-2, :] tương ứng với I[i-1, j] (dịch lên 1)
    gy[1:-1, :] = image[2:, :] - image[:-2, :]
    
    return gy

def compute_magnitude(image):
    gx = compute_x_derivative(image)
    gy = compute_y_derivative(image)
    
    # np.hypot tự động tính sqrt(x^2 + y^2) an toàn và nhanh
    return np.hypot(gx, gy)

# PHẦN KIỂM TRA VÀ VẼ ĐỒ THỊ (Giữ nguyên)
image = np.zeros((100, 100), dtype=np.uint8)
image[25:75, 30:70] = 200
gx = compute_x_derivative(image)
gy = compute_y_derivative(image)
magnitude = compute_magnitude(image)

assert gx.shape == gy.shape == magnitude.shape == image.shape
assert np.allclose(gx[:, [0, -1]], 0) and np.allclose(gy[[0, -1], :], 0)
assert np.any(gx != 0) and np.any(gy != 0)
assert np.allclose(magnitude, np.hypot(gx, gy))

fig, axes = plt.subplots(1, 4, figsize=(14, 3.5))
for ax, arr, title, cmap in zip(
    axes, [image, gx, gy, magnitude],
    ["Ảnh gốc", "Gx", "Gy", "Độ lớn gradient"],
    ["gray", "coolwarm", "coolwarm", "gray"]):
    ax.imshow(arr, cmap=cmap); ax.set_title(title); ax.axis("off")
plt.tight_layout(); plt.show()