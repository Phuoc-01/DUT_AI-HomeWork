import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from abc import ABC, abstractmethod
from typing import Callable, Tuple

np.random.seed(42)

def loss_function(w1, w2):
    base = 0.05 * w1**2 + 0.04 * (w2 + 1.0)**2
    peak = 18.0 * np.exp(-((w1 - 3.3)**2 / 1.5 + (w2 - 0.0)**2 / 1.8))
    b_deep = -2.5 * np.exp(-((w1 + 2.5)**2 / 2.0 + (w2 - 2.5)**2 / 2.0))
    b_shallow = -1.2 * np.exp(-((w1 - 0.2)**2 / 1.0 + (w2 - 2.0)**2 / 1.0))
    b_puddle = -0.8 * np.exp(-((w1 - 0.1)**2 / 0.4 + (w2 + 1.0)**2 / 0.4))
    b_global = -4.5 * np.exp(-((w1 - 0.5)**2 / 2.0 + (w2 + 3.5)**2 / 2.0))
    return base + peak + b_deep + b_shallow + b_puddle + b_global

def compute_gradients(w1, w2):
    db_dw1 = 0.10 * w1
    db_dw2 = 0.08 * (w2 + 1.0)
    
    exp_peak = np.exp(-((w1 - 3.3)**2 / 1.5 + (w2 - 0.0)**2 / 1.8))
    dp_dw1 = 18.0 * exp_peak * (-2.0 * (w1 - 3.3) / 1.5)
    dp_dw2 = 18.0 * exp_peak * (-2.0 * (w2 - 0.0) / 1.8)
    
    exp_deep = np.exp(-((w1 + 2.5)**2 / 2.0 + (w2 - 2.5)**2 / 2.0))
    dd_dw1 = -2.5 * exp_deep * (-2.0 * (w1 + 2.5) / 2.0)
    dd_dw2 = -2.5 * exp_deep * (-2.0 * (w2 - 2.5) / 2.0)
    
    exp_sh = np.exp(-((w1 - 0.2)**2 / 1.0 + (w2 - 2.0)**2 / 1.0))
    ds_dw1 = -1.2 * exp_sh * (-2.0 * (w1 - 0.2) / 1.0)
    ds_dw2 = -1.2 * exp_sh * (-2.0 * (w2 - 2.0) / 1.0)
    
    exp_pud = np.exp(-((w1 - 0.1)**2 / 0.4 + (w2 + 1.0)**2 / 0.4))
    dpu_dw1 = -0.8 * exp_pud * (-2.0 * (w1 - 0.1) / 0.4)
    dpu_dw2 = -0.8 * exp_pud * (-2.0 * (w2 + 1.0) / 0.4)
    
    exp_glob = np.exp(-((w1 - 0.5)**2 / 2.0 + (w2 + 3.5)**2 / 2.0))
    dg_dw1 = -4.5 * exp_glob * (-2.0 * (w1 - 0.5) / 2.0)
    dg_dw2 = -4.5 * exp_glob * (-2.0 * (w2 + 3.5) / 2.0)
    
    dw1 = db_dw1 + dp_dw1 + dd_dw1 + ds_dw1 + dpu_dw1 + dg_dw1
    dw2 = db_dw2 + dp_dw2 + dd_dw2 + ds_dw2 + dpu_dw2 + dg_dw2
    return np.array([dw1, dw2], dtype=float)

start_scenarios = {
    "Vùng yên ngựa phẳng (Saddle Flat)": np.array([0.1, -0.1]),
    "Sườn dốc gần đỉnh núi (Steep Slope)": np.array([2.5, 1.5]),
    "Hẻm núi mấp mô (Rugged Ravine)": np.array([-2.5, 2.0]),
    "Góc đối diện xa (Far Basin Area)": np.array([-1.5, -4.0])
}

print("Khởi tạo không gian hàm Loss phân hóa thành công!")

class BaseOptimizer(ABC):
    def __init__(self, loss_fn: Callable, gradient_fn: Callable):
        self.loss_fn = loss_fn
        self.gradient_fn = gradient_fn 
        
    @abstractmethod    
    def run(self, w_start: np.ndarray, num_epochs=140) -> Tuple[np.ndarray, np.ndarray]:
        pass 

BaseOptmizer = BaseOptimizer

class GradientDescent(BaseOptimizer):
    def __init__(self, loss_fn: Callable, gradient_fn: Callable, lr=0.04):
        super().__init__(loss_fn, gradient_fn)
        self.lr = lr 
        
    def run(self, w_start: np.ndarray, num_epochs=140, verbose=False):
        w = np.copy(w_start).astype(float)
        history_w = [np.copy(w_start)]
        history_loss = [self.loss_fn(w_start[0], w_start[1])]
        for epoch in range(num_epochs):
            grad = self.gradient_fn(w[0], w[1])
            grad = np.clip(grad, -100.0, 100.0)
            
            w = w - self.lr * grad
            new_loss = self.loss_fn(w[0], w[1])
            if verbose and epoch % 20 == 0:
                print(f"Loss = {new_loss}")
            history_w.append(np.copy(w))
            history_loss.append(new_loss)
        return np.array(history_w), np.array(history_loss)
    
class Momentum(BaseOptimizer):
    def __init__(self, loss_fn: Callable, gradient_fn: Callable, v_0: float | None = None, lr=0.035, gamma=0.88):
        super().__init__(loss_fn, gradient_fn)
        self.lr = lr
        self.gamma = gamma
        self.v = v_0

    def run(self, w_start, num_epochs=140):
        if self.v is None:
            self.v = np.zeros_like(w_start)

        w = np.copy(w_start).astype(float)
        history_w = [np.copy(w_start)]
        history_loss = [self.loss_fn(w_start[0], w_start[1])]
        v = np.zeros_like(w) if not isinstance(self.v, np.ndarray) else np.copy(self.v)

        for epoch in range(num_epochs):
            grad = self.gradient_fn(w[0], w[1])
            grad = np.clip(grad, -100.0, 100.0)
            v = self.gamma * v + self.lr * grad
            w = w - v
            new_loss = self.loss_fn(w[0], w[1])
            history_w.append(np.copy(w))
            history_loss.append(new_loss)
        return np.array(history_w), np.array(history_loss)
    
class AdaGrad(BaseOptimizer):
    def __init__(self, loss_fn: Callable, gradient_fn: Callable, lr=0.25, eps=1e-8):
        super().__init__(loss_fn, gradient_fn)
        self.lr = lr
        self.eps = eps
        self.G = None

    def run(self, w_start, num_epochs=140):
        if self.G is None:
            self.G = np.zeros_like(w_start)

        w = np.copy(w_start).astype(float)
        history_w = [np.copy(w_start)]
        history_loss = [self.loss_fn(w_start[0], w_start[1])]
        G = np.zeros_like(w) if not isinstance(self.G, np.ndarray) else np.copy(self.G)

        for epoch in range(num_epochs):
            grad = self.gradient_fn(w[0], w[1])
            grad = np.clip(grad, -100.0, 100.0)
            G = G + grad**2
            w = w - (self.lr / (np.sqrt(G) + self.eps)) * grad
            new_loss = self.loss_fn(w[0], w[1])
            history_w.append(np.copy(w))
            history_loss.append(new_loss)
        return np.array(history_w), np.array(history_loss)
    
class RMSProp(BaseOptimizer):
    def __init__(self, loss_fn: Callable, gradient_fn: Callable, lr=0.035, beta=0.9, eps=1e-8):
        super().__init__(loss_fn, gradient_fn)
        self.lr = lr
        self.beta = beta
        self.eps = eps
        self.v = None

    def run(self, w_start, num_epochs=140):
        if self.v is None:
            self.v = np.zeros_like(w_start)

        w = np.copy(w_start).astype(float)
        history_w = [np.copy(w_start)]
        history_loss = [self.loss_fn(w_start[0], w_start[1])]
        v = np.zeros_like(w) if not isinstance(self.v, np.ndarray) else np.copy(self.v)

        for epoch in range(num_epochs):
            grad = self.gradient_fn(w[0], w[1])
            grad = np.clip(grad, -100.0, 100.0)
            v = self.beta * v + (1 - self.beta) * (grad**2)
            w = w - (self.lr / (np.sqrt(v) + self.eps)) * grad
            new_loss = self.loss_fn(w[0], w[1])
            history_w.append(np.copy(w))
            history_loss.append(new_loss)
        return np.array(history_w), np.array(history_loss)
    
class Adam(BaseOptimizer):
    def __init__(self, loss_fn: Callable, gradient_fn: Callable, lr=0.35, beta1=0.9, beta2=0.999, eps=1e-8):
        super().__init__(loss_fn, gradient_fn)
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.m = None
        self.v = None
        self.t = 0

    def run(self, w_start, num_epochs=140):
        if self.m is None:
            self.m = np.zeros_like(w_start)
            self.v = np.zeros_like(w_start)

        w = np.copy(w_start).astype(float)
        history_w = [np.copy(w_start)]
        history_loss = [self.loss_fn(w_start[0], w_start[1])]
        m = np.zeros_like(w) if not isinstance(self.m, np.ndarray) else np.copy(self.m)
        v = np.zeros_like(w) if not isinstance(self.v, np.ndarray) else np.copy(self.v)
        t = 0

        for epoch in range(num_epochs):
            t += 1
            grad = self.gradient_fn(w[0], w[1])
            grad = np.clip(grad, -100.0, 100.0)
            m = self.beta1 * m + (1 - self.beta1) * grad
            v = self.beta2 * v + (1 - self.beta2) * (grad**2)
            m_hat = m / (1 - self.beta1**t)
            v_hat = v / (1 - self.beta2**t)
            w = w - (self.lr / (np.sqrt(v_hat) + self.eps)) * m_hat
            new_loss = self.loss_fn(w[0], w[1])
            history_w.append(np.copy(w))
            history_loss.append(new_loss)
        return np.array(history_w), np.array(history_loss)
    

def create_optimizers() -> dict[str, BaseOptimizer]: 
    return {
        "Gradient Descent": GradientDescent(loss_fn=loss_function, gradient_fn=compute_gradients, lr=0.04),
        "Momentum": Momentum(loss_fn=loss_function, gradient_fn=compute_gradients, lr=0.035, gamma=0.88),
        "AdaGrad": AdaGrad(loss_fn=loss_function, gradient_fn=compute_gradients, lr=0.25),
        "RMSProp": RMSProp(loss_fn=loss_function, gradient_fn=compute_gradients, lr=0.035, beta=0.9),
        "Adam": Adam(loss_fn=loss_function, gradient_fn=compute_gradients, lr=0.35, beta1=0.9, beta2=0.999),
    }

# Chạy thực nghiệm
all_scenario_results = {}
epochs = 140
for sc_name, w_start in start_scenarios.items():
    opts = create_optimizers()
    sc_res = {}
    for opt_name, opt in opts.items():
        traj, losses = opt.run(w_start, num_epochs=epochs)
        sc_res[opt_name] = {"traj": traj, "loss": losses}
    all_scenario_results[sc_name] = sc_res

# Thiết lập màu sắc và marker
opt_colors = {'Gradient Descent': '#f87171', 'Momentum': '#fb923c', 'AdaGrad': '#facc15', 'RMSProp': '#38bdf8', 'Adam': '#c084fc'}
opt_markers = {'Gradient Descent': 'o', 'Momentum': '^', 'AdaGrad': 'v', 'RMSProp': 'x', 'Adam': '*'}

# 1. Vẽ bản đồ địa hình ban đầu
w1_vals = np.linspace(-5, 5, 250)
w2_vals = np.linspace(-5, 5, 250)
W1, W2 = np.meshgrid(w1_vals, w2_vals)
Z = loss_function(W1, W2)

fig = plt.figure(figsize=(18, 8))
ax1 = fig.add_subplot(1, 2, 1, projection="3d")
surf = ax1.plot_surface(W1, W2, Z, cmap="viridis", edgecolor="none", alpha=0.8)
start_colors = ["lime", "cyan", "magenta", "orange"]
for (label, pt), col in zip(start_scenarios.items(), start_colors):
    z_pt = loss_function(pt[0], pt[1])
    ax1.scatter(pt[0], pt[1], z_pt, color=col, edgecolors="black", s=110, marker="D", label=f"Start: {label}", zorder=10)
z_global = loss_function(0.5, -3.5)
ax1.scatter([0.5], [-3.5], [z_global], color="red", edgecolors="white", s=160, marker="*", label="Global Min (0.5, -3.5)", zorder=11)
ax1.set_title("Không Gian Loss Landscape 3D & Các Điểm Khởi Tạo", fontsize=13, fontweight="bold")
ax1.set_xlabel("$w_1$"); ax1.set_ylabel("$w_2$"); ax1.set_zlabel("Loss")
ax1.view_init(elev=50, azim=-118); ax1.legend(loc="upper right", fontsize=8.5)

ax2 = fig.add_subplot(1, 2, 2)
contour = ax2.contourf(W1, W2, Z, levels=35, cmap="viridis")
lines = ax2.contour(W1, W2, Z, levels=14, colors="black", alpha=0.35, linewidths=0.7)
ax2.clabel(lines, inline=True, fontsize=8)
fig.colorbar(contour, ax=ax2, label="Loss", fraction=0.046, pad=0.04)
for (label, pt), col in zip(start_scenarios.items(), start_colors):
    ax2.scatter(pt[0], pt[1], color=col, edgecolors="black", s=100, marker="D", label=f"Start: {label}", zorder=5)
ax2.scatter([0.5], [-3.5], color="red", edgecolors="white", s=150, marker="*", label="Global Min (0.5, -3.5)", zorder=6)
ax2.set_title("Bản Đồ 2D Contour & Các Điểm Khởi Tạo", fontsize=13, fontweight="bold")
ax2.set_xlabel("$w_1$"); ax2.set_ylabel("$w_2$")
plt.tight_layout(); plt.show()

# 2. Vẽ 4 biểu đồ Contour so sánh quỹ đạo
fig, axes = plt.subplots(2, 2, figsize=(18, 16))
axes = axes.flatten()
for idx, (sc_name, w_start) in enumerate(start_scenarios.items()):
    ax = axes[idx]
    sc_res = all_scenario_results[sc_name]
    ax.contourf(W1, W2, Z, levels=35, cmap='viridis', alpha=0.9)
    c_lines = ax.contour(W1, W2, Z, levels=14, colors='black', alpha=0.3, linewidths=0.6)
    ax.clabel(c_lines, inline=True, fontsize=8)
    ax.scatter(w_start[0], w_start[1], color='lime', edgecolors='black', s=130, marker='D', zorder=10, label='Start Point')
    ax.scatter([0.5], [-3.5], color='red', edgecolors='white', s=140, marker='*', zorder=10, label='Global Min')
    for opt_name, res in sc_res.items():
        traj = res['traj']
        ax.plot(traj[:, 0], traj[:, 1], label=opt_name, color=opt_colors[opt_name], linewidth=2.2, alpha=0.85)
        ax.scatter(traj[-1, 0], traj[-1, 1], color=opt_colors[opt_name], edgecolors='black', s=80, marker=opt_markers[opt_name], zorder=9)
    ax.set_title(f'Kịch bản {idx+1}: {sc_name}', fontsize=12, fontweight='bold')
    ax.set_xlabel('$w_1$'); ax.set_ylabel('$w_2$')
    ax.set_xlim(-5, 5); ax.set_ylim(-5, 5)
    ax.legend(loc='lower left', framealpha=0.9, fontsize=9)
plt.tight_layout(); plt.show()

# 3. Vẽ 4 biểu đồ đường cong Loss
fig, axes = plt.subplots(2, 2, figsize=(18, 12))
axes = axes.flatten()
for idx, (sc_name, _) in enumerate(start_scenarios.items()):
    ax = axes[idx]
    sc_res = all_scenario_results[sc_name]
    for opt_name, res in sc_res.items():
        ax.plot(res['loss'], label=opt_name, color=opt_colors[opt_name], linewidth=2.2)
    ax.set_title(f'Đường cong Loss: {sc_name}', fontsize=12, fontweight='bold')
    ax.set_xlabel('Epoch'); ax.set_ylabel('Loss Value')
    ax.grid(True, linestyle='--', alpha=0.5); ax.legend(framealpha=0.9, fontsize=9)
plt.tight_layout(); plt.show()

# 4. In bảng so sánh định lượng
for sc_name, sc_res in all_scenario_results.items():
    print("=" * 90)
    print(f"Kịch bản: {sc_name.upper()}")
    print("-" * 90)
    print(f"{'Thuật toán (Optimizer)':<20} | {'Tọa độ cuối (w1, w2)':<24} | {'Loss cuối':<12} | {'Đánh giá hội tụ'}")
    print("-" * 90)
    for opt_name, res in sc_res.items():
        traj = res['traj']; losses = res['loss']
        w_end = f"({traj[-1, 0]:.3f}, {traj[-1, 1]:.3f})"
        final_loss = f"{losses[-1]:.4f}"
        dist_to_global = np.linalg.norm(traj[-1] - np.array([0.5, -3.5]))
        if dist_to_global < 0.8: status = "🎯 Chạm Cực Tiểu Toàn Cục"
        elif losses[-1] < 2.0: status = "⚠️ Kẹt Cực Tiểu Địa Phương"
        else: status = "❌ Hội tụ chậm / Chưa tới đáy"
        print(f"{opt_name:<20} | {w_end:<24} | {final_loss:<12} | {status}")
    print("=" * 90 + "\n")