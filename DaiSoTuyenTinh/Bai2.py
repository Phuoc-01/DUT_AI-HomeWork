import numpy as np

def eigen_decomposition(A, num_iterations=1000, tol=1e-9):
    n = A.shape[0]
    V = np.zeros((n, n))
    eigenvalues = np.zeros(n)
    
    A_temp = A.copy().astype(float)
    
    for k in range(n):
        v = np.random.rand(n)
        v = v / np.linalg.norm(v)
        
        for _ in range(num_iterations):
            v_new = np.dot(A_temp, v)
            v_new = v_new / np.linalg.norm(v_new)
            
            if np.linalg.norm(v_new - v) < tol:
                break
            v = v_new
            
        lam = np.dot(v.T, np.dot(A_temp, v))
        eigenvalues[k] = lam
        V[:, k] = v
        
        A_temp = A_temp - lam * np.outer(v, v)
        
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    V = V[:, idx]
    
    return eigenvalues, V


def svd_decomposition(A):
    A = np.asarray(A, dtype=float)
    m, n = A.shape
    
    AtA = np.dot(A.T, A)
    
    eigenvalues, V = eigen_decomposition(AtA)
    
    singular_values = np.sqrt(np.maximum(eigenvalues, 0))
    
    S = np.zeros((m, n))
    for i in range(min(m, n)):
        S[i, i] = singular_values[i]
        
    U = np.zeros((m, m))
    k = np.sum(singular_values > 1e-10) 
    for i in range(k):
        U[:, i] = (1.0 / singular_values[i]) * np.dot(A, V[:, i])
        
    for i in range(k, m):
        u_new = np.random.rand(m)
        for j in range(i):
            u_new -= np.dot(u_new, U[:, j]) * U[:, j]
        U[:, i] = u_new / np.linalg.norm(u_new)
        
    Vt = V.T
    return U, S, Vt


if __name__ == "__main__":
    np.random.seed(42)
    
    print("1. KIỂM TRA PHÂN RÃ EIGEN")
    A_sym = np.random.randn(3, 3)
    A_sym = np.dot(A_sym, A_sym.T) 
    
    vals_my, vecs_my = eigen_decomposition(A_sym)
    
    vals_lib, vecs_lib = np.linalg.eigh(A_sym)
    
    idx = np.argsort(vals_lib)[::-1]
    vals_lib = vals_lib[idx]
    
    print("Trị riêng tự code: ", np.round(vals_my, 4))
    print("Trị riêng thư viện:", np.round(vals_lib, 4))
    print("=> Khớp trị riêng:", np.allclose(vals_my, vals_lib, atol=1e-5))
    
    v = vecs_my[:, 0]
    lam = vals_my[0]
    print("Kiểm tra A*v = lambda*v:", np.allclose(np.dot(A_sym, v), lam * v, atol=1e-5))
    
    print("\n" + "="*40 + "\n")
    
    print("2. KIỂM TRA PHÂN RÃ SVD")
    A_rect = np.random.randn(4, 3)
    
    U_my, S_my, Vt_my = svd_decomposition(A_rect)
    
    U_lib, S_lib, Vt_lib = np.linalg.svd(A_rect, full_matrices=True)
    
    print("Giá trị suy biến (S) tự code: ", np.round(S_my.diagonal(), 4))
    print("Giá trị suy biến (S) thư viện:", np.round(S_lib, 4))
    
    A_reconstructed = np.dot(U_my, np.dot(S_my, Vt_my))
    print("=> Tái tạo A thành công:", np.allclose(A_rect, A_reconstructed, atol=1e-5))