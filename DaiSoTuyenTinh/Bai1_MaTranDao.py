def mtcon(matrix,i,j):
    mtc = []
    for r in range(len(matrix)):
        if r == i:
            continue
        
        new_r = []
        for c in range(len(matrix[r])):
            if c == j:
                continue
            new_r.append(matrix[r][c])
        
        mtc.append(new_r)
        
    return mtc

def dt(matrix):
    n = len(matrix)
    
    if n == 1:
        return matrix[0][0]
    if n == 2:
        return matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]
    
    det = 0
    
    for i in range(n):
        s = 1 if i % 2 == 0 else -1
        det += s*matrix[0][i] * dt(mtcon(matrix,0,i))
    
    return det

def gauss_jordan(matrix):
    n  =len(matrix)
    
    det = dt(matrix)
    if det == 0:
        raise ValueError("Ma tran khong kha nghich")
    
    a = []
    
    for i in range(n):
        row = []
        for j in range(n):
            row.append(float(matrix[i][j]))
        for j in range(n):
            row.append(1.0 if i == j else 0.0)
        a.append(row)
    
    for i in range(n):
        p = a[i][i]
        if p == 0:
            for r in range(i+1,n):
                if a[r][i] != 0:
                    a[i], a[r] = a[r],a[i]
                    p = a[i][i]
                    break
                
        p = a[i][i]
        for j in range(2*n):
            a[i][j] /= p
        
        for k in range(n):
            if k != i:
                f = a[k][i]
                for j in range(2*n):
                    a[k][j] -= f * a[i][j]
                    
    tran = [row[n:] for row in a]
    return tran
            
            
def mtph(matrix):
    n = len(matrix)
    adj = [[0]*n for _ in range(n)]
    
    for i in range(n):
        for j in range(n):
            minor = mtcon(matrix, i, j)
            cofactor = ((-1)**(i+j)) * dt(minor)
            adj[j][i] = cofactor 
            
    return adj


def inverse_adjugate(matrix):
    det = dt(matrix)
    if det == 0:
        raise ValueError("Ma trận không khả nghịch (Định thức bằng 0)")
    
    adj = mtph(matrix)
    
    n = len(matrix)
    inverse = [[adj[i][j] / det for j in range(n)] for i in range(n)]
    
    return inverse


if __name__ == "__main__":
    A = [
        [2, -1, 0],
        [-1, 2, -1],
        [0, -1, 2]
    ]
    
    print("Ma trận A:", A)
    print("\n--- Nghịch đảo Gauss-Jordan ---")
    for row in gauss_jordan(A):
        print([round(x, 4) for x in row])
        
    print("\n--- Nghịch đảo Ma trận phụ hợp ---")
    for row in inverse_adjugate(A):
        print([round(x, 4) for x in row])