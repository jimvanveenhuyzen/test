import numpy as np

print("improved crout")

A = np.array([[3,8,1,-12,-4],[1,0,0,-1,0],[4,4,3,-40,-3],[0,2,1,-3,-2],[0,1,0,-12,0]], dtype=float)
b = np.array([2,0,1,0,0], dtype=float)

print(A)

def improved_crout(matrix):
    if matrix.shape[0] != matrix.shape[1]:
        return "Error: not a square matrix. No solution"
    LU = np.copy(matrix)
    index_max = np.zeros(len(matrix[0]), dtype=int)
    for k in range(len(matrix[0])):
        index_max[k] = int(k)
        for i in range(len(matrix[0])):
            if i >= k:
                if np.abs(matrix[i][k]) > np.abs(matrix[int(index_max[k])][k]): 
                    index_max[k] = i
        if index_max[k] != k: 
            for j in range(len(matrix[0])):
                LU[index_max][j] = LU[k][j]
        for i in range(len(matrix[0])):
            if i > k:
                LU[i][k] = LU[i][k] / LU[k][k]
                for j in range(len(matrix[0])):
                    if j > k:
                        LU[i][j] = LU[i][j] - LU[i][k]*LU[k][j]
    return LU

def forward_substitution(LU,b):
    y = np.zeros(b.shape)
    for i in range(len(y)):
        y[i] = b[i]
        for j in range(len(y)-1):
            if j < i: 
                y[i] = y[i] - LU[i][j]*y[j]
    return y 

def backward_substitution(LU,y):
    x = np.zeros(y.shape)
    N_1 = len(x)-1
    x[N_1] = y[N_1] / LU[N_1][N_1]
    for i in range(len(x)-1,-1,-1):
        x[i] = 1/LU[i][i] * y[i]
        for j in range(len(x)-1,-1,-1):
            if j > i:
                x[i] = x[i] - 1/LU[i][i] * LU[i][j]*x[j]
    return x 

def LU(matrix,sol):
    LU = improved_crout(matrix)
    y = forward_substitution(LU,sol)
    return backward_substitution(LU,y)

print(LU(A,b))

def matrix_multiplication(A,x):
    output = np.zeros(len(A))
    for i in range(len(A)):
        for j in range(len(A)):
            output[j] = output[j] + A[j][i] * x[i]
    return output

print(matrix_multiplication(A,LU(A,b)))

A_times_x = matrix_multiplication(A,LU(A,b))
delta_b = A_times_x - b
error = LU(A,delta_b)
print(error)
    