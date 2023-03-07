import numpy as np

print("problem 1a")

A = np.array([[3,8,1,-12,-4],[1,0,0,-1,0],[4,4,3,-40,-3],[0,2,1,-3,-2],[0,1,0,-12,0]], dtype=float)
b = np.array([2,0,1,0,0], dtype=float)

def gaussian_elem(matrix,sol):
    if matrix.shape[0] != matrix.shape[1]:
        return "Error: not a square matrix. No solution"
    if matrix.shape[0] != sol.shape[0]:
        return "Error: A has different dimension from b. No solution"
    A = np.copy(matrix)
    b = np.copy(sol)
    pivot = np.zeros(b.shape)
    x = np.copy(pivot)
    for i in range(len(A[0])):
        for j in range(len(A[0])):
            if i == j:
                pivot[i] = A[j][i]
                if pivot[i] != 0: #checks if a pivot is 0, if so, return that matrix is singular 
                    A[j][:] = A[j][:] / pivot[i]
                    b[i] = b[i] / pivot[i]
                else:
                    return "The matrix is singular. No solution"
            if j > i and A[j][i] != 0:
                factor = A[j][i]
                A[j][:] = A[j][:] - factor * A[i][:]
                b[j] = b[j] - factor * b[i]
   
    for i in range(len(x)-1,-1,-1):
        x[i] = b[i]
        for j in range(len(x)-1,-1,-1):
            if j > i:
                x[i] = x[i] - A[i][j]*x[j] 
    return x

print(gaussian_elem(A,b))
        
print("problem 1b")

def crout(matrix):
    if matrix.shape[0] != matrix.shape[1]:
        return "Error: not a square matrix. No solution"
    L = np.zeros(shape=matrix.shape)
    U = np.copy(L)
    for i in range(len(matrix[0])):
        for j in range(len(matrix[0])):
            if i == j:
                L[i][j] = 1
            if i <= j:
                U[i][j] = matrix[i][j]
                for k in range(len(matrix[0]-1)):
                    if k < i:
                        U[i][j] = U[i][j] - L[i][k]*U[k][j]
            if i > j:
                L[i][j] = 1/U[j][j] * matrix[i][j]
                for k in range(len(matrix[0]-1)):
                    if k < j and U[j][j] != 0:
                        L[i][j] = L[i][j] - 1/U[j][j] * L[i][k]*U[k][j]
    return L,U

lower = crout(A)[0]
upper = crout(A)[1]
print(lower)
print(upper)
x_values1 = np.around(np.matmul(lower,upper),1)
#print(x_values1)

import scipy.linalg as la

#print(A)
P,L,U = la.lu(A)
#print(L)
#print(U)

x_values2 = np.around(np.matmul(lower,upper),1)
#print(x_values2)

y_test = gaussian_elem(lower,b)
x_test = gaussian_elem(upper,y_test)
print(x_test) #correct solution

def forward_sub(lower,b):
    y = np.zeros(b.shape)
    y[0] = b[0] / lower[0][0]
    for i in range(len(y)):
        y[i] = 1/lower[i][i] * b[i]
        for j in range(len(y)-1):
            if j < i: 
                y[i] = y[i] - 1/lower[i][i] * lower[i][j]*y[j]
    
    return y

def backward_sub(upper,y):
    x = np.zeros(y.shape)
    N_1 = len(x)-1
    x[N_1] = y[N_1] / upper[N_1][N_1]
    for i in range(len(x)-1,-1,-1):
        x[i] = 1/upper[i][i] * y[i]
        for j in range(len(x)-1,-1,-1):
            if j > i:
                x[i] = x[i] - 1/upper[i][i] * upper[i][j]*x[j]
    return x 

def LU_decomp(matrix,b):
    L = crout(matrix)[0]
    U = crout(matrix)[1]
    y = forward_sub(L,b)
    print(y)
    return backward_sub(U,y)

print(LU_decomp(A,b))

#x_order1 = LU_decomp(A,b)
#A_times_x = np.around(np.matmul(A,x_order1))
#print(A_times_x)


