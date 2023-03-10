import numpy as np
import sys
import os
import matplotlib.pyplot as plt

data=np.genfromtxt(os.path.join(sys.path[0],"Vandermonde.txt"),\
                   comments='#',dtype=np.float64)
x=data[:,0]
y=data[:,1]
xx=np.linspace(x[0],x[-1],1000) #x values to interpolate at

V = np.zeros((20,20))

for i in range(len(V[0])):
    for j in range(len(V[0])):
        V[i][j] = x[i]**(j)
        
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

c = LU(V,y)
print(c)

y_polynomial = np.zeros(len(xx))
y_diff = np.zeros(len(y))

for i in range(len(y_polynomial)):
    for j in range(len(c)):
        y_polynomial[i] = y_polynomial[i] + c[j]*(xx[i]**j)
    if i % 50 == 0:
        y_diff[int(i/50)] = np.abs(y_polynomial[i] - y[int(i/50)])

fig,ax=plt.subplots()
ax.plot(x,y,marker='o',linewidth=0)
ax.plot(xx,y_polynomial)
plt.xlim(-1,101)
plt.ylim(-400,400)
ax.set_xlabel('$x$')
ax.set_ylabel('$y$')
plt.show()

fig,ax=plt.subplots()
ax.plot(x,y_diff)
plt.ylim(0,150)
ax.set_xlabel('$x$')
ax.set_ylabel('$|y(x)-y_i|$')
plt.show()

def neville(x,xdata,ydata):
    n = len(xdata) #order M-1
    P = np.copy(ydata)
    for k in range(1,n):
        for i in range(0,n-k):
            P[i] = ((xdata[i+k] - x) * P[i] + (x - xdata[i]) * P[i+1])\
                / (xdata[i+k] - xdata[i])
    return P[0]

y_interp = np.zeros(len(xx))
for i in range(len(xx)):
    y_interp[i] = neville(xx[i],x,y)
    
y_diff_b = np.zeros(len(y))
    
for i in range(len(y_interp)):
    if i % 50 == 0:
        y_diff_b[int(i/50)] = np.abs(y_interp[i] - y[int(i/50)])
    
fig,ax=plt.subplots()
ax.plot(x,y,marker='o',linewidth=0)
ax.plot(xx,y_interp)
plt.xlim(-1,101)
plt.ylim(-400,400)
ax.set_xlabel('$x$')
ax.set_ylabel('$y$')
plt.show()

fig,ax=plt.subplots()
ax.plot(x,y_diff, label='2a')
ax.plot(x,y_diff_b, label='2b')
plt.ylim(0,150)
ax.set_xlabel('$x$')
ax.set_ylabel('$|y(x)-y_i|$')
plt.legend(loc='upper left')
plt.show()

#2C

def matrix_multiplication(A,x):
    output = np.zeros(len(A))
    for i in range(len(A)):
        for j in range(len(A)):
            output[j] = output[j] + A[j][i] * x[i]
    return output

def LU_iterations(A,b,number):
    improved_sol = LU(A,b)
    for i in range(number-1):
        A_times_x = matrix_multiplication(A,improved_sol)
        delta_b = A_times_x - b
        error = LU(A,delta_b)
        improved_sol = LU(A,b) - error
    return improved_sol


A_times_x = matrix_multiplication(V,LU(V,y))
delta_y = A_times_x - y
error = LU(V,delta_y)
print(c)
print(error)
new_c = LU(V,y) - error
print(new_c)
print(LU_iterations(V,y,5))

#print(c)
c_10it = LU_iterations(V,y,10)
#print(c_10it)
y_polynomial_10it = np.zeros(len(xx))

for i in range(len(y_polynomial_10it)):
    for j in range(len(c_10it)):
        y_polynomial_10it[i] = y_polynomial_10it[i] + c_10it[j]*(xx[i]**j)
        
#print(y_polynomial[0:50])
#print(y_polynomial_10it[0:50])

fig,ax=plt.subplots()
ax.plot(x,y,marker='o',linewidth=0,label='data')
ax.plot(xx,y_polynomial,label='1 iter')
ax.plot(xx,y_polynomial_10it,label='10 iter')
plt.xlim(-1,101)
plt.ylim(-400,400)
plt.legend()
ax.set_xlabel('$x$')
ax.set_ylabel('$y$')
plt.show()

#2D

import timeit

begin_2a = timeit.default_timer()

for k in range(100): #100 iterations 
    c = LU(V,y)
    y_polynomial = np.zeros(len(xx))
    for i in range(len(y_polynomial)):
        for j in range(len(c)):
            y_polynomial[i] = y_polynomial[i] + c[j]*(xx[i]**j)
    
print('time taken for 2a', (timeit.default_timer() - begin_2a)/100, 's')

begin_2b = timeit.default_timer()

for k in range(10): #10 iterations 
    y_interp = np.zeros(len(xx))
    for i in range(len(xx)):
        y_interp[i] = neville(xx[i],x,y)
        
print('time taken for 2b', (timeit.default_timer() - begin_2b)/10, 's')



