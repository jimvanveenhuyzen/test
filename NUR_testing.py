import numpy as np 

def factorial(n):
    value = 1.0
    for i in range(1,n+1):
        value *= i
    return value

def ln_factorial(n):
    value = 0.0
    for i in range(1,n+1):
        value += np.log(i)
    return value 

print(ln_factorial(5))


def ln_poisson(lambdas,k):
    return np.int32(k)*np.float32(np.log(lambdas)) - lambdas - np.float32(ln_factorial(k))

print(ln_poisson(1,0))
print(ln_poisson(5,10))
print(ln_poisson(3,21))
print(ln_poisson(2.6,40))
print(ln_poisson(101,200))

"""
print(pivot)

for i in range(len(A[0])):
    print('column',A[0][i])
    largest_value = np.abs(A[0][i])
    for j in range(len(A[0])):
        print('row',A[j][i])
        if np.abs(A[j][i]) > np.abs(largest_value):
            largest_value = A[j][i]
            print('largest value at index',j)

print(A)
    
for i in range(len(A[0])):
    print('column',A[0][i])
    largest_value = 0
    for j in range(len(A[0])):
        print('row',A[j][i])
        if j > i:   
            if np.abs(A[j][i]) > np.abs(largest_value):
                largest_value = A[j][i]
                pivot[i] = A[j][i]
                pivot_index[i] = j 
                print('largest value at index',j)
          
print(A)
print(pivot)
print(pivot_index)
print(b)

print('new attempt')

x[4] = b[4]
x[3] = b[3] - A[3][4]*x[4] 
x[2] = b[2] - A[2][4]*x[4] - A[2][3]*x[3]
x[1] = b[1] - A[1][4]*x[4] - A[1][3]*x[3] - A[1][2]*x[2]
x[0] = b[0] - A[0][4]*x[4] - A[0][3]*x[3] - A[0][2]*x[2] - A[0][1]*x[1]
print(x)
"""
