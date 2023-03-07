#Tutorial 1 NUR
import numpy as np
import matplotlib.pyplot as plt

#1a
def sinc_pow(x,n): #sinc func using the power series expansion
    value = 0
    for i in range(n+1):
        value += (-1)**i * np.float64(x**(2*i)) / np.float64(np.math.factorial(2*i+1))
    return value

def sinc_lib(x): #sinc func using library function
    return np.sin(x)/x
    

#truncation error

#b

print(sinc_pow(7,8)) #a lower value for n, say n = 5, gives much worse accuracy
print(sinc_lib(7)) 

A = np.linspace(0,8,9)
B = []
C = []
error = []
for i in range(len(A)):
    B.append(sinc_pow(7,int(A[i])))
    C.append(sinc_pow(2,int(A[i])))
    error.append(C[i]-sinc_lib(2))

plt.plot(A,B,label='Value of sinc(x) for varying n')
plt.xlabel('n')
plt.ylabel('sinc(x)')
plt.legend()
plt.title("1B, error")
plt.show()


#we can see oscillation due to the nature of the sinc(x) function

#c

#testing different values, we see that the accuracy is good for x ~ n. 

print(sinc_lib(2))
#print(error)

n_values = np.linspace(0,15,16)
sinc = []
err = []

for i in range(len(n_values)):
    sinc.append(sinc_pow(2,int(n_values[i])))
    err.append(sinc_pow(2,int(n_values[i])) - sinc_lib(2))
    if i > 0 and err[i] == err[i-1]: #check where error becomes constant 
        print(i-1)
    

plt.plot(n_values,sinc,label='Value of sinc(x) for varying n')
plt.hlines(sinc_lib(2),0,8,label='sinc(x) library')
plt.xlabel('n')
plt.ylabel('sinc(x)')
plt.legend()
plt.show()

print(err)

#we see that at n = 11, there is no change in error, so max is reached at n=10
#this minimal error is 1.66e-16

#2a

BH = np.random.normal(10**6,10**5,10000)
BH_dist = np.histogram(BH,bins=20)
plt.plot(BH_dist[1][0:20],BH_dist[0])
plt.show()

#2b

import timeit

code = '''

BH = np.random.normal(10**6,10**5,10000)
c = 3e5 #in km/s
G = 4.3e-3 #in solar masses, parsec and km/s

def SC_radius(M):
    return 2*G*M / (c**2) * 206265 #in AU
SC_radius(BH)

'''

print(timeit.timeit(code,'import numpy as np',number=1000))
#print(timeit.timeit(lambda: SC_radius(BH),number=1000))

code2 = '''

BH = np.random.normal(10**6,10**5,10000)
c = 3e5 #in km/s
G = 4.3e-3 #in solar masses, parsec and km/s
c_inv2 = (1/c) * (1/c)

def SC_radius_2(M):
    return 2*G*M * c_inv2 * 206265
SC_radius_2(BH)

''' 
print(timeit.timeit(code2,'import numpy as np',number=1000))

#print(timeit.timeit(lambda: SC_radius_2(BH),number=1000))

BH = np.random.normal(10**6,10**5,10000)
c = 3e5 #in km/s
G = 4.3e-3 #in solar masses, parsec and km/s

def SC_radius(M):
    return 2*G*M / (c**2) * 206265 #in AU
SC_radius(BH)

start = timeit.default_timer()

for i in range(1000):
    SC_radius(BH)
    
print('time',(timeit.default_timer() - start)/1000,'s')

start = timeit.default_timer()

c_inv2 = (1/c) * (1/c)

def SC_radius_2(M):
    return 2*G*M * c_inv2 * 206265
SC_radius_2(BH)

for i in range(1000):
    SC_radius_2(BH)

print('time',(timeit.default_timer() - start)/1000,'s')
    



