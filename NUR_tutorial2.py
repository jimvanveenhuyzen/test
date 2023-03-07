import numpy as np
import matplotlib.pyplot as plt 

#2A

t = np.array([1.0000,4.3333,7.6667,11.000,14.333,17.667,21.000])
I = np.array([1.4925,15.323,3.2356,-29.472,-22.396,24.019,36.863])
t_interp = np.linspace(0,40,101)

def bisection(x,sample,order,err):
    if x < sample[0]: #extrapolation: if x < smallest sample value, j_low=0
        return 0 
    if x > sample[-1]: #extrapolation: if x > largest sample value, j_low=N-2
        return int(len(sample)-order)
    start = 0
    end = len(sample)-1
    size = (end-start)*0.5
    while np.abs(start-end) > err:
        if sample[int(start)] <= x <= sample[int(start+size)]:
            start = start
            end = end - size
        else: 
            start = start + size
            end = end 
        size = (end-start)*0.5
    return int(start) #return the j_low index

def linear_interp(x,xdata,ydata):
    y_interp = np.zeros(len(x))
    for i in range(len(x)):
        j_low = bisection(x[i],xdata,2,0.1) #choose small enough error 
        a = (ydata[j_low+1]-ydata[j_low])/(xdata[j_low+1]-xdata[j_low])
        b = ydata[j_low]
        y_interp[i] = a*(x[i]-xdata[j_low]) + b
    return y_interp

#print(linear_interp(t_interp,t,I))

plt.scatter(t,I,label='data')
plt.plot(t_interp,linear_interp(t_interp,t,I),label='interp')
plt.xlabel('t')
plt.ylabel('I')
plt.legend(loc='upper left')
plt.show()

#2B

x_interp = np.linspace(1,10,101)
print(t)
print(I)
print(x_interp[30])

M = 4
indices = np.zeros(M)
for i in range(len(indices)):
    if i == 0:
        indices[i] = bisection(x_interp[30],t,4,0.1)
    else:
        indices[i] = indices[i-1] + 1
        

P = np.zeros((4,4))
print(indices)
print(int(indices[0]))
print(P)
for i in range(len(P[0])):
    P[i][0] = I[int(indices[i])]

print(P)

for i in range(1,len(P[0])):
    print('i', i)
    for j in range(len(P[0])-i):
            print('j', j)
            P[j][i] = ((t[int(indices[j+1])]-x_interp[30])*P[j][i-1]\
                       +(x_interp[30]-t[int(indices[j])])*P[j+1][i-1])\
                /(t[int(indices[j+1])] - t[int(indices[j])])
            
            
print(P)
print(P[0][3])  
            
def neville(x,xdata,ydata):
    indices = np.zeros(4) #choose M=4
    for i in range(len(indices)):
        if i == 0:
            indices[i] = bisection(x,xdata,4,0.1)
        else:
            indices[i] = indices[i-1] + 1
    P = np.zeros((4,4))
    
    #print(indices)
    for i in range(len(P[0])):
        P[i][0] = I[int(indices[i])]
        
    for i in range(1,len(P[0])):
        #print('i', i)
        for j in range(len(P[0])-i):
                P[j][i] = ((xdata[int(indices[j+1])]-x)*P[j][i-1]\
                           +(x-xdata[int(indices[j])])*P[j+1][i-1])\
                    /(xdata[int(indices[j+1])] - xdata[int(indices[j])])
                #print(P)
    return P[0][3]

y_interp = np.zeros(len(x_interp))
for i in range(len(x_interp)):
    y_interp[i] = neville(x_interp[i],t,I)
    
"""
plt.scatter(t,I,label='data')
plt.plot(x_interp,y_interp,label='interp')
plt.xlabel('t')
plt.ylabel('I')
plt.legend(loc='upper left')
plt.show()
"""


x_interp = np.linspace(1,21,50)
t_interp = np.linspace(0,40,101)

print('new attempt')

def nevilles(x,xdata,ydata):
    n = len(xdata)
    P = np.copy(ydata)
    #closest_x = bisection(x,xdata,2,0.1)
    for k in range(1,n):
        for i in range(0,n-k):
            P[i] = ((xdata[i+k] - x) * P[i] + (x - xdata[i]) * P[i+1]) / (xdata[i+k] - xdata[i])
    return P[0]

print(x_interp[30])
print(nevilles(x_interp[30],t,I))

y_interp = np.zeros(len(x_interp))
for i in range(len(x_interp)):
    y_interp[i] = nevilles(x_interp[i],t,I)
    
print(y_interp)

I_interp = np.zeros(len(t_interp))
for i in range(len(t_interp)):
    I_interp[i] = nevilles(t_interp[i],t,I)

plt.scatter(t,I,label='data')
plt.plot(x_interp,y_interp,label='interp')
#plt.plot(t_interp,I_interp)
plt.xlabel('t')
plt.ylabel('I')
plt.legend(loc='upper left')
plt.show()

    
    
    

