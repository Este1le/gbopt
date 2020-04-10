import numpy as np
import itertools
from gbopt.graph.graph import Graph

def objective_function1d(x):
    x = np.array([x])
    y = 0.5*np.sin(3*x[0]) * x[0] # x_max around 9
    return y

n = 101
nb = 4
X = np.linspace(0, 10, num=n).reshape(n, -1)
Y = np.array([objective_function1d(x) for x in X]).reshape(n, -1)
I = []
W = []
for i in range(n):
    if i == 0:
        I.append([1,2,3,4])
        W.append([1,1/2,1/3,1/4])
    elif i==1:
        I.append([0,2,3,4])
        W.append([1,1,1/2,1/3])
    elif i==n-2:
        I.append([n-5,n-4,n-3,n-1])
        W.append([1/3,1/2,1,1])
    elif i==n-1:
        I.append([n-5,n-4,n-3,n-2])
        W.append([1/4,1/3,1/2,1])
    else:
        I.append([i-2,i-1,i+1,i+2])
        W.append([1/2,1,1,1/2])

I = np.array(I)
W = np.array(W)
graph1d = Graph(X, Y, I, W)


def objective_function2d(x):
    y = x[0]**2 + x[1]**2
    return y
x = np.linspace(0, 10, num=11)
X = np.array([list(it) for it in itertools.product(x, repeat=2)])
Y = np.array([objective_function2d(x) for x in X]).reshape(X.shape[0], -1)
I = []
W = []
for i in range(X.shape[0]):
    x1 = X[i]
    I_x1 = []
    W_x1 = []
    for j in range(X.shape[0]):
        x2 = X[j]
        if i != j:
            sd = np.sum(np.abs(x1-x2))
            if sd <= 2:
                I_x1.append(j)
                W_x1.append(1./sd)
    I.append(I_x1)
    W.append(W_x1)
I = np.array(I)
W = np.array(W)

graph2d = Graph(X, Y, I, W)
