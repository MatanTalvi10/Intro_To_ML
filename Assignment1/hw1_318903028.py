import numpy as np
import matplotlib.pyplot as plt
import math
from functions import *


# Q1
## a

N = 200000
n = 20
A_assist = np.random.binomial(size=N*n, n=1, p= 0.5)
A = A_assist.reshape((N,n))
epsilon_50 = np.linspace(0,1,50)
x = []  #epsilon values
y = []  #empirical_prob
A_means = np.mean(A,axis=1)

for eps in epsilon_50:
    x.append(eps)
    cnt = 0
    for i in range(N):
        cnt += emprical_assist(A_means[i],eps)
    y.append(cnt/N)
    
plt.title("Q1.b")
plt.xlabel("Epsilon")
plt.ylabel("Empirical Probability")
plt.plot(x,y)
plt.show()




## b