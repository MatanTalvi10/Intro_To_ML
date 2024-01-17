import numpy as np
import matplotlib.pyplot as plt
import math
from functions import *

def main():
    Q1()


def Q1():
## a
    N = 200000
    n = 20
    A_assist = np.random.binomial(size=N*n, n=1, p= 0.5)
    A = A_assist.reshape((N,n))
    epsilon_50 = np.linspace(0,1,50)
    x = []  #epsilon values
    y = []  #empirical_prob
    h_b = [] #Hoeffding bound
    A_means = np.mean(A,axis=1)

    ## b
    for eps in epsilon_50:
        x.append(eps)
        cnt = 0
        for i in range(N):
            cnt += emprical_assist(A_means[i],eps)
        y.append(cnt/N)
        h_b.append(2*math.exp(-2*n*(eps**2)))

    plt.title("Q1.b")
    plt.xlabel("Epsilon")
    plt.ylabel("Empirical Probability")
    plt.plot(x,y)
    plt.show()

    ## c
    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    ax1.plot(x, y, 'g-')
    ax2.plot(x, h_b, 'b-')
    ax1.set_xlabel('Epsilon')
    ax1.set_ylabel('Empirical Probability', color='g')
    ax2.set_ylabel('Hoeffding bound', color='b')
    plt.title("Q1.c")
    plt.show()

if __name__ == "__main__":
    main()