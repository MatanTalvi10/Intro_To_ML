import numpy as np
import matplotlib.pyplot as plt
import math
import sys
from functions import *
from sklearn.datasets import fetch_openml
from statistics import mode


mnist = fetch_openml('mnist_784', as_frame=False)
data = mnist['data']
labels = mnist['target']
idx = np.random.RandomState(0).choice(70000, 11000)
train = data[idx[:10000], :].astype(int)
train_labels = labels[idx[:10000]]
test = data[idx[10000:], :].astype(int)
test_labels = labels[idx[10000:]]

def main():
    Q2b()
    Q2c()
    Q2d()

# Q2a
def knn(train_img: np.array,train_labels: np.array, q_image: np.array,k: int) -> int:
    if k > train_img.shape[0]:
        raise ValueError('Invalid k value')
        sys.exit()
    else:
        dist = []
        k_labels = []
        for i in range(len(train_img)):
            d = np.linalg.norm(train_img[i]-q_image)
            dist.append((d,train_labels[i]))
        dist_sorted_k = (sorted(dist, key=lambda item: item[0]))[:k]
        for item in dist_sorted_k:
            k_labels.append(item[1])
        return mode(k_labels)

def test_knn(n: int,k: int) -> float:
    input_img = train[:n]
    input_labels = train_labels[:n]
    wrong_cnt = 0
    for i in range(len(test)):
        res = knn(input_img,input_labels,test[i],k)
        if res != test_labels[i]:
            wrong_cnt += 1
    acc = 1 - wrong_cnt/len(test_labels)
    return acc

def Q2b():
    acc_res = test_knn(1000,10)

def Q2c():
    k_list = []
    acc_k = []
    for k_val in range(1,101):
        k_list.append(k_val)
        acc_k.append(test_knn(1000,k_val))
    plt.title("Q2.c")
    plt.xlabel("k values")
    plt.ylabel("accuracy")
    plt.plot(k_list, acc_k, color ="blue")
    plt.show()

def Q2d():
    n_list = []
    acc_n = []
    for n_val in range(100, 5001, 100):
        n_list.append(n_val)
        acc_n.append(test_knn(n_val,1))
    plt.title("Q2.d")
    plt.xlabel("n values")
    plt.ylabel("accuracy")
    plt.plot(n_list, acc_n, color ="blue")
    plt.show()

if __name__ == "__main__":
    main()
