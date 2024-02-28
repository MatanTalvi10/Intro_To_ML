#################################
# Your name: Matan Talvi
#################################


import numpy as np
import numpy.random
from sklearn.datasets import fetch_openml
import sklearn.preprocessing
from numpy import linalg as l2
import matplotlib.pyplot as plt

"""
Please use the provided function signature for the SGD implementation.
Feel free to add functions and other code, and submit this file with the name sgd.py
"""


def helper():
    mnist = fetch_openml('mnist_784', as_frame=False)
    data = mnist['data']
    labels = mnist['target']

    neg, pos = "0", "8"
    train_idx = numpy.random.RandomState(0).permutation(np.where((labels[:60000] == neg) | (labels[:60000] == pos))[0])
    test_idx = numpy.random.RandomState(0).permutation(np.where((labels[60000:] == neg) | (labels[60000:] == pos))[0])

    train_data_unscaled = data[train_idx[:6000], :].astype(float)
    train_labels = (labels[train_idx[:6000]] == pos) * 2 - 1

    validation_data_unscaled = data[train_idx[6000:], :].astype(float)
    validation_labels = (labels[train_idx[6000:]] == pos) * 2 - 1

    test_data_unscaled = data[60000 + test_idx, :].astype(float)
    test_labels = (labels[60000 + test_idx] == pos) * 2 - 1

    # Preprocessing
    train_data = sklearn.preprocessing.scale(train_data_unscaled, axis=0, with_std=False)
    validation_data = sklearn.preprocessing.scale(validation_data_unscaled, axis=0, with_std=False)
    test_data = sklearn.preprocessing.scale(test_data_unscaled, axis=0, with_std=False)
    return train_data, train_labels, validation_data, validation_labels, test_data, test_labels



def SGD_hinge(data, labels, C, eta_0, T):
    """
    Implements SGD for hinge loss.
    """
    sample_size = data.shape[0]
    sample_info = data.shape[1]
    w = np.zeros(sample_info)
    for t in range(1, T + 1):
        i = np.random.randint(0, sample_size, 1)[0]
        eta_t = eta_0 / t
        x_i, y_i = data[i], labels[i]
        if (y_i * (w @ x_i)) < 1:
            w = (1 - eta_t) * w + eta_t * C * y_i * x_i
        else:
            w = (1 - eta_t) * w
    return w




def SGD_log(data, labels, eta_0, T):
    """
    Implements SGD for log loss.
    """
    # TODO: Implement me
    pass

#################################

# Place for additional code

#################################

def q1_a():
    T = 1000
    C = 1
    etas = [10 ** i for i in range(-5, 4)] 
    etas_accuracy = []
    for eta0 in etas:
        sum_acc = 0
        for i in range(10):
            w = SGD_hinge(train_data, train_labels, 1, eta0, 1000)
            sum_acc += accuracy_calc(validation_data, validation_labels, w)
        etas_accuracy.append(sum_acc/10)
    plt.title("accuracy - SGD_Hinge loss per eta_0")
    plt.xlabel('eta_0')
    plt.ylabel('average accuracy')
    plt.xscale('log')
    plt.plot(etas, etas_accuracy, marker='o')
    plt.show()
    return etas[np.argmax(etas_accuracy)]

def q1_b(eta0):
    T = 1000
    c_list = [10 ** i for i in range(-5, 4)] 
    c_accuracy = []
    for c in c_list:
        sum_acc = 0
        for i in range(10):
            w = SGD_hinge(train_data, train_labels, c, eta0, 1000)
            sum_acc += accuracy_calc(validation_data, validation_labels, w)
        c_accuracy.append(sum_acc/10)
    print(c_list[np.argmax(c_accuracy)])
    plt.title("acc of SGD_Hinge loss per C")
    plt.xlabel('C')
    plt.ylabel('average acc')
    plt.xscale('log')
    plt.plot(c_list, c_accuracy, marker='o')
    plt.show()
    return c_list[np.argmax(c_accuracy)]

def q1_c(eta_0, c):
    w = SGD_hinge(train_data, train_labels, c, eta_0, 20000)
    plt.imshow(np.reshape(w, (28, 28)), interpolation='nearest')
    plt.show()

def q1_d(eta_0, C):
    T = 20000
    w = SGD_hinge(train_data, train_labels, C, eta_0, T)
    return accuracy_calc(test_data, test_labels, w)


def hinge_loss_LReg(w,x,y,C=1):
    """Computes the Hinge loss withL2-regularization."""
    in_prod = np.inner(w,x)
    norm_2 = (l2.norm(w))**2
    res = 0
    tmp = 1-y*in_prod + 0.5*(norm_2)
    if tmp > res:
        res = C*tmp
    return res

def accuracy_calc(data,labels,w):
    """Computes the accuracy (correct labales/total data) given a vector w."""
    size_data = data.shape[0]
    cnt = 0
    for i in range(size_data):
        x_i, y_i = data[i], labels[i]
        y_pred = 1 if x_i @ w >= 0 else -1
        if y_i == y_pred:
            cnt += 1
    return cnt/size_data 
        


train_data, train_labels, validation_data, validation_labels, test_data, test_labels = helper()
best_eta = q1_a()
best_c = q1_b(best_eta)
q1_c(best_eta,best_c)
our_accuracy = q1_d(best_eta,best_c)