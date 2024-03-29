#################################
# Your name: Matan Talvi
#################################


import numpy as np
import numpy.random
from sklearn.datasets import fetch_openml
import sklearn.preprocessing
from numpy import linalg as l2
import matplotlib.pyplot as plt
import scipy

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
    sample_size = data.shape[0]
    sample_info = data.shape[1]
    w = np.zeros(sample_info)
    for t in range(1, T + 1):
        w = w.reshape(sample_info, )
        i = np.random.randint(0, sample_size, 1)[0]
        gradient_w_fi = log_loss_gradient_calc(w, data[i], labels[i])
        w = np.add(w, (eta_0 / t) * (np.dot(gradient_w_fi, data[i])))
    return w

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

def q2_a():
    T = 1000
    C = 1
    etas = [10 ** i for i in range(-5, 4)] 
    etas_accuracy = []
    for eta0 in etas:
        sum_acc = 0
        for i in range(10):
            w = SGD_log(train_data, train_labels,eta0, 1000)
            sum_acc += accuracy_calc(validation_data, validation_labels, w)
        etas_accuracy.append(sum_acc/10)
        plt.title("accuracy of SGD_Log loss per eta_0")
    plt.xlabel('eta_0')
    plt.ylabel('average acc')
    plt.xscale('log')
    plt.plot(etas, etas_accuracy, marker='o')
    plt.show()
    return etas[np.argmax(etas_accuracy)]

def q2_b(train_data, train_labels, eta_0):
    T = 20000
    w = SGD_log(train_data, train_labels, eta_0, T)
    plt.imshow(np.reshape(w, (28, 28)), interpolation='nearest')
    plt.show()
    return accuracy_calc(train_data, train_labels, w)

def q2_c(train_data, train_labels, eta_0):
    T = 20000
    w_norms = norm_iterations(train_data, train_labels, eta_0, T)
    iterates = np.arange(1, 20001)
    plt.plot(iterates, w_norms)
    plt.xlabel('iteration')
    plt.ylabel('norm(w)')
    plt.show()

def norm_iterations(data, labels, eta_0, T):
    """
    Calculate w norm for each of T iterates
    """
    w = np.zeros(784)  
    w_norms = np.zeros(T+1)
    for t in range(1, T + 1):
        i = np.random.randint(data.shape[0]) 
        y = labels[i]
        x = data[i]
        exp = (-y * x) * scipy.special.softmax(-y * np.dot(w, x))
        eta = eta_0 / t
        w = w - (eta * exp)
        w_norms[t] = np.linalg.norm(w)
    return w_norms[1:]


def log_loss_gradient_calc(w, x, y):
    exp = scipy.special.softmax(-y * np.dot(w, x))
    return (exp * ((-y)) / (1 + exp)) * x


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
        

if __name__ == '__main__':
    train_data, train_labels, validation_data, validation_labels, test_data, test_labels = helper()
    best_eta = q1_a()
    best_c = q1_b(best_eta)
    q1_c(best_eta,best_c)
    our_accuracy = q1_d(best_eta,best_c)
    best_eta_2 = q2_a()
    q2_b(train_data,train_labels,best_eta_2)
    q2_c(train_data,train_labels,1e-05)
    
