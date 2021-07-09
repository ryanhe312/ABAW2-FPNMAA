import numpy as np
from sklearn.metrics import f1_score, confusion_matrix

"""
Evalution Metrics: F1 score, accuracy and CCC
borrow from https://github.com/wtomin/Multitask-Emotion-Recognition-with-Incomplete-Labels/ 
"""

epsilon = 1e-5


def averaged_f1_score(input, target):
    N, label_size = input.shape
    f1s = []
    for i in range(label_size):
        f1 = f1_score(input[:, i], target[:, i], zero_division=True)
        f1s.append(f1)
    return np.mean(f1s), f1s


def accuracy(input, target):
    assert len(input.shape) == 1
    return sum(input==target)/input.shape[0]


def averaged_accuracy(x, y):
    assert len(x.shape) == 2
    N, C =x.shape
    accs = []
    for i in range(C):
        acc = accuracy(x[:, i], y[:, i])
        accs.append(acc)
    return np.mean(accs), accs


def CCC_score(x, y):
    vx = x - np.mean(x)
    vy = y - np.mean(y)
    rho = np.sum(vx * vy) / (np.sqrt(np.sum(vx**2)) * np.sqrt(np.sum(vy**2)) + epsilon)
    x_m = np.mean(x)
    y_m = np.mean(y)
    x_s = np.std(x)
    y_s = np.std(y)
    ccc = 2*rho*x_s*y_s/(x_s**2 + y_s**2 + (x_m - y_m)**2 + epsilon)
    return ccc


def VA_metric(x, y):
    x = np.clip(x, -0.99, 0.99)
    items = [CCC_score(x[:,0], y[:,0]), CCC_score(x[:,1], y[:,1])]
    return items, sum(items)/2


def EXPR_metric(x, y):
    if not len(x.shape) == 1:
        if x.shape[1] == 1:
            x = x.reshape(-1)
        else:
            x = np.argmax(x, axis=-1)
    if not len(y.shape) == 1:
        if y.shape[1] == 1:
            y = y.reshape(-1)
        else:
            y = np.argmax(y, axis=-1)

    f1 = f1_score(x, y, average='macro')
    acc = accuracy(x, y)
    matrix = confusion_matrix(x,y)
    return [f1, acc], 0.67*f1 + 0.33*acc, matrix


def AU_metric(x, y):
    x = (x > 0.5).astype(int)
    y = y.clip(0, 1).astype(int)
    f1_av, _ = averaged_f1_score(x, y)
    x = x.reshape(-1)
    y = y.reshape(-1)
    acc_av = accuracy(x, y)
    return [f1_av, acc_av], 0.5*f1_av + 0.5*acc_av