import csv
import sys
import math
import numpy as np


def load_train_x(train_x_path):
    """Reads in features of training data"""
    
    text = open(train_x_path, 'r')
    row = csv.reader(text , delimiter=",")
    x = []
    n_row = 0
    for r in row:
        if n_row != 0:
            for j in range(23):
                x.append(float(r[j]))
        n_row += 1
    text.close()
    x = np.array(x)
    x = np.reshape(x, (20000,23))
    
    return x


def load_train_y(train_y_path):
    """Reads in labels of training data"""
    
    text = open(train_y_path, 'r')
    row = csv.reader(text)
    y = []
    n_row = 0
    for r in row:
        if n_row != 0:
            y.append(float(r[0]))
        n_row += 1
    text.close()
    y = np.array(y)
    
    return y


def load_test_x(test_x_path):
    """Reads in features of testing data"""

    text = open('data/test_x.csv', 'r')
    row = csv.reader(text , delimiter=",")
    x_test = []
    n_row = 0
    for r in row:
        if n_row != 0:
            for j in range(23):
                x_test.append(float(r[j]))
        n_row += 1
    text.close()
    x_test = np.array(x_test)
    x_test = np.reshape(x_test, (10000,23))
    
    return x_test