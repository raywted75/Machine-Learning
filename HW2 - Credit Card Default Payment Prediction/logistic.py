import os
import csv
import sys
import math
import time
import numpy as np
from utils import *


def sigmoid(x):
    """Calculates the result of sigmoid function"""

    res = 1.0/(1.0+np.exp(-1*x))
    return res


def feature_scaling(x):
    """Does feature scaling"""

    mean = np.mean(x, axis=0)
    std = np.std(x, axis=0)
    x = (x - mean) / std
    return x


def train(train_x, train_y, lr = 1, repeat = 100000, loss_min = 1):
    """Trains the data using logistic regression"""

    ###define variables
    #weight
    w_1 = np.zeros((1,len(train_x[0])))
    w_2 = np.zeros((1,len(train_x[0])))
    w_3 = np.zeros((1,len(train_x[0])))
    w_4 = np.zeros((1,len(train_x[0])))
    w_layer1 = np.concatenate((w_1,w_2), axis=0)
    w_layer1 = np.concatenate((w_layer1,w_3), axis=0)
    w_layer1 = np.concatenate((w_layer1,w_4), axis=0)
    w_layer2 = np.zeros((1,4))

    #bias
    b_layer1 = np.array([[0],[0],[0],[0]])
    b_layer2 = np.array([[0]])

    ###training model
    grad_w1 = 1
    grad_w2 = 1
    for i in range(repeat):

        ##forward
        #layer1
        z_layer1 = np.dot(w_layer1, train_x.T) + b_layer1
        a = sigmoid(z_layer1)

        #layer2
        z_layer2 = np.dot(w_layer2, a) + b_layer2
        y = sigmoid(z_layer2)


        ##backward
        #layer2
        diff_y = y - train_y.T
        w_grad_layer2 = np.dot(diff_y, a.T) / train_x.shape[0]
        b_grad_layer2 = np.mean(diff_y)
        #adagrad
        grad_w2 += w_grad_layer2**2
        sgrad_w2 = np.sqrt(grad_w2) 

        #layer1
        a_deri = a.T - a.T**2
        z_grad = diff_y.T * (w_layer2 * a_deri)
        w_grad_layer1 = np.dot(train_x.T, z_grad) / train_x.shape[0]
        b_grad_layer1 = np.mean(z_grad, axis=0)
        b_grad_layer1 = np.reshape(b_grad_layer1, (4,1))
        #adagrad
        grad_w1 += w_grad_layer1**2
        sgrad_w1 = np.sqrt(grad_w1) 


        ##loss function
        abs_y = np.absolute(diff_y)
        loss = np.mean(abs_y)

        if loss < loss_min:
            loss_min = loss
        else:
            print('over the min point, loss = ', loss)
            break


        ##gradient descent
        #layer2
        w_layer2 = w_layer2 - lr * w_grad_layer2  / sgrad_w2
        b_layer2 = b_layer2 - lr * b_grad_layer2

        #layer1
        w_layer1 = w_layer1 - lr * w_grad_layer1.T  / sgrad_w1.T
        b_layer1 = b_layer1 - lr * b_grad_layer1 


        #print loss and save model
        if i % 10000 == 0 or i == repeat-1:
            print ('i =', i, ' | loss = ', loss)
            np.save('model/model_w1.npy', w_layer1)
            np.save('model/model_w2.npy', w_layer2)
            np.save('model/model_b1.npy', b_layer1)
            np.save('model/model_b2.npy', b_layer2)
            
            
def predict(test_x, output_path):
    """Predicts and saves the payment status"""

    #read model
    w_layer1 = np.load('model/model_w1.npy')
    w_layer2 = np.load('model/model_w2.npy')
    b_layer1 = np.load('model/model_b1.npy')
    b_layer2 = np.load('model/model_b2.npy')


    #calculate predict ans
    ans = []
    for i in range(10000):
        ans.append(["id_"+str(i)])
        x_now = np.reshape(test_x[i], (23,1))

        z_layer1 = np.dot(w_layer1, x_now) + b_layer1
        a = sigmoid(z_layer1)
        z_layer2 = np.dot(w_layer2, a) + b_layer2
        y = sigmoid(z_layer2)

        p_result = 0
        if y > 0.45:
            p_result = 1
        ans[i].append(p_result)


    #save predict ans
    text = open(output_path, "w+")
    s = csv.writer(text,delimiter=',',lineterminator='\n')
    s.writerow(["id","Value"])
    for i in range(len(ans)):
        s.writerow(ans[i]) 
    text.close()


def main():
    """Main function of the training."""

    train_x_path = sys.argv[1]
    train_y_path = sys.argv[2]
    test_x_path = sys.argv[3]
    output_path = sys.argv[4]

    train_x = load_train_x(train_x_path)
    train_y = load_train_y(train_y_path)
    test_x = load_test_x(test_x_path)
    
    train_x = feature_scaling(train_x)
    test_x = feature_scaling(test_x)
    
    os.mkdir('model')
    train(train_x, train_y, lr = 1, repeat = 100000, loss_min = 1)
    
    predict(test_x, output_path)


if __name__ == "__main__":
    main()