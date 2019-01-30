import csv
import sys
import math
import numpy as np
from utils import *


def gaussion(mu, sigma, x_i):
    """Gaussian distribution probability density function"""
    
    d = sigma.shape[0] #dimension of sigma
    pi = math.pi
    det = np.linalg.det(sigma) #determinant of sigma
    diff = x_i - mu
    diff_t = np.transpose(diff)
    sigma_inv = np.linalg.inv(sigma)
    maha = np.dot(np.dot(diff, sigma_inv), diff_t)

    a = 1 / (2*pi)**(d/2)
    b = 1 / det**(1/2)
    c = (-1/2) * maha

    gs = a * b * math.exp(c)
    return gs


def mean_mu(x):
    """Calculates mean of each feature of x and return mu """
    
    mu = np.mean(x, axis=0)
    return mu


def covar_matrix(x, mu):
    """Calculates covariance matrix"""
    
    s = np.zeros((x.shape[1],x.shape[1]))
    for i in range(x.shape[0]):
        dis = x[i] - mu
        diff = dis.reshape(1, x.shape[1])
        diff_t = np.transpose(diff)
        porduct = np.dot(diff_t, diff)
        s += porduct
    cm = s / x.shape[0]
    return cm


def prior(y, clas):
    """Calculates probability of getting clas from y"""
    
    s = 0
    for i in range(len(y)):
        if y[i] == clas:
            s += 1
    p = s / len(y)
    return p


def param(x, y):
    """Calculates probability values including mean, covariance matrix, 
    and prior probability"""
    
    x_c1 = []
    x_c2 = []
    c1_cnt = 0
    c2_cnt = 0
    for i in range(x.shape[0]):
        if y[i] == 1:
            x_c1.append([])
            x_c1[c1_cnt] = x[i]
            c1_cnt += 1
        elif y[i] == 0:
            x_c2.append([])
            x_c2[c2_cnt] = x[i]
            c2_cnt += 1
    x_c1 = np.array(x_c1)
    x_c2 = np.array(x_c2)

    mu1 = mean_mu(x_c1)
    sigma1 = covar_matrix(x_c1, mu1)
    prior1 = c1_cnt / (c1_cnt+c2_cnt)

    mu2 = mean_mu(x_c2)
    sigma2 = covar_matrix(x_c2, mu2)
    prior2 = c2_cnt / (c1_cnt+c2_cnt)

    sigma = prior1 * sigma1 + prior2 * sigma2

    return mu1, mu2, sigma, prior1, prior2


def predict(x_p, mu1, mu2, sigma, prior1, prior2):
    """Predicts the payment status"""
    
    p1 = gaussion(mu1, sigma, x_p) * prior1
    p2 = gaussion(mu2, sigma, x_p) * prior2

    if p1 > p2:
        return 1
    else:
        return 0


def cross_valid(x, y, split=4):
    """Creates x_c, y_c for calculating param and 
    x_p, y_p for testing prediction accuracy"""
    
    x_c = []
    x_p = []
    y_c = []
    y_p = []
    for i in range(split):
        x_i = []
        y_i = []
        cnt = 0
        for j in range(x.shape[0]):
            if j % split == i:
                x_i.append([])
                x_i[cnt] = x[j]
                y_i.append([])
                y_i[cnt] = y[j]
                cnt += 1
        x_i = np.array(x_i)
        y_i = np.array(y_i)
        
        if i == 0:
            x_c = x_i
            y_c = y_i
        elif i != 0 and i != (split-1):
            x_c =  np.concatenate((x_c, x_i))
            y_c =  np.concatenate((y_c, y_i))
        else:
            x_p = x_i
            y_p = y_i
            
    return x_c, x_p, y_c, y_p


def check_accuracy(x_p, y_p, mu1, mu2, sigma, prior1, prior2):
    """Check accuracy on the training data"""
    
    correct_cnt = 0
    for i in range(x_p.shape[0]):
        p_result = predict(x_p[i], mu1, mu2, sigma, prior1, prior2)
        if p_result == y_p[i]:
            correct_cnt += 1
    accuracy = correct_cnt / x_p.shape[0]
    print(accuracy)
    

def generate_ans(test_x, output_path, mu1, mu2, sigma, prior1, prior2):
    """Generates and saves the prediction"""
    
    ans = []
    for i in range(10000):
        ans.append(["id_"+str(i)])
        p_result = predict(test_x[i], mu1, mu2, sigma, prior1, prior2)
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

    # Checks the accuracy on the training data
    x_c, x_p, y_c, y_p = cross_valid(train_x, train_y, split=4)
    mu1, mu2, sigma, prior1, prior2 = param(x_c, y_c)
    check_accuracy(x_p, y_p, mu1, mu2, sigma, prior1, prior2)

    # Calculates prediction answers
    mu1, mu2, sigma, prior1, prior2 = param(train_x, train_y)
    generate_ans(test_x, output_path, mu1, mu2, sigma, prior1, prior2)


if __name__ == "__main__":
    main()

