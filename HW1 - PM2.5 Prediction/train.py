import sys
import csv
import math
import time
import numpy as np
import human_adjust


def load_train(train_path):
    """Reads in the training data"""
            
    data = []
    for i in range(18):
        data.append([])
    
    file = open(train_path, 'r', encoding='big5') 
    rows = csv.reader(file , delimiter=',')
    row_cnt = 0
    for row in rows:
        if row_cnt != 0:
            for i in range(3,27):
                if row[i] != "NR" and float(row[i]) > 0:
                    data[(row_cnt-1)%18].append(float(row[i]))
                else:
                    data[(row_cnt-1)%18].append(float(0)) 
        row_cnt = row_cnt + 1
    file.close()
    
    return data


def load_test(test_path):
    """Reads in the testing data"""
    
    test_x = []
    n_row = 0
    text = open(test_path, 'r')
    row = csv.reader(text , delimiter=',')
    for r in row:
        if n_row %18 == 0:
            test_x.append([])
            for i in range(2,11):
                test_x[n_row//18].append(float(r[i]) )
        else :
            for i in range(2,11):
                if r[i] != "NR":
                    test_x[n_row//18].append(float(r[i]))
                else:
                    test_x[n_row//18].append(0)
        n_row = n_row + 1
    text.close()
    test_x = np.array(test_x)
    
    return test_x


def preprocess(data, type=''):
    """Preprocesses the data and makes it trainable or predictable"""
    
    if type == 'Train':
        # Deals with the unnormal values in the data by human-adjustment
        r = human_adjust.row
        c = human_adjust.column
        v = human_adjust.value
        for i in range(len(r)):
            data[r[i]][c[i]] = v[i]

        # Corrects the invalid value
        for i in range(len(data[8])):
            if data[8][i] == 0 or data[8][i] > 200:
                data[8][i] = (data[8][i-1] + data[8][i+1]) / 2
        for i in range(len(data[9])):
            if data[9][i] == 0 or data[9][i] > 150:
                data[9][i] = (data[9][i-1] + data[9][i+1]) / 2

        # Extracts features and labels from the data
        # Takes continuous 9 hours data as a unit
        x = []
        y = []
        for i in range(12):
            for j in range(471):
                x.append([])
                for t in range(18):
                    for s in range(9):
                        x[471*i+j].append(data[t][480*i+j+s] )
                y.append(data[9][480*i+j+9])
        x = np.array(x)
        y = np.array(y)
        return x, y
    
    elif type == 'Test':
        # Corrects the invalid value
        for i in range(260):
            for j in range(72, 81):
                if data[i][j] == 0 or data[i][j] > 200:
                    if j != 80:
                        data[i][j] = data[i][80]
                    else:
                        data[i][j] = data[i][79]
            for k in range(81, 90):
                if data[i][k] == 0 or data[i][k] > 150:
                    if k != 89:
                        data[i][k] = data[i][89]
                    else:
                        data[i][k] = data[i][88]
        return data
    
    else:
        print("Please assign the data type.")
        

def training(x, y, l_rate=0.1, repeat=500000, nap=10):
    """Trains the data using linear regression with AdaGrad"""
    
    # Adds bias
    x = np.concatenate((np.ones((x.shape[0],1)),x), axis=1)
    x_t = x.transpose()

    # Training
    w = np.zeros(len(x[0]))
    x_t = x.transpose()
    s_gra = np.zeros(len(x[0]))

    for i in range(repeat):
        hypo = np.dot(x,w)
        loss = hypo - y
        mse = np.sum(loss**2) / len(x)
        rmse  = math.sqrt(mse)
        gra = np.dot(x_t,loss)
        s_gra += gra**2
        ada = np.sqrt(s_gra)
        w = w - l_rate * gra/ada
        if i % 10000 == 0:
            print ('iteration: %d | RMSE: %f  ' % (i, rmse))
        if i != 0 and i % 100000 == 0:
            time.sleep(nap) 

    # Saves model
    np.save('model.npy', w)
    
    
def predict(data, output_path):
    """Predicts PM2.5 by the model"""
    
    # Reads model
    w = np.load('model.npy')

    # Adds bias
    data = np.concatenate((np.ones((data.shape[0],1)), data), axis=1)

    # Calculates prediction
    prediction = []
    for i in range(len(data)):
        prediction.append(["id_"+str(i)])
        value = np.dot(w, data[i])
        prediction[i].append(value)

    # Saves predict ans
    text = open(output_path, "w+")
    s = csv.writer(text,delimiter=',',lineterminator='\n')
    s.writerow(["id","value"])
    for i in range(len(prediction)):
        s.writerow(prediction[i]) 
    text.close()


def main():
    """Main function of the training."""

    train_path = sys.argv[1]
    test_path = sys.argv[2]
    output_path = sys,argv[3]

    train = load_train(train_path)
    train_features, train_labels = preprocess(train, type='Train')    
    training(train_features, train_labels, l_rate=0.1, repeat=500000, nap=10)
      
    test = load_test(test_path)
    test_features = preprocess(test, type='Test')   
    predict(test_features, output_path)


if __name__ == "__main__":
    main()