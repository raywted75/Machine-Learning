# Credit Card Default Payment Prediction
* Implement **generative model** and **logistic regression** to predict the default payment status of credit cards. <br>
* Ranking: 61/109 <br>
* [Kaggle Link](https://www.kaggle.com/c/ml2018fall-hw2/leaderboard) <br>
* [Assignment Slides](https://drive.google.com/open?id=1vFNiRoXd4_fD6WpflQhiafHbCoB9ZY83) <br>
* [Data Link](https://drive.google.com/open?id=1aqz7e-97srh06UR4DTdCVHdUQ8IX0HU_) <br>

## Probabilistic Generative Model
Usage
```
$ python3 probabilistic.py [TRAINING_FEATURE_PATH] [TRAINING_LABEL_PATH] [TESTING_FEATURE_PATH] [PREDICTION_FILE_PATH]
```
Example
```
$ python3 probabilistic.py ./data/train_x.csv ./data/train_y.csv ./data/test_x.csv ./result/prediction.csv
```

## Logistic Regression
Usage
```
$ python3 logistic.py [TRAINING_FEATURE_PATH] [TRAINING_LABEL_PATH] [TESTING_FEATURE_PATH] [PREDICTION_FILE_PATH]
```
Example
```
$ python3 logistic.py ./data/train_x.csv ./data/train_y.csv ./data/test_x.csv ./result/prediction.csv
```
