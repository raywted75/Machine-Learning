import csv
import sys
import os

import re
import pandas as pd
from io import StringIO

import jieba
import multiprocessing
import numpy as np

from gensim.models import Word2Vec

from keras.layers import Embedding, Dense, LSTM, GRU, Activation, Dropout, BatchNormalization, Bidirectional, LeakyReLU
from keras.models import Sequential, load_model
from keras.optimizers import RMSprop
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

maxlen = 128

def preprocess(s):
    s = re.sub(r',', chr(23), s.rstrip(), count=1)
    s = re.sub(r' +', ',', s)
    s = re.sub(r'b[0-9]+', '<SB>', s) 
    s = re.sub(r'B[0-9]+', '<SB>', s)   
    return s

def line_to_token(df):
    tokenized = []
    for line in df:
        tokens = jieba.cut(line, cut_all=False)
        tokenized.append(list(tokens))
    return tokenized

def make_dictionary(model):
    word2idx = {}
    word2idx['<pad>'] = 0
    word2idx['<unk>'] = 1
    for i, word in enumerate(model.wv.index2word):
        word2idx[word] = i+2

    idx2word = {}
    idx2word[0] = '<pad>'
    idx2word[1] = '<unk>'
    for i, word in enumerate(model.wv.index2word):
        idx2word[i+2] = word
        
    return word2idx, idx2word

def encode(tokenized_samples, word2idx, UNK=1):
    features = []
    for sample in tokenized_samples:
        feature = []
        for token in sample:
            if token in word2idx:
                feature.append(word2idx[token])
            else:
                feature.append(UNK)
        features.append(feature)
    return features

def pad(features, padlen=100, PAD=0):
    padded_features = []
    for feature in features:
        if len(feature) >= padlen:
            padded_feature = feature[:padlen]
        else:
            padded_feature = feature
            while(len(padded_feature) < padlen):
                padded_feature.append(PAD)
        padded_features.append(padded_feature)
    return padded_features


test_x_path = sys.argv[1]
dict_path = sys.argv[2]
output_path = sys.argv[3]


#read in testing data
test_x_io = StringIO()
with open(test_x_path, encoding="utf-8") as file:
    for line in file:
        line = preprocess(line)
        print(line, file=test_x_io)
        
test_x_io.seek(0)
test_df = pd.read_csv(test_x_io, sep=chr(23))


#word segmentation
jieba.load_userdict(dict_path)
jieba.enable_parallel(multiprocessing.cpu_count())

tokenized_test = line_to_token(test_df.dropna()['comment'].values)


#word2vec
w2v_model = Word2Vec.load("word2vec.model")
word2idx, idx2word = make_dictionary(w2v_model)


#encoding and padding
test_features = encode(tokenized_test, word2idx)
padded_test_features = pad(test_features, padlen=maxlen)
test_x = np.array(padded_test_features)


#testing file prediction
model = load_model('ckpt.hdf5')
pred_y = model.predict(test_x, batch_size=512, verbose=1)

test_na = test_df['comment'].isna()
ans = []
i = 0
j = 0
for is_na in test_na:
    ans.append([])
    if is_na:
        ans[i].append(i)
        ans[i].append(0)
    else:
        if pred_y[j][0] >= 0.5:
            ans[i].append(i)
            ans[i].append(0)
        else:
            ans[i].append(i)
            ans[i].append(1)
        j += 1
    i += 1
    
text = open(output_path, "w+")
s = csv.writer(text, delimiter=',')
s.writerow(["id","label"])
for i in range(len(ans)):
    s.writerow(ans[i])
text.close()