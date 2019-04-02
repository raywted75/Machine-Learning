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


#hyperparameter
embed_size = 256
maxlen = 128
batch_size = 400
epochs = 20
validation_split= 0.1


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


train_x_path = sys.argv[1]
train_y_path = sys.argv[2]
test_x_path = sys.argv[3]
dict_path = sys.argv[4]

#read in training data
train_x_io = StringIO()
with open(train_x_path, encoding="utf-8") as file:
    for line in file:
        line = preprocess(line)
        print(line, file=train_x_io)
        
train_x_io.seek(0)
train_x_df = pd.read_csv(train_x_io, sep=chr(23))

train_y_df = pd.read_csv(train_y_path)
train_df = pd.merge(train_x_df, train_y_df).dropna()

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

tokenized_train = line_to_token(train_df['comment'].values)
tokenized_test = line_to_token(test_df.dropna()['comment'].values)
tokenized = np.concatenate((tokenized_train, tokenized_test), axis=0)
np.save('tokenized.npy', tokenized)



#word2vec
w2v_model = Word2Vec(tokenized, 
                     size=embed_size,
                     iter=30, 
                     min_count=1,
                     workers=multiprocessing.cpu_count())
w2v_model.save("word2vec.model")
word2idx, idx2word = make_dictionary(w2v_model)



#encoding and padding
train_features = encode(tokenized_train, word2idx)
padded_train_features = pad(train_features, padlen=maxlen)
train_x = np.array(padded_train_features)
train_y = train_df['label'].values
train_y = pd.get_dummies(train_y).values

test_features = encode(tokenized_test, word2idx)
padded_test_features = pad(test_features, padlen=maxlen)
test_x = np.array(padded_test_features)



#make embedding matrix
vocab = []
for k, v in w2v_model.wv.vocab.items():
    vocab.append([k, w2v_model.wv[k]])

embedding_matrix = np.zeros((len(idx2word), w2v_model.vector_size))
for word, vec in vocab:
    index = word2idx[word]
    embedding_matrix[index] = vec



# build model
embedding_layer = Embedding(input_dim=len(embedding_matrix),
                            output_dim=embed_size,
                            weights=[embedding_matrix],
                            input_length=maxlen)
embedding_layer.trainable = False

model = Sequential()
model.add(embedding_layer)

model.add(Bidirectional(LSTM(512, dropout=0.5, recurrent_dropout=0.5, return_sequences=True)))
model.add(LeakyReLU())

model.add(Bidirectional(LSTM(512, dropout=0.5, recurrent_dropout=0.5)))
model.add(LeakyReLU())

model.add(BatchNormalization())

model.add(Dense(512))
model.add(LeakyReLU())
model.add(Dropout(0.5))

model.add(BatchNormalization())

model.add(Dense(512))
model.add(LeakyReLU())
model.add(Dropout(0.5))

model.add(BatchNormalization())

model.add(Dense(2))
model.add(Activation('softmax'))
model.compile(loss="categorical_crossentropy", optimizer=RMSprop(lr=0.001), metrics=['accuracy'])
model.summary()



#ckpt
model_checkpoint = ModelCheckpoint('ckpt.hdf5', monitor='val_acc', save_best_only=True, verbose=0, mode='auto', period=1)
early_stopping = EarlyStopping(monitor='val_loss', min_delta=0.01, patience=3, verbose=1, mode='auto')
reduce_LR_on_plateau = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=1, verbose=1, mode='auto', min_lr=0.0000000001)

callbacks = [model_checkpoint, early_stopping, reduce_LR_on_plateau]


#training
history = model.fit(train_x, train_y, 
                    batch_size=batch_size, 
                    epochs=epochs, 
                    validation_split=validation_split, 
                    callbacks=callbacks)


# import matplotlib.pyplot as plt
# # Plot training & validation accuracy values
# plt.plot(history.history['acc'])
# plt.plot(history.history['val_acc'])
# plt.title('Model accuracy')
# plt.ylabel('Accuracy')
# plt.xlabel('Epoch')
# plt.legend(['Train', 'Test'], loc='upper left')
# plt.savefig('acc.png')

# # Plot training & validation loss values
# plt.plot(history.history['loss'])
# plt.plot(history.history['val_loss'])
# plt.title('Model loss')
# plt.ylabel('Loss')
# plt.xlabel('Epoch')
# plt.legend(['Train', 'Test'], loc='upper left')
# plt.savefig('loss.png')


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
    
text = open('predict.csv', "w+")
s = csv.writer(text, delimiter=',')
s.writerow(["id","label"])
for i in range(len(ans)):
    s.writerow(ans[i])
text.close()