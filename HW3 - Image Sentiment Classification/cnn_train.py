#!/usr/bin/env python
# coding: utf-8

# In[ ]:

"""
ML2018 hw3
b05705018@ntu.edu.tw
"""
import sys
import csv
import math
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential, load_model 
from keras.layers import Dense, Activation, Dropout, Flatten, Conv2D, MaxPooling2D, LeakyReLU
from keras.layers import BatchNormalization
from keras.utils import np_utils
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator


# In[ ]:


#load training data
train_file = open(sys.argv[1], 'r')
row = csv.reader(train_file, delimiter=" ")
row = list(row)

x = []
y = []
n_row = 0
for r in row:
    if n_row != 0:
        text = r[0].split(',')
        y.append(text[0])
        x.append(text[1])
        for i in range(1,len(r)):
            x.append(r[i])     
    n_row += 1
train_file.close()

x = np.array(x)
y = np.array(y)
x = x.reshape(n_row-1, 48, 48, 1)
x = x.astype('float32')
y = y.astype('float32')

#normalization
x = x / 255

#one-hot encoding
y = np_utils.to_categorical(y, num_classes=7)

#preprocessing
valid_num = 3000
train_x, valid_x = x[:-valid_num], x[-valid_num:]
train_y, valid_y = y[:-valid_num], y[-valid_num:]


# In[ ]:


#construct model
model = Sequential()

model.add(Conv2D(64, (3, 3), padding='same', input_shape=(48,48,1)))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(Dropout(0.2))

model.add(Conv2D(128, (3, 3), padding='same'))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
model.add(Dropout(0.25))

model.add(Conv2D(256, (3, 3), padding='same'))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
model.add(Dropout(0.3))

model.add(Conv2D(512, (3, 3), padding='same'))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
model.add(Dropout(0.35))

model.add(Flatten())

model.add(Dense(256, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))

model.add(Dense(256, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))

model.add(Dense(7))
model.add(Activation('softmax'))

model.summary()

model.compile(loss='categorical_crossentropy', 
              optimizer='adam', 
              metrics=['accuracy'])


# In[ ]:


#generate more images
datagen = ImageDataGenerator( 
    rotation_range=10, 
    width_shift_range=0.1, 
    height_shift_range=0.1,
    shear_range=0.1, 
    zoom_range=[0.9,1.1], 
    horizontal_flip=True)


# In[ ]:


callbacks = []
callbacks.append(ModelCheckpoint('model-{epoch:05d}-{val_acc:.5f}.h5', 
                                 monitor='val_acc', 
                                 save_best_only=True, 
                                 period=1))


# In[ ]:


#find the optimal network parameters
batch_size = 128
epochs = 200

history = model.fit_generator(datagen.flow(train_x, train_y, batch_size=batch_size), 
                              epochs=epochs,
                              validation_data=(valid_x, valid_y),
                              steps_per_epoch = 5*train_x.shape[0]//batch_size,
                              callbacks=callbacks)


# # In[ ]:


# # Plot training & validation accuracy values
# plt.plot(history.history['acc'])
# plt.plot(history.history['val_acc'])
# plt.title('Model accuracy')
# plt.ylabel('Accuracy')
# plt.xlabel('Epoch')
# plt.legend(['Train', 'Test'], loc='upper left')
# plt.savefig('accuracy.png')


# # In[ ]:


# # Plot training & validation loss values
# plt.plot(history.history['loss'])
# plt.plot(history.history['val_loss'])
# plt.title('Model loss')
# plt.ylabel('Loss')
# plt.xlabel('Epoch')
# plt.legend(['Train', 'Test'], loc='upper left')
# plt.savefig('loss.png')

