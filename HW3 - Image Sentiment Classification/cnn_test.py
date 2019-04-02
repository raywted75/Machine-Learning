import sys
import csv
import math
import numpy as np
from keras.models import Sequential, load_model 
from keras.layers import Dense, Activation, Dropout, Flatten, Conv2D, MaxPooling2D, LeakyReLU, AveragePooling2D
from keras.layers import BatchNormalization
from keras.utils import np_utils
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator


# In[2]:


#load testing data
print('start loading testing data')

test_file = open(sys.argv[1], 'r')
row = csv.reader(test_file, delimiter=" ")
row = list(row)

test_x = []
n_row = 0
for r in row:
    if n_row != 0:
        text = r[0].split(',')
        test_x.append(text[1])
        for i in range(1,len(r)):
            test_x.append(r[i])     
    n_row += 1
test_file.close()

test_x = np.array(test_x)
test_x = test_x.reshape(n_row-1, 48, 48, 1)
test_x = test_x.astype('float32')
test_x = test_x / 255

print('done!')
# In[22]:


# Construct the my model
print('start my prediction')

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


model.compile(loss='categorical_crossentropy', 
              optimizer='adam', 
              metrics=['accuracy'])


# In[5]:


model = load_model('model-00049-0.71733.h5')
result = model.predict(test_x)
np.save('result49.npy',result)


# In[6]:


model = load_model('model-00057-0.71833.h5')
result = model.predict(test_x)
np.save('result57.npy',result)


# In[7]:


model = load_model('model-00086-0.71867.h5')
result = model.predict(test_x)
np.save('result86.npy',result)


# In[8]:


model = load_model('model-00087-0.72033.h5')
result = model.predict(test_x)
np.save('result87.npy',result)


# In[10]:


model.load_weights('model-00149-0.71533.h5')
result = model.predict(test_x)
np.save('result149.npy',result)


# In[11]:


model.load_weights('model-00161-0.71700.h5')
result = model.predict(test_x)
np.save('result161.npy',result)


# In[12]:


model.load_weights('model-00164-0.72400.h5')
result = model.predict(test_x)
np.save('result164.npy',result)


# In[13]:


model.load_weights('model-00188-0.71533.h5')
result = model.predict(test_x)
np.save('result188.npy',result)


# In[23]:


model.load_weights('model-00198-0.71700.h5')
result = model.predict(test_x)
np.save('result198.npy',result)


# In[ ]:


#w model
print('start w prediction')




# In[15]:


model = load_model('model-w1.h5')
result = model.predict(test_x)
np.save('result_w1.npy',result)


# In[16]:


model = load_model('model-w2.h5')
result = model.predict(test_x)
np.save('result_w2.npy',result)


# In[ ]:


#y model
print('start y prediction')




# In[17]:


model = load_model('model-y1.h5')
result = model.predict(test_x)
np.save('result_y1.npy',result)


# In[18]:
model = load_model('model-y2.h5')
result = model.predict(test_x)
np.save('result_y2.npy',result)


# In[24]:


#write ans
print('start final prediction')

my1 = np.load('result49.npy')
my2 = np.load('result57.npy')
my3 = np.load('result86.npy')
my4 = np.load('result87.npy')
my5 = np.load('result149.npy')
my6 = np.load('result161.npy')
my7 = np.load('result164.npy')
my8 = np.load('result188.npy')
my9 = np.load('result198.npy')
w1 = np.load('result_w1.npy')
w2 = np.load('result_w2.npy')
y1 = np.load('result_y1.npy')
y2 = np.load('result_y2.npy')

result = w1 + w2 + y1 + y2 + my1 + my2 + my3 + my4 + my5 + my6 + my7 + my8 + my9

ans = []
for i in range(len(result)):
    max = 0
    senti = 0
    for j in range(7):
        if result[i][j] > max:
            max = result[i][j]
            senti = j
    ans.append([i])
    ans[i].append(senti)
    
text = open(sys.argv[2], "w+")
s = csv.writer(text, delimiter=',')
s.writerow(["id","label"])
for i in range(len(ans)):
    s.writerow(ans[i])
text.close()

print('done, please check!')