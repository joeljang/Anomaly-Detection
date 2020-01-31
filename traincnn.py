import cv2
import os
import matplotlib.pyplot as plt
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Conv3D, Conv2D, BatchNormalization, Dropout, Flatten
import sys
from random import sample

#DATA LOADING
data_dir = sys.argv[1] #training data
data_dir2 = sys.argv[2] #testing data

ratio = .3 #Training to non-training data ratio on the training data set.

x_train = []
y_train = []
test = []

#Loading the training data
count=0
lst = os.listdir(data_dir)
lst.sort()
totalnum = len(lst)
down_threshhold = int((ratio/2)*totalnum)
up_threshhold = int((1 - (ratio/2))*totalnum)
print('Total number of files in training datset: ',totalnum)
print('Down Threshhold for healthy labeling: ',down_threshhold)
print('Up Threshhold for unhealthy labeling: ',up_threshhold)
for filename in lst:
    count += 1
    print(filename)
    image = cv2.imread('.//'+data_dir+'//'+filename)
    if(count<down_threshhold or count >up_threshhold):
        x_train.append(image)
        if(count<down_threshhold):
            y_train.append(0)
        else:
            y_train.append(1)

#Loading the testing data
lst = os.listdir(data_dir2)
lst.sort()
totalnum = len(lst)
for filename in lst:
    print(filename)
    image = cv2.imread('.//'+data_dir2+'//'+filename)
    test.append(image)

x_train = np.array(x_train)
y_train = np.array(y_train)
test = np.array(test)

print('Shape of training data:',x_train.shape)
print('Shape of training data label:',y_train.shape)
print('Shape of testing data',test.shape)

#Building the model
model = Sequential()
K1 = 10    # Conv1 layer feature map depth
K2 = 20    # Conv2 layer feature map depth
K3 = 40    # Conv3 layer feature map depth
K4 = 20    # Conv4 layer feature map depth
F1 = 500   # Full1 layer node size
F2 = 50    # Full2 layer node size
output = 1

#add model layers
model.add(Conv2D(K2, kernel_size=(10, 10),strides=(2,2), activation='relu', input_shape=(128,128,3)))
model.add(BatchNormalization())
model.add(Conv2D(K3, kernel_size=(5, 5),strides=(2,2), activation='relu'))
model.add(Conv2D(K4, kernel_size=(3, 3),strides=(1,1), activation='relu'))
model.add(Flatten())
model.add(Dense(F1, activation='relu'))
model.add(Dropout(0.75))
model.add(Dense(F2, activation='relu'))
model.add(Dropout(0.75))
model.add(Dense(output, activation='sigmoid'))

#compile model using accuracy to measure model performance
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

#train the model
print(model.summary())
history = model.fit(x_train, y_train, batch_size=8, validation_split=.1, epochs=50,shuffle=True)

#Evaluating model with test data
prediction = model.predict(test)

plt.scatter(range(len(prediction)),prediction,s=1)
plt.savefig('prediction_test.png')