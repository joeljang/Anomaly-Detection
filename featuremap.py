import cv2
import os
import numpy as np
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D
from keras.models import Model, Sequential
from keras import backend as K
from keras.callbacks import TensorBoard
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets.samples_generator import make_blobs
from sklearn.cluster import KMeans

def save_tsne(data,label,name):
    print("making t-sne...")

    tsne = TSNE(n_components=2, perplexity=30.0, early_exaggeration=12.0,
        learning_rate=200.0, n_iter=1000, n_iter_without_progress=300,
        min_grad_norm=1e-07, metric='euclidean', init='random', verbose=0,
        random_state=None, method='barnes_hut', angle=0.5)

    pca = PCA(n_components=100)

    pca_data = pca.fit_transform(data)
    result = tsne.fit_transform(pca_data)

    x = []
    y = []

    for i in range(len(result)):
        x.append(float(result[i][0]))
    for i in range(len(result)):
        y.append(float(result[i][1]))
            
            
    x = np.asarray(x)
    y = np.asarray(y)

    cdict = { 0:'r', 1:'g', 2: 'b', 3: 'k', 4: 'k', 5: 'm', 6: 'y', 7: 'k', 8: 
            '#ffff00', 9: '#ff0090'}
    plt.figure()
    for g in np.unique(label):
        ix = np.where(label == g)
        plt.scatter(x[ix],y[ix],c=cdict[g],s=1)
    plt.savefig(name+'.png')
    print('t-sne image saved')

def reshapeFeature(feature):
    feature = np.reshape(feature, (feature.shape[0],2048))
    return feature

#DATA LOADING
data_dir = sys.argv[1] #training data
data_dir2 = sys.argv[2] #testing data

#Training data
train=[]
trainy=[]

lst = os.listdir(data_dir)
lst.sort()
totalnum = len(lst)

#Data partition into 4 stages
one=int(totalnum/4)
two=int(totalnum/2)
three=int(one*3)
count=0
for filename in lst:
    count+=1
    image = cv2.imread('.//'+data_dir+'//'+filename)
    train.append(image)
    if(count<=one):
        trainy.append(0)
    elif(count>one and count<=two):
        trainy.append(1)
    elif(count>two and count<=three):
        trainy.append(2)
    else:
        trainy.append(3)
train = np.array(train)
trainy = np.array(trainy)

#Testing data
test=[]
testy=[]
lst = os.listdir(data_dir2)
lst.sort()
totalnum = len(lst)
one=int(totalnum/4)
two=int(totalnum/2)
three=int(one*3)
count=0
for filename in lst:
    count+=1
    image = cv2.imread('.//'+data_dir2+'//'+filename)
    test.append(image)
    if(count<=one):
        testy.append(0)
    elif(count>one and count<=two):
        testy.append(1)
    elif(count>two and count<=three):
        testy.append(2)
    else:
        testy.append(3)
test = np.array(test)
testy = np.array(testy)

#****************************AUTOENCODER**********************

input_img = Input(shape=(128, 128, 3))  # adapt this if using `channels_first` image data format

x = Conv2D(16, (3, 3), activation='relu', padding='same')(input_img)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
encoded = MaxPooling2D((2, 2), padding='same', name='fe')(x)

# at this point the representation is (4, 4, 8) i.e. 128-dimensional

x = Conv2D(8, (3, 3), activation='relu', padding='same')(encoded)
x = UpSampling2D((2, 2))(x)
x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
x = UpSampling2D((2, 2))(x)
x = Conv2D(16, (3, 3), activation='relu', padding='same')(x)
x = UpSampling2D((2, 2))(x)
decoded = Conv2D(3, (3, 3), activation='sigmoid', padding='same')(x)

autoencoder = Model(input_img, decoded)
autoencoder.summary()
autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')

autoencoder.fit(train, train,
                epochs=1,
                batch_size=32,
                shuffle=True,
                validation_split=.05)
layer_name = 'fe'
intermediate_layer_model = Model(inputs=autoencoder.input,
                                 outputs=autoencoder.get_layer(layer_name).output)

feature = intermediate_layer_model.predict(train)
save_tsne(reshapeFeature(feature),trainy,'train')

feature_test = intermediate_layer_model.predict(test)
save_tsne(reshapeFeature(feature_test),testy,'test')