# Common imports
import os
import pandas as pd
import numpy as np
from sklearn import preprocessing
import seaborn as sns
sns.set(color_codes=True)
import matplotlib.pyplot as plt
import csv

from numpy.random import seed
from tensorflow import set_random_seed

from keras.layers import Input, Dropout
from keras.layers.core import Dense 
from keras.models import Model, Sequential, load_model
from keras import regularizers
from keras.models import model_from_json

from sklearn.decomposition import PCA

#read CSV file
def readCsv(csvpath):
  rawdata = []

  # read in csv file if exists
  if os.path.isfile(csvpath):
    with open(csvpath) as csvfile:
      reader = csv.reader(csvfile, delimiter=',')
      for row in reader:
        values = []
        for col in row:
          try:
            values.append(float(col))
          except:
            print(' ERROR: inadequate data value:', col)
            print('       in file:', csvpath)
            if col == '∞':
              values.append(0.0)
            elif col == '-∞':
              values.append(0.0)
            else:
              print(' FATAL ERROR: exit process', csvpath)
              exit()
        rawdata.append([ values[0], values[1], values[2], values[3], values[4], values[5] ])
      if rawdata == []:
        print('ERROR: cannot find data in file:', csvpath)
        exit()
      else:
        waste = len(rawdata)-SAMPLE_LENGTH
        for i in range(waste):
          rawdata.pop()
        rawdata = np.array(rawdata)
  else:
    print('ERROR: cannot find input file:', csvpath)
    exit()
  return rawdata

#Calculate the covariance matrix
def cov_matrix(data, verbose=False):
    covariance_matrix = np.cov(data, rowvar=False)
    if is_pos_def(covariance_matrix):
        inv_covariance_matrix = np.linalg.inv(covariance_matrix)
        if is_pos_def(inv_covariance_matrix):
            return covariance_matrix, inv_covariance_matrix
        else:
            print("Error: Inverse of Covariance Matrix is not positive definite!")
    else:
        print("Error: Covariance Matrix is not positive definite!")

#Calculate the Mahalanobis distance
def MahalanobisDist(inv_cov_matrix, mean_distr, data, verbose=False):
    inv_covariance_matrix = inv_cov_matrix
    vars_mean = mean_distr
    diff = data - vars_mean
    md = []
    for i in range(len(diff)):
        md.append(np.sqrt(diff[i].dot(inv_covariance_matrix).dot(diff[i])))
    return md

#Detecting outliers:
def MD_detectOutliers(dist, extreme=False, verbose=False):
    k = 3. if extreme else 2.
    threshold = np.mean(dist) * k
    outliers = []
    for i in range(len(dist)):
        if dist[i] >= threshold:
            outliers.append(i)  # index of the outlier
    return np.array(outliers)

#Calculate threshold value for classifying datapoint as anomaly:
def MD_threshold(dist, extreme=False, verbose=False):
    k = 4. if extreme else 3.
    threshold = np.mean(dist) * k
    return threshold

#Check if matrix is positive definite:
def is_pos_def(A):
    if np.allclose(A, A.T):
        try:
            np.linalg.cholesky(A)
            return True
        except np.linalg.LinAlgError:
            return False
    else:
        return False

#Data loading and pre-processing
data_dir = './Learning_set/Bearing1_1'
merged_data = pd.DataFrame()
SAMPLE_LENGTH = 2560
count=0

for filename in os.listdir(data_dir):
    count += 1
    print(filename)
    path = data_dir+'//'+filename
    data = readCsv(path)
    time = pd.DataFrame({'Hour':data[:,0], 'Minute':data[:,1], 'Second':data[:,2]})
    dataset = pd.DataFrame({'Horizontal':data[:,4], 'Vertical':data[:,5]})
    dataset_mean_abs = np.array(dataset.abs().mean())
    dataset_mean_abs = pd.DataFrame(dataset_mean_abs.reshape(1,2))
    dataset_mean_abs.index = [count]
    merged_data = merged_data.append(dataset_mean_abs)

merged_data.columns = ['Horizontal','Vertical']

#Transform the index to datetime format\
merged_data = merged_data.sort_index()
merged_data.to_csv('training1.csv')

#Defining train/test data
dataset_train = merged_data[0:500]
dataset_test = merged_data[500:]

#Normalizing the data from scale to 0 to 1
scaler = preprocessing.MinMaxScaler()

X_train = pd.DataFrame(scaler.fit_transform(dataset_train), 
                              columns=dataset_train.columns, 
                              index=dataset_train.index)
# Random shuffle training data
X_train.sample(frac=1)

X_test = pd.DataFrame(scaler.transform(dataset_test), 
                             columns=dataset_test.columns, 
                             index=dataset_test.index)

#Using PCA for dimension reduction
'''
pca = PCA(n_components=2, svd_solver= 'full')
X_train_PCA = pca.fit_transform(X_train)
X_train_PCA = pd.DataFrame(X_train_PCA)
X_train_PCA.index = X_train.index

X_test_PCA = pca.transform(X_test)
X_test_PCA = pd.DataFrame(X_test_PCA)
X_test_PCA.index = X_test.index
'''

X_train_PCA = X_train
X_test_PCA = X_test
#*************************************

#Set up PCA model
data_train = np.array(X_train_PCA.values)
data_test = np.array(X_test_PCA.values)

#Calculate the covariance matrix and its invers, based on data in the training set:
cov_matrix, inv_cov_matrix  = cov_matrix(data_train)

#Mean value for input variables in training used to calculate the Mahalanobis 
#distance to datapoints in the test set
mean_distr = data_train.mean(axis=0)

#Calculating the Mahalanobis distance under normal conditions
dist_test = MahalanobisDist(inv_cov_matrix, mean_distr, data_test, verbose=False)
dist_train = MahalanobisDist(inv_cov_matrix, mean_distr, data_train, verbose=False)
threshold = MD_threshold(dist_train, extreme = True)

print('Let us start visualizing!!')

#Visualizing the square of the Mahalanobis distance
plt.figure()
sns.distplot(np.square(dist_train),
             bins = 10, 
             kde= False);
plt.xlim([0.0,15])

#Visualizing the Mahalanobis distance itself
plt.figure()
sns.distplot(dist_train,
             bins = 10, 
             kde= True, 
            color = 'green');
plt.xlim([0.0,5])
plt.xlabel('Mahalanobis dist')
plt.show()

#Flagging anomaly
anomaly_train = pd.DataFrame()
anomaly_train['Mob dist']= dist_train
anomaly_train['Thresh'] = threshold
# If Mob dist above threshold: Flag as anomaly
anomaly_train['Anomaly'] = anomaly_train['Mob dist'] > anomaly_train['Thresh']
anomaly_train.index = X_train_PCA.index
anomaly = pd.DataFrame()
anomaly['Mob dist']= dist_test
anomaly['Thresh'] = threshold
# If Mob dist above threshold: Flag as anomaly
anomaly['Anomaly'] = anomaly['Mob dist'] > anomaly['Thresh']
anomaly.index = X_test_PCA.index
anomaly.head()

#Merge into single dataframe and save as .csv file
anomaly_alldata = pd.concat([anomaly_train, anomaly])
anomaly_alldata.to_csv('Anomaly_distance.csv')

anomaly_alldata.plot(logy=True, figsize = (10,6), ylim = [1e-1,1e3], color = ['green','red'])
plt.show()


#****************************AUTOENCODER**********************
seed(10)
set_random_seed(10)
act_func = 'elu'

# Input layer:
model=Sequential()
# First hidden layer, connected to input vector X. 
model.add(Dense(10,activation=act_func,
                kernel_initializer='glorot_uniform',
                kernel_regularizer=regularizers.l2(0.0),
                input_shape=(X_train.shape[1],)
               )
         )

model.add(Dense(2,activation=act_func,
                kernel_initializer='glorot_uniform'))

model.add(Dense(10,activation=act_func,
                kernel_initializer='glorot_uniform'))

model.add(Dense(X_train.shape[1],
                kernel_initializer='glorot_uniform'))

model.compile(loss='mse',optimizer='adam')

# Train model for 100 epochs, batch size of 10: 
NUM_EPOCHS=100
BATCH_SIZE=10

#Fitting the model
history=model.fit(np.array(X_train),np.array(X_train),
                  batch_size=BATCH_SIZE, 
                  epochs=NUM_EPOCHS,
                  validation_split=0.05,
                  verbose = 1)

#Visualize training/validation loss:
plt.plot(history.history['loss'],
         'b',
         label='Training loss')
plt.plot(history.history['val_loss'],
         'r',
         label='Validation loss')
plt.legend(loc='upper right')
plt.xlabel('Epochs')
plt.ylabel('Loss, [mse]')
plt.ylim([0,.1])
plt.show()

#Distribution of loss function in the training set
X_pred = model.predict(np.array(X_train))
X_pred = pd.DataFrame(X_pred, 
                      columns=X_train.columns)
X_pred.index = X_train.index

scored = pd.DataFrame(index=X_train.index)
scored['Loss_mae'] = np.mean(np.abs(X_pred-X_train), axis = 1)
plt.figure()
sns.distplot(scored['Loss_mae'],
             bins = 10, 
             kde= True,
            color = 'blue');
plt.xlim([0.0,.1])
plt.show()

#Checking when the test set cross anomaly boundary which is loss of .3
X_pred = model.predict(np.array(X_test))
X_pred = pd.DataFrame(X_pred, 
                      columns=X_test.columns)
X_pred.index = X_test.index

scored = pd.DataFrame(index=X_test.index)
scored['Loss_mae'] = np.mean(np.abs(X_pred-X_test), axis = 1)
scored['Threshold'] = 0.03
scored['Anomaly'] = scored['Loss_mae'] > scored['Threshold']

#Also calculate the same metrics for the training set, and merge all data to single dataframe.
X_pred_train = model.predict(np.array(X_train))
X_pred_train = pd.DataFrame(X_pred_train, 
                      columns=X_train.columns)
X_pred_train.index = X_train.index

scored_train = pd.DataFrame(index=X_train.index)
scored_train['Loss_mae'] = np.mean(np.abs(X_pred_train-X_train), axis = 1)
scored_train['Threshold'] = 0.03
scored_train['Anomaly'] = scored_train['Loss_mae'] > scored_train['Threshold']
scored = pd.concat([scored_train, scored])

scored.plot(logy=True,  figsize = (10,6), ylim = [0,1e2], color = ['blue','red'])
plt.show()