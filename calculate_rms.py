from sklearn.decomposition import PCA
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
import math
import re

def sorted_aphanumeric(data):
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ] 
    return sorted(data, key=alphanum_key)

def getRMSData(x,y):
    return [ math.sqrt((x[i]*x[i])+(y[i]*y[i]))  for i in range(0,len(x))]

data_dir = sys.argv[1]

pca = PCA(n_components=2, svd_solver= 'full')

lst = sorted_aphanumeric(os.listdir(data_dir))
b = []

for filename in lst:
    print(filename)
    data = np.genfromtxt(os.path.join(data_dir, filename), delimiter=',')
    print(data.shape)
    data = data[:,0:3]
    x = data[:,0]
    y = data[:,1]
    rms = getRMSData(x,y)
    rms = np.mean(rms)
    b.append(rms)

b = np.array(b)

plt.figure(1)
plt.plot(range(len(b)),b,color='black')
plt.savefig('rms.png')