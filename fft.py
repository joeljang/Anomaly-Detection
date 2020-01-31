from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import time
import numpy as np
import csv
import six
import tensorflow as tf
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as im
import re

def sorted_aphanumeric(data):
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ] 
    return sorted(data, key=alphanum_key)

data_dir = sys.argv[1]

SAMPLE_LENGTH = 5000
SAMPLE_RATE = 5000

def fft(signal):
  with tf.Graph().as_default():
    signal = tf.Variable(signal, dtype=tf.complex64)
    fft = tf.fft(signal)
    with tf.Session() as sess:
        tf.variables_initializer([signal]).run()
        result = sess.run(fft)
  return result

lst = sorted_aphanumeric(os.listdir(data_dir))
cnt=0
for filename in lst:
    cnt+=1
    if(cnt<10 or cnt>255):
      print(filename)
      data = np.genfromtxt(os.path.join(data_dir, filename), delimiter=',')
      print(data.shape)
      data = data[:,0:3]
      x = data[:,0]
      y = data[:,1]
      transformx = fft(x)
      transformy = fft(y)
      for i in range(len(transformx)):
          if(transformx[i]>1000):
              print(i)
      plt.figure(cnt)
      plt.plot(range(len(transformy)),transformy)
      plt.savefig(filename+'yfft.png')
      cnt+=1
      plt.figure(cnt)
      plt.plot(range(len(transformx)),transformx)
      plt.savefig(filename+'xfft.png')