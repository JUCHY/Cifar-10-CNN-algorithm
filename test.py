# -*- coding: utf-8 -*-
"""
Created on Fri Mar 20 22:23:34 2020

@author: joshu
"""
from PIL import Image
import json
import matplotlib.pyplot as plt
import numpy as np


def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict
test = unpickle('./cifar-10-batches-py/test_batch')
print(test.keys())
print(test[b'batch_label'])
print(test[b'filenames'])
print(len(test[b'data']))
labels = test[b'labels'][1]
print(labels)
img = test[b'data'][1]
full_img = np.reshape(img, (3,32,32))
plt.imshow(np.transpose(full_img, (1, 2, 0)))
plt.show()
meta = unpickle('./cifar-10-batches-py/batches.meta')
print(meta[b'label_names'])