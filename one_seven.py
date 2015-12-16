import numpy as np
import matplotlib.pyplot as plt

import input_data

# Constants
AREA = 50
SIZE = 1000
O = 'r'
S = 'b'


# Data Loading
mnist = input_data.read_data_sets("./data")

print 'Reading data set complete'

def extract_1_7(images, labels):
    array1 = []
    array7 = []

    for i in range(len(labels)):
        if labels[i] == 1:
            array1.append(images[i])
        elif labels[i] == 7:
            array7.append(images[i])
            
    return array1, array7

images = mnist.train.images[:SIZE]
labels = mnist.train.labels[:SIZE]

array1, array7 = extract_1_7(images, labels)

print '1, 7 extraction complete'


# Feature Extraction
"""
Feature function is a function from 784 dimensional float array to float value.

[0., 0., ..., 0.2, 0.7] -> 0.7
"""
def featureX(x):
    """
    Count the number of pixels > 0
    """
    return len(filter(lambda x: x>0, x))

def featureY(x):
    """
    Find the leftmost x which is not zero
    """
    v = 28

    for j in range(0, 28):
        for i in range(0, 28):
            if x[i + j * 28] > 0 and i < v:
		v = i

    return v    

x1 = map(featureX, array1)
y1 = map(featureY, array1)
x7 = map(featureX, array7)
y7 = map(featureY, array7)

print 'Feature extraction complete'


# Plotting
plt.scatter(x1, y1, s=AREA, c=O, alpha=0.4)
plt.scatter(x7, y7, s=AREA, c=S, alpha=0.4)
plt.show()
