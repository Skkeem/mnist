import numpy as np
import matplotlib.pyplot as plt

import input_data

# Constants
AREA = 200
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

SIZE = 10000
images = mnist.train.images[:SIZE]
labels = mnist.train.labels[:SIZE]

array0, array7 = extract_1_7(images, labels)

# Feature Extraction

# Plotting
x1 = np.asarray([0.2, 0.1, -1])
y1 = np.asarray([-2, 0.7, 1.3])
x7 = np.asarray([0.5, 2.1, -4])
y7 = np.asarray([2, -0.7, 2.1])
colors = [O, S, S]

plt.scatter(x1, y1, s=AREA, c=O)
plt.scatter(x7, y7, s=AREA, c=S)
plt.show()
