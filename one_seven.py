import numpy as np
import matplotlib.pyplot as plt

import input_data

# Constants
AREA = 200
O = 'r'
S = 'b'

# Data Loading

# Feature Extraction

# Plotting
x = np.asarray([0.2, 0.1, -1])
y = np.asarray([-2, 0.7, 1.3])
colors = [O, S, S]

plt.scatter(x, y, s=AREA, c=colors)
plt.show()
