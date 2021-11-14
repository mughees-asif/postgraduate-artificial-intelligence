import matplotlib.pyplot as plt

from load_data_ex1 import *
from normalize_features import *
from plot_data_function import *

# this loads our data
X, y = load_data_ex1()

# this normalizes our data
X_normalized, mean_vec, std_vec = normalize_features(X)

# After normalizing, we append a column of ones to X_normalized, as the bias term
column_of_ones = np.ones((X_normalized.shape[0], 1))
# append column to the dimension of columns (i.e., 1)
X_normalized = np.append(column_of_ones, X_normalized, axis=1)
# print(X)
fig, ax1 = plt.subplots()
ax1 = plot_data_function(X_normalized, y, ax1)

# enter non-interactive mode of matplotlib, to keep figures open
plt.ioff()
plt.show()
