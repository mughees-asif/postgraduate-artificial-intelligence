import numpy as np
import matplotlib.pyplot as plt
from sigmoid import *


def plot_sigmoid():
    x = np.linspace(1, 2000) / 100.0 - 10
    y = sigmoid(x)
    fig, ax1 = plt.subplots()
    ax1.plot(x, y)
    # set label of horizontal axis
    ax1.set_xlabel('x')
    # set label of vertical axis
    ax1.set_ylabel('sigmoid(x)')
    ax1.spines['left'].set_position('zero')
    ax1.spines['right'].set_position('zero')
    plt.show()

# plot_sigmoid()
