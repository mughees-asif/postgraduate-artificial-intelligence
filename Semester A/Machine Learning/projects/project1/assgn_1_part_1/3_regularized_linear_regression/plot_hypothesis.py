import matplotlib.pyplot as plt
from hypothesis_to_vector import *
import os

def plot_hypothesis(X, y, theta, ax1):
    """
        :param X            : 2D array of our dataset
        :param y            : 1D array of the groundtruth labels of the dataset
        :param theta        : 1D array of the trainable parameters
        :param ax1          : existing subplot, to draw on it
    """
    
    # clear subplot from previous (if any) drawn stuff
    ax1.clear()
    # set label of horizontal axis
    ax1.set_xlabel('x1')
    # set label of vertical axis
    ax1.set_ylabel('y=f(x1)')
    # scatter the points representing the groundtruth prices of the training samples, with red color
    ax1.scatter(X[:,1], y, c='red', marker='x', label='groundtruth')
    # plot the line that connects the points representing the predicted values, with blue color
    outputs = hypothesis_to_vector(X, theta)
    ax1.plot(X[:,1], outputs, c='blue', marker='o', label='prediction')
    # add legend to the subplot
    ax1.legend()
    
    # Pause for a short time, to allow the plotted points to be shown in the figure
    plt.pause(0.001)
