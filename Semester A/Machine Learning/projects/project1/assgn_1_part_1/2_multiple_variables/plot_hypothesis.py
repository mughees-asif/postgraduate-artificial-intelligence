import matplotlib.pyplot as plt
from hypothesis_to_vector import *

def plot_hypothesis(X, y, theta, ax1):
    # clear subplot from previous (if any) drawn stuff
    ax1.clear()
    # set label of horizontal axis
    ax1.set_xlabel('x1')
    # set label of vertical axis
    ax1.set_ylabel('y=f(x1)')
    # scatter the points representing the groundtruth prices of the training samples, with red color
    ax1.scatter(X[:,1], y, c='red', marker='x', label='groundtruth')
    # scatter the points representing the predicted prices, with blue color
    ax1.scatter(X[:,1], hypothesis_to_vector(X, theta), c='blue', marker='+', label='prediction')
    # add legend to the subplot
    ax1.legend()
    
    # Pause for a short time, to allow the plotted points to be shown in the figure
    plt.pause(0.001)
