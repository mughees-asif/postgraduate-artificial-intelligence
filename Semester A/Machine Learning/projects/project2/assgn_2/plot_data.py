import matplotlib.pyplot as plt
import os

def plot_data(X, title_string, ax):
    """
        :param X            : 2D array of our dataset
        :param y            : 1D array of the groundtruth labels of the dataset
        :param ax          : existing subplot, to draw on it
    """
    
    # clear subplot from previous (if any) drawn stuff
    ax.clear()
    # set label of horizontal axis
    ax.set_xlabel('f1')
    # set label of vertical axis
    ax.set_ylabel('f2')
    # scatter the points, with red color
    ax.scatter(X[:,0], X[:,1], c='black', marker='.', label=title_string)
    # add legend to the subplot
    ax.legend()
    
    # Pause for a short time, to allow the plotted points to be shown in the figure
    plt.pause(0.001)