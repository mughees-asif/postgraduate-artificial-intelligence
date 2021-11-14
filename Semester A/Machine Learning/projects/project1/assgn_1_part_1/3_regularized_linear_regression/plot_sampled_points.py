import matplotlib.pyplot as plt
import numpy as np

def get_hypothesis(x, theta):
    """
        :param x        : scalar of a given sample
        :param theta    : 1D array of the trainable parameters
    """
    
    hypothesis = 0.0
    for t in range(len(theta)):
        # compute prediction by raising x to each power t
        hypothesis += theta[t] * (x ** t)
    
    return hypothesis

def plot_sampled_points(X, y, X_sampled, theta, ax2):
    """
        :param X            : 2D array of our dataset
        :param y            : 1D array of the groundtruth labels of the dataset
        :param X_sampled    : 2D array of sampled points
        :param theta        : 1D array of the trainable parameters
        :param ax1          : existing subplot, to draw on it
    """
    
    # clear subplot from previous (if any) drawn stuff
    ax2.clear()
    # set label of horizontal axis
    ax2.set_xlabel('x1')
    # set label of vertical axis
    ax2.set_ylabel('y=f(x1)')
    # scatter the points representing the groundtruth prices of the training samples, with red color
    ax2.scatter(X[:,1], y, c='red', marker='x', label='groundtruth')
    
    # y_sampled will be an 1D array of the predicted values on the sampled points X_sampled
    y_sampled = np.array([], np.float32)
    # for each one of the sampled points of X_sampled, compute the prediction and append it to y_sampled
    for x in X_sampled:
        y_sampled = np.append(y_sampled, get_hypothesis(x, theta))
    # plot the line that connects the points representing the predicted values, with blue color
    ax2.plot(X_sampled, y_sampled, c='blue', label='prediction')
    # add legend to the subplot
    ax2.legend()
    
    # Pause for a short time, to allow the plotted points to be shown in the figure
    plt.pause(0.001)
