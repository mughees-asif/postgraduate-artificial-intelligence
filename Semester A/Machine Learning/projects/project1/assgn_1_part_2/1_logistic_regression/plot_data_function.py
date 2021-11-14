import numpy as np
import matplotlib.pyplot as plt

def plot_data_function(X, y, ax1):
    
    # clear subplot from previous (if any) drawn stuff
    #ax1.clear()
    # set label of horizontal axis
    ax1.set_xlabel('x1')
    # set label of vertical axis
    ax1.set_ylabel('x2')
    
    # get indices of points with groundtruth label equal to 0
    inds_zero = np.where(y==0)
    # get indices of points with groundtruth label equal to 1
    inds_one = np.where(y==1)
    
    # scatter the points representing with groundtruth label equal to 0
    ax1.scatter(X[inds_zero,1], X[inds_zero,2], c='red', marker='x', label='class 0')
    # scatter the points representing with groundtruth label equal to 1
    ax1.scatter(X[inds_one,1], X[inds_one,2], c='blue', marker='+', label='class 1')
    
    return ax1
