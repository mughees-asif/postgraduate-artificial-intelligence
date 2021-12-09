import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
import numpy as np


def plot_data_all_phonemes(X, phoneme_id, ax):
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

    # get the unique values of the phoneme ids
    unique_ids = np.unique(phoneme_id)
    # get the number of unique values
    N_ids = len(unique_ids)

    for ph_id in unique_ids:
        phoneme_string = 'Phoneme {:02}'.format(ph_id)
        # scatter the points
        ax.scatter(X[phoneme_id==ph_id,0], X[phoneme_id==ph_id,1], marker='.', label=phoneme_string)
        #ax.scatter(X[phoneme_id==ph_id,0], X[phoneme_id==ph_id,1], c=color[ph_cnt], marker='.', label=phoneme_string)
    # add legend to the subplot
    ax.legend()
    
    # Pause for a short time, to allow the plotted points to be shown in the figure
    plt.pause(0.001)