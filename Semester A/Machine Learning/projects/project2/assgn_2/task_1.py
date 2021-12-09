import numpy as np
import os
import matplotlib.pyplot as plt
from print_values import *
from plot_data_all_phonemes import *
from plot_data import *

# File that contains the data
data_npy_file = 'data/PB_data.npy'

# Loading data from .npy file
data = np.load(data_npy_file, allow_pickle=True)
data = np.ndarray.tolist(data)

# Make a folder to save the figures
figures_folder = os.path.join(os.getcwd(), 'figures')
if not os.path.exists(figures_folder):
    os.makedirs(figures_folder, exist_ok=True)

# Array that contains the phoneme ID (1-10) of each sample
phoneme_id = data['phoneme_id']

# frequencies f1 and f2
f1 = data['f1']
f2 = data['f2']
print('f1 statistics:')
print_values(f1)
print('f2 statistics:')
print_values(f2)

# Initialize array containing f1 & f2, of all phonemes.
X_full = np.zeros((len(f1), 2))
#########################################
# Write your code here
# Store f1 in the first column of X_full, and f2 in the second column of X_full
X_full[:, 0] = f1
X_full[:, 1] = f2
########################################/
X_full = X_full.astype(np.float32)

# you can use the p_id variable, to store the ID of the chosen phoneme that will be used (e.g. phoneme 1, or phoneme 2)
p_id = 1

#########################################
# Write your code here

# Create an array named "X_phoneme_1", containing only samples that belong to the chosen phoneme. The shape of
# X_phoneme_1 will be two-dimensional. Each row will represent a sample of the dataset, and each column will
# represent a feature (e.g. f1 or f2) Fill X_phoneme_1 with the samples of X_full that belong to the chosen phoneme
# To fill X_phoneme_1, you can leverage the phoneme_id array, that contains the ID of each sample of X_full

# Create array containing only samples that belong to phoneme 1
# X_phoneme_1 = np.zeros((np.sum(phoneme_id==1), 2))
X_phoneme_1 = X_full[phoneme_id == p_id, :]

########################################

# Plot array containing all phonemes

# Create a figure and a subplot
fig, ax1 = plt.subplots()
# plot the full dataset (f1 & f2, all phonemes)
plot_data_all_phonemes(X=X_full, phoneme_id=phoneme_id, ax=ax1)
# save the plotted dataset as a figure
plot_filename = os.path.join(os.getcwd(), 'figures', 'dataset_full.png')
plt.savefig(plot_filename)

################################################
# Plot array containing phoneme 1

# Create a figure and a subplot
fig, ax2 = plt.subplots()
title_string = 'Phoneme 1'
# plot the samples of the dataset, belonging to phoneme 1 (f1 & f2, phoneme 1)
plot_data(X=X_phoneme_1, title_string=title_string, ax=ax2)
# save the plotted points of phoneme 1 as a figure
plot_filename = os.path.join(os.getcwd(), 'figures', 'dataset_phoneme_1.png')
plt.savefig(plot_filename)

# enter non-interactive mode of matplotlib, to keep figures open
plt.ioff()
plt.show()