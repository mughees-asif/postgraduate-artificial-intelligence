import numpy as np
import os
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
from print_values import *
from plot_data_all_phonemes import *
from plot_data_3D import *
import random
from sklearn.preprocessing import normalize
from get_predictions import *
from plot_gaussians import *

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

# Initialize array containing f1, f2 & f1+f2, of all phonemes.
X_full = np.zeros((len(f1), 3))
#########################################
# Write your code here
# Store f1 in the first column of X_full, f2 in the second column of X_full and f1+f2 in the third column of X_full

########################################/
X_full = X_full.astype(np.float32)

# We will train a GMM with k components, on a selected phoneme id which is stored in variable "p_id" 

# id of the phoneme that will be used (e.g. 1, or 2)
p_id = 1
# number of GMM components
k = 6
#########################################
# Write your code here

# Create an array named "X_phoneme", containing only samples that belong to the chosen phoneme.
# The shape of X_phoneme will be two-dimensional. Each row will represent a sample of the dataset, and each column will
# represent a feature (e.g. f1 or f2 or f1+f2)
# Fill X_phoneme with the samples of X_full that belong to the chosen phoneme
# To fill X_phoneme, you can leverage the phoneme_id array, that contains the ID of each sample of X_full
X_full[:, 0] = f1
X_full[:, 1] = f2
X_full[:, 2] = f1 + f2
X_phoneme = X_full[phoneme_id == p_id, :]

########################################/

################################################
# Plot array containing the chosen phoneme

# Create a figure and a subplot
fig = plt.figure()
ax1 = plt.axes(projection='3d')

title_string = 'Phoneme {}'.format(p_id)
# plot the samples of the dataset, belonging to the chosen phoneme (f1 & f2, phoneme 1 or 2)
plot_data_3D(X=X_phoneme, title_string=title_string, ax=ax1)
# save the plotted points of phoneme 1 as a figure
plot_filename = os.path.join(os.getcwd(), 'figures', 'dataset_3D_phoneme_{}.png'.format(p_id))
plt.savefig(plot_filename)

################################################
# Train a GMM with k components, on the chosen phoneme

# as dataset X, we will use only the samples of the chosen phoneme
X = X_phoneme.copy()

# get number of samples
N = X.shape[0]
# get dimensionality of our dataset
D = X.shape[1]

# common practice : GMM weights initially set as 1/k
p = np.ones((k))/k
# GMM means are picked randomly from data samples
random_indices = np.floor(N*np.random.rand((k)))
random_indices = random_indices.astype(int)
mu = X[random_indices,:] # shape kxD
# covariance matrices
s = np.zeros((k,D,D)) # shape kxDxD
# number of iterations for the EM algorithm
n_iter = 150

# initialize covariances
for i in range(k):
    cov_matrix = np.cov(X.transpose())
    # initially set to fraction of data covariance
    s[i,:,:] = cov_matrix/k

# Initialize array Z that will get the predictions of each Gaussian on each sample
Z = np.zeros((N,k)) # shape Nxk

###############################
# run Expectation Maximization algorithm for n_iter iterations
for t in range(n_iter):
    #print('****************************************')
    print('Iteration {:03}/{:03}'.format(t+1, n_iter))

    # Do the E-step
    Z = get_predictions(mu, s, p, X)
    Z = normalize(Z, axis=1, norm='l1')

    # Do the M-step:
    for i in range(k):
        mu[i,:] = np.matmul(X.transpose(), Z[:,i])/np.sum(Z[:,i])
        
        ###################################################
        # We will fit Gaussians with full covariance matrices:
        mu_i = mu[i,:]
        mu_i = np.expand_dims(mu_i, axis=1)
        mu_i_repeated = np.repeat(mu_i, N, axis=1)

        term_1 = X.transpose() - mu_i_repeated
        term_2 = np.repeat(np.expand_dims(Z[:,i], axis=1), D, axis=1) * term_1.transpose()
        s[i,:,:] = np.matmul(term_1, term_2)/np.sum(Z[:,i])
        #########################################
        # Write your code here
        # Suggest ways of overcoming the singularity
        s[1,:,:] += 0.001 * np.identity(D)
        ########################################/
        p[i] = np.mean(Z[:,i])
    ax1.clear()
    # plot the samples of the dataset, belonging to the chosen phoneme (f1, f2, f1+f2 | phoneme 1 or 2)
    plot_data_3D(X=X, title_string=title_string, ax=ax1)
    # Plot gaussians after each iteration
    plot_gaussians(ax1, 2*s, mu)
print('\nFinished.\n')

print('Implemented GMM | Mean values')
for i in range(k):
    print(mu[i])
print('Implemented GMM | Covariances')
for i in range(k):
    print(s[i,:,:])
print('Implemented GMM | Weights')
print(p)
print('')

# enter non-interactive mode of matplotlib, to keep figures open
plt.ioff()
plt.show()
