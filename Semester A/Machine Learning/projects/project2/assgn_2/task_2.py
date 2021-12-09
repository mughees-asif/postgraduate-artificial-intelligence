import numpy as np
import os
import matplotlib.pyplot as plt
from print_values import *
from plot_data_all_phonemes import *
from plot_data import *
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

# Initialize array containing f1 & f2, of all phonemes.
X_full = np.zeros((len(f1), 2))
#########################################
# Write your code here
# Store f1 in the first column of X_full, and f2 in the second column of X_full
X_full[:, 0] = f1
X_full[:, 1] = f2
########################################/
X_full = X_full.astype(np.float32)

# We will train a GMM with k components, on a selected phoneme id which is stored in variable "p_id" 

# number of GMM components
k = 6
# you can use the p_id variable, to store the ID of the chosen phoneme that will be used (e.g. phoneme 1, or phoneme 2)
p_id = 1

#########################################
# Write your code here

# Create an array named "X_phoneme", containing only samples that belong to the chosen phoneme.
# The shape of X_phoneme will be two-dimensional. Each row will represent a sample of the dataset,
# and each column will represent a feature (e.g. f1 or f2)
# Fill X_phoneme with the samples of X_full that belong to the chosen phoneme
# To fill X_phoneme, you can leverage the phoneme_id array, that contains the ID of each sample of X_full

X_phoneme = X_full[phoneme_id == p_id, :]

########################################/

# Plot array containing the chosen phoneme

# Create a figure and a subplot
fig, ax1 = plt.subplots()

title_string = 'Phoneme {}'.format(p_id)
# plot the samples of the dataset, belonging to the chosen phoneme (f1 & f2, phoneme 1 or 2)
plot_data(X=X_phoneme, title_string=title_string, ax=ax1)
# save the plotted points of phoneme 1 as a figure
plot_filename = os.path.join(os.getcwd(), 'figures', 'dataset_phoneme_{}.png'.format(p_id))
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
n_iter = 100

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
    print('Iteration {:03}/{:03}'.format(t+1, n_iter))

    # Do the E-step
    Z = get_predictions(mu, s, p, X)
    Z = normalize(Z, axis=1, norm='l1')

    # Do the M-step:
    for i in range(k):
        mu[i,:] = np.matmul(X.transpose(),Z[:,i]) / np.sum(Z[:,i])
        # We will fit Gaussians with diagonal covariance matrices
        mu_i = mu[i,:]
        mu_i = np.expand_dims(mu_i, axis=1)
        mu_i_repeated = np.repeat(mu_i, N, axis=1)
        X_minus_mu = (X.transpose() - mu_i_repeated)**2
        res_1 = np.squeeze( np.matmul(X_minus_mu, np.expand_dims(Z[:,i], axis=1)))/np.sum(Z[:,i])
        s[i,:,:] = np.diag(res_1)
        p[i] = np.mean(Z[:,i])
    ax1.clear()
    # plot the samples of the dataset, belonging to the chosen phoneme (f1 & f2, phoneme 1 or 2)
    plot_data(X=X_phoneme, title_string=title_string, ax=ax1)
    # Plot gaussians after each iteration
    plot_gaussians(ax1, 2*s, mu)
print('\nFinished.\n')

# save the trained GMM's plot as a figure
plot_filename = os.path.join(os.getcwd(), 'figures', 'GMM_phoneme_{}_k_{}.png'.format(p_id, k))
plt.savefig(plot_filename)

print('Implemented GMM | Mean values')
for i in range(k):
    print(mu[i])
print('')
print('Implemented GMM | Covariances')
for i in range(k):
    print(s[i,:,:])
print('')
print('Implemented GMM | Weights')
print(p)
print('')

# enter non-interactive mode of matplotlib, to keep figures open
plt.ioff()
plt.show()

# Create a dictionary to store the trained GMM's parameters
GMM_parameters = {}
GMM_parameters['mu'] = mu
GMM_parameters['s'] = s
GMM_parameters['p'] = p

# Save the trained GMM's parameters in a numpy file
npy_filename = 'data/GMM_params_phoneme_02_k_06.npy'
np.save(npy_filename, GMM_parameters)
