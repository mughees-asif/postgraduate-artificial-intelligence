from load_data_ex1 import *
from normalize_features import *
from gradient_descent import *
from plot_data_function import *
from plot_boundary import *
import matplotlib.pyplot as plt
from plot_sigmoid import *
from return_test_set import *
from compute_cost import *
import os

figures_folder = os.path.join(os.getcwd(), 'figures')
if not os.path.exists(figures_folder):
    os.makedirs(figures_folder, exist_ok=True)

# this loads our data
X, y = load_data_ex1()

# split the dataset into training and test set, using random shuffling
train_samples = 20
X_train, y_train, X_test, y_test = return_test_set(X, y, train_samples)

# Compute mean and std on train set
# Normalize both train and test set using these mean and std values
X_train_normalized, mean_vec, std_vec = normalize_features(X_train)
X_test_normalized = normalize_features(X_test, mean_vec, std_vec)

# After normalizing, we append a column of ones to X_normalized, as the bias term
# We append the column to the dimension of columns (i.e., 1)
# We do this for both train and test set
column_of_ones = np.ones((X_train_normalized.shape[0], 1))
X_train_normalized = np.append(column_of_ones, X_train_normalized, axis=1)
column_of_ones = np.ones((X_test_normalized.shape[0], 1))
X_test_normalized = np.append(column_of_ones, X_test_normalized, axis=1)

# initialise trainable parameters theta, set learning rate alpha and number of iterations
theta = np.zeros((3))
alpha = 1.0
iterations = 100

# call the gradient descent function to obtain the trained parameters theta_final and the cost vector
theta_final, cost_vector = gradient_descent(X_train_normalized, y_train, theta, alpha, iterations)

###################################################
# Train set 

# Plot the cost for all iterations
fig, ax1 = plt.subplots()
plot_cost(cost_vector, ax1)
plot_filename = os.path.join(os.getcwd(), 'figures', 'ex2_train_cost.png')
plt.savefig(plot_filename)
min_cost = np.min(cost_vector)
argmin_cost = np.argmin(cost_vector)
print('Final training cost: {:.5f}'.format(cost_vector[-1]))
print('Minimum training cost: {:.5f}, on iteration #{}'.format(min_cost, argmin_cost+1))

# plot our data and decision boundary
fig, ax1 = plt.subplots()
ax1 = plot_data_function(X_train_normalized, y_train, ax1)
plot_boundary(X_train_normalized, theta_final, ax1)
# save the plotted decision boundary as a figure
plot_filename = os.path.join(os.getcwd(), 'figures', 'ex2_decision_boundary_train.png')
plt.savefig(plot_filename)

###################################################
# Test set

cost_test = compute_cost(X_test_normalized, y_test, theta_final)
print('Final test cost: {:.5f}'.format(cost_test))

fig, ax1 = plt.subplots()
ax1 = plot_data_function(X_test_normalized, y_test, ax1)
plot_boundary(X_test_normalized, theta_final, ax1)
# save the plotted decision boundary as a figure
plot_filename = os.path.join(os.getcwd(), 'figures', 'ex2_decision_boundary_test.png')
plt.savefig(plot_filename)

# enter non-interactive mode of matplotlib, to keep figures open
plt.ioff()
plt.show()
