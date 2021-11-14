from load_data_ex1 import *
from gradient_descent import *
import os

figures_folder = os.path.join(os.getcwd(), 'figures')
if not os.path.exists(figures_folder):
    os.makedirs(figures_folder, exist_ok=True)

# This loads our data
X, y = load_data_ex1()

# initialise trainable parameters theta, set learning rate alpha and number of iterations
theta = np.zeros((2, 1))
alpha = 1
iterations = 50

# do plotting
do_plot = True

# run gradient descent
t = gradient_descent(X, y, theta, alpha, iterations, do_plot)
