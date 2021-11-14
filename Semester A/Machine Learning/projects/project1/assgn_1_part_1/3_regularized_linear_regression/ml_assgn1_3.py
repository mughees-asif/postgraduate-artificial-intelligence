import numpy as np
from gradient_descent import *
import os


def main():
    figures_folder = os.path.join(os.getcwd(), 'figures')
    if not os.path.exists(figures_folder):
        os.makedirs(figures_folder, exist_ok=True)

    # load input data
    X_original = np.array([-0.99768, -0.69574, -0.40373, -0.10236, 0.22024, 0.47742, 0.82229])
    # we'll use variable "X" to store the bias column and the columns of the higher order terms
    X = X_original.copy()
    X = np.expand_dims(X, axis=1)

    # load output data
    y = np.array([2.0885, 1.1646, 0.3287, 0.46013, 0.44808, 0.10013, -0.32952])

    # insert the bias column of ones, into the input data
    column_of_ones = np.ones((X.shape[0], 1))
    # append column to the dimension of columns (i.e., 1)
    X = np.append(column_of_ones, X, axis=1)

    # perform a polynomial expansion to the fifth order
    for j in range(2, 6):
        new_column = X[:, 1] ** j
        new_column = np.expand_dims(new_column, axis=1)
        X = np.append(X, new_column, axis=1)

    # initialise trainable parameters theta, set learning rate alpha, regularization parameter l and
    # number of iterations
    theta = np.zeros((6))
    alpha = 0.02
    l = 5
    iterations = 200

    # plot predictions for every iteration?
    do_plot = True

    # call the gradient descent function to obtain the trained parameters theta_final
    # you will need to modify the gradient_descent function to accept an additional argument lambda (l)
    theta_final = gradient_descent(X, y, theta, alpha, iterations, do_plot, l)


if __name__ == '__main__':
    main()
