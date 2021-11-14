import numpy as np
from compute_cost import *
from compute_cost_regularised import *
from plot_hypothesis import *
from plot_sampled_points import *
from plot_cost import *
from calculate_hypothesis import *


def gradient_descent(X, y, theta, alpha, iterations, do_plot, l):
    """
        :param X            : 2D array of our dataset
        :param y            : 1D array of the groundtruth labels of the dataset
        :param theta        : 1D array of the trainable parameters
        :param alpha        : scalar, learning rate
        :param iterations   : scalar, number of gradient descent iterations
        :param do_plot      : boolean, used to plot groundtruth & prediction values during the gradient descent iterations
    """

    # Create just a figure and two subplots.
    # The first subplot (ax1) will be used to plot the predictions on the given 7 values of the dataset
    # The second subplot (ax2) will be used to plot the predictions on a densely sampled space, to get a more smooth curve
    fig, (ax1, ax2) = plt.subplots(1, 2)
    if do_plot == True:
        plot_hypothesis(X, y, theta, ax1)

    m = X.shape[0]  # the number of training samples is the number of rows of array X
    cost_vector = np.array([], dtype=np.float32)  # empty array to store the cost for every iteration

    # Gradient Descent loop
    for it in range(iterations):

        # initialize temporary theta, as a copy of the existing theta array
        theta_temp = theta.copy()

        sigma = np.zeros((len(theta)))
        for i in range(m):
            #########################################
            # Write your code here
            # Calculate the hypothesis for the i-th sample of X, with a call to the "calculate_hypothesis" function
            hypothesis = calculate_hypothesis(X, theta, i)
            ########################################/
            output = y[i]
            #########################################
            # Write your code here
            # Adapt the code, to compute the values of sigma for all the elements of theta
            sigma = sigma + (hypothesis - output) * X[i]
            ########################################/

        # update theta_temp
        #########################################
        # Write your code here
        # Update theta_temp, using the values of sigma
        # Make sure to use lambda, if necessary
        bias = theta_temp[0]
        theta_temp = theta_temp * (1 - (alpha * (l / m))) - (alpha / m) * sigma
        theta_temp[0] = bias - (alpha / m) * sigma[0]
        ########################################/

        # copy theta_temp to theta
        theta = theta_temp.copy()

        # append current iteration's cost to cost_vector
        iteration_cost = compute_cost_regularised(X, y, theta, l)
        cost_vector = np.append(cost_vector, iteration_cost)

        # plot predictions for current iteration
        if do_plot == True:
            plot_hypothesis(X, y, theta, ax1)
    ####################################################

    # plot predictions on the dataset's points using the final parameters
    plot_hypothesis(X, y, theta, ax1)
    # sample 1000 points, from -1.0 to +1.0
    X_sampled = np.linspace(-1.0, 1.0, 1000)
    # plot predictions on the sampled points using the final parameters
    plot_sampled_points(X, y, X_sampled, theta, ax2)

    # save the predictions as a figure
    plot_filename = os.path.join(os.getcwd(), 'figures', 'predictions.png')
    plt.savefig(plot_filename)
    print('Gradient descent finished.')

    # Plot the cost for all iterations
    plot_cost(cost_vector)
    min_cost = np.min(cost_vector)
    argmin_cost = np.argmin(cost_vector)
    print('Minimum cost: {:.5f}, on iteration #{}'.format(min_cost, argmin_cost + 1))

    return theta
