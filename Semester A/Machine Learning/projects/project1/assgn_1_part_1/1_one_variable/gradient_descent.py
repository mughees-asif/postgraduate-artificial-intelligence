import numpy as np
from compute_cost import *
from plot_hypothesis import *
from plot_cost import *

def gradient_descent(X, y, theta, alpha, iterations, do_plot):
    """
        :param X            : 2D array of our dataset
        :param y            : 1D array of the groundtruth labels of the dataset
        :param theta        : 1D array of the trainable parameters
        :param alpha        : scalar, learning rate
        :param iterations   : scalar, number of gradient descent iterations
        :param do_plot      : boolean, used to plot groundtruth & prediction values during the gradient descent iterations
    """
    
    # Create just a figure and only one subplot
    fig, ax1 = plt.subplots()
    if do_plot==True:
        plot_hypothesis(X, y, theta, ax1)

    m = X.shape[0] # the number of training samples is the number of rows of array X
    cost_vector = np.array([], dtype=np.float32) # empty array to store the cost for every iteration
    
    # Gradient Descent
    for it in range(iterations):
        # get temporary variables for theta's parameters
        theta_0 = theta[0]
        theta_1 = theta[1]

        # update temporary variable for theta_0
        sigma = 0.0
        for i in range(m):
            #########################################
            # Write your code here
            # Replace the above line that calculates the hypothesis, with a call to the "calculate_hypothesis" function
            hypothesis = calculate_hypothesis(X, theta, i)
            ########################################/
            output = y[i]
            sigma = sigma + (hypothesis - output)
        theta_0 = theta_0 - (alpha/m) * sigma

        # update temporary variable for theta_1
        sigma = 0.0
        for i in range(m):
            #########################################
            # Write your code here
            # Replace the above line that calculates the hypothesis, with a call to the "calculate_hypothesis" function
            hypothesis = calculate_hypothesis(X, theta, i)
            ########################################/
            output = y[i]
            sigma = sigma + (hypothesis - output) * X[i, 1]
        theta_1 = theta_1 - (alpha/m) * sigma

        # update theta, using the temporary variables
        theta = np.array([theta_0, theta_1])

        # append current iteration's cost to cost_vector
        iteration_cost = compute_cost(X, y, theta)
        cost_vector = np.append(cost_vector, iteration_cost)
        
        # plot predictions for current iteration
        if do_plot==True:
            plot_hypothesis(X, y, theta, ax1)
    ####################################################
    
    # plot predictions using the final parameters
    plot_hypothesis(X, y, theta, ax1)
    # save the predictions as a figure
    plot_filename = os.path.join(os.getcwd(), 'figures', 'predictions.png')
    plt.savefig(plot_filename)
    print('Gradient descent finished.')

    # Plot the cost for all iterations
    plot_cost(cost_vector)
    min_cost = np.min(cost_vector)
    argmin_cost = np.argmin(cost_vector)
    print('Minimum cost: {:.5f}, on iteration #{}'.format(min_cost, argmin_cost+1))
    
    return theta
