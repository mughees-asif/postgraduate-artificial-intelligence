from calculate_hypothesis import *


def compute_cost(X, y, theta):
    """
        :param X            : 2D array of our dataset
        :param y            : 1D array of the groundtruth labels of the dataset
        :param theta        : 1D array of the trainable parameters
    """

    # initialize cost
    J = 0.0
    # get number of training examples
    m = y.shape[0]

    # Compute cost for logistic regression.
    for i in range(m):
        hypothesis = calculate_hypothesis(X, theta, i)
        output = y[i]
        #########################################
        # Write your code here
        # You must calculate the cost
        cost = (1 / m) * (-output * np.log(hypothesis) - (1 - output) * np.log(1 - hypothesis))
        #########################################

        J += cost

    return J
