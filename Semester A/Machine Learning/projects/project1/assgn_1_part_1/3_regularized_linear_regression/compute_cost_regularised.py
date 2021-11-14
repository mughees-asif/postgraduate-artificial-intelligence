from calculate_hypothesis import *


def compute_cost_regularised(X, y, theta, l):
    """
    :param X        : 2D array of our dataset
    :param y        : 1D array of the groundtruth labels of the dataset
    :param theta    : 1D array of the trainable parameters
    :param l        : scalar, regularization parameter
    """

    # initialize costs
    total_squared_error = 0.0
    total_regularised_error = 0.0

    # get number of training examples
    m = y.shape[0]

    for i in range(m):
        hypothesis = calculate_hypothesis(X, theta, i)
        output = y[i]
        squared_error = (hypothesis - output) ** 2
        total_squared_error += squared_error

    for i in range(1, len(theta)):
        total_regularised_error += theta[i] ** 2

    j = (total_squared_error + l * total_regularised_error) / (2 * m)

    return j
