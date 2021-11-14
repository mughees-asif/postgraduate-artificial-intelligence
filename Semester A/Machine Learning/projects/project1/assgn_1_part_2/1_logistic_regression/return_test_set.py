import numpy as np

def return_test_set(X, y, train_samples):
    """
        :param X                : 2D array of our dataset
        :param y                : 1D array of the groundtruth labels of the dataset
        :param train_samples    : the number of samples to keep as training set. the rest samples will fill the test set.
    """
    
    # total number of samples in the dataset
    N = X.shape[0]
    
    # create a list of random indices, from 1 to N
    indices_all = np.random.permutation(N)
    # split these random indices, to train set's indices and test set's indices
    indices_train = indices_all[:train_samples]
    indices_test = indices_all[train_samples:]
    
    training_input = X[indices_train,:]
    training_output = y[indices_train]

    test_input = X[indices_test,:]
    test_output = y[indices_test]
    
    return training_input, training_output, test_input, test_output
