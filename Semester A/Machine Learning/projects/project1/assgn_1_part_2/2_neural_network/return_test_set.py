import numpy as np

def return_test_set(X, y):
    """
        :param X                : 2D array of our dataset
        :param y                : 1D array of the groundtruth labels of the dataset
    """
    
    # total number of samples in the dataset
    N = X.shape[0]
    
    indices_all = list(np.arange(N))
    
    # Create train set's indices and test set's indices
    # Train set will have samples 1-25, 51-75, 101-125
    # Test set will have the rest samples
    
    indices_train = []
    indices_test = []
    
    for i in range(0, 25):
        indices_train.append(i)
    for i in range(50, 75):
        indices_train.append(i)
    for i in range(100, 125):
        indices_train.append(i)
    
    indices_test = [x for x in indices_all if x not in indices_train]
    
    indices_train = np.array(indices_train)
    indices_test = np.array(indices_test)
    
    X_train = X[indices_train,:]
    y_train = y[indices_train]
    
    X_test = X[indices_test,:]
    y_test = y[indices_test]
    
    return X_train, y_train, X_test, y_test
