import numpy as np

def normalize_features(X):
    """
        :param X : two-dimensional array of our dataset, with shape NxD. N is the number of rows (each row is a sample) and D is the number of columns
    """
    # Normalize our features by subtracting the mean and
    # dividing by the standard deviation
    
    # get the number of samples in the dataset
    N = X.shape[0]
    
    # get the mean for every column
    mean_vector = np.mean(X, axis=0)
    # get the std for every column
    std_vector = np.std(X, axis=0)
    
    
    # insert extra singleton dimension, to obtain 1xD shape
    mean_vector = np.expand_dims(mean_vector, axis=0)
    std_vector = np.expand_dims(std_vector, axis=0)
    # repeat N times across first dimension, to obtain NxD shape
    repeated_mean = np.repeat(mean_vector, N, axis=0)
    repeated_std = np.repeat(std_vector, N, axis=0)
    
    #if np.sum(1.0*(repeated_std==0))>0:
    if np.any(repeated_std==0):
        print('Adding epsilon to avoid division by zero during normalization')
        repeated_std += np.finfo(float).eps
    # subtract column mean and divide each element by column standard deviation
    X_normalized = (X - repeated_mean) / repeated_std
    print('Dataset normalization complete.')
    
    return X_normalized, mean_vector, std_vector
