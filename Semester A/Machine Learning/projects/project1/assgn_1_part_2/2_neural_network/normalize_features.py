import numpy as np

def normalize_features(X, mean_vector=None, std_vector=None):
    """
        :param X            : two-dimensional array of our dataset, with shape NxD. N is the number of rows (each row is a sample) and D is the number of columns
        :param mean_vector  : optionally given, precomputed vector of mean values.
        :param std_vector   : optionally given, precomputed vector of standard deviation values.
        
        If the function is called passing *only X* as input argument, then the mean and std values will be computed from X, and then X will be normalized using these values.
        If the function is called passing *all three* input arguments, X array will be normalized using the precomputed mean and std values that were passed as input arguments.
        
        Normalization is performed by subtracting the mean and dividing by the standard deviation: x_normalized = (x - x_mean)/(x_std+ε)
    """
    
    if (mean_vector is None) and (std_vector is None):
        return_mean_std = True
        # get the mean for every column
        mean_vector = np.mean(X, axis=0)
        # get the std for every column
        std_vector = np.std(X, axis=0)
        
        # insert extra singleton dimension, to obtain 1xD shape
        mean_vector = np.expand_dims(mean_vector, axis=0)
        std_vector = np.expand_dims(std_vector, axis=0)
    else:
        return_mean_std = False
    
    # get the number of samples in the dataset
    N = X.shape[0]
    
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
    
    if return_mean_std==True:
        return X_normalized, mean_vector, std_vector
    else:
        return X_normalized
    
    
