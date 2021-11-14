import numpy as np

def load_data_ex1():
    # loads the data for excercise 1
    
    # read our data from a text file
    data = np.loadtxt("ex1data1.txt", comments="#", delimiter=",", unpack=False)

    # load from the first column into X
    X = data[:, 0]
    X = np.expand_dims(X, axis=1)
    # load from the second column into y
    y = data[:, 1]

    # create column of ones to be appended to X
    column_of_ones = np.ones((X.shape[0],1))
    # append ones to the dimension of columns (i.e., 1)
    X = np.append(column_of_ones, X, axis=1)
    
    return X, y
