import numpy as np

def load_data_ex2():
    # loads the data for excercise 2
    
    # read our data from a text file
    data = np.loadtxt("ex1data2.txt", comments="#", delimiter=",", unpack=False)

    # load the first two columns into X
    X = data[:, :2]
    
    # load from the third column into y
    y = data[:, 2]
    
    return X, y
