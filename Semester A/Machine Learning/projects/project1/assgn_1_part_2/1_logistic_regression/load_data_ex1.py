import numpy as np

def load_data_ex1():
    #loads data for exercise 2
    
    # read our input data
    X = np.loadtxt("ex4x.dat", comments="#", delimiter=",", unpack=False)
    
    # read groundtruth labels
    y = np.loadtxt("ex4y.dat", comments="#", delimiter=",", unpack=False)
    
    return X, y
