import numpy as np

def load_data_ex1():
    #loads data
    
    # read our input data
    data = np.loadtxt("iris.dat", comments="#", delimiter=",", unpack=False)
    
    X = data[:,:4]
    target = data[:,4].astype(int)
    
    N = X.shape[0]
    
    y = np.zeros((N, 3))
    
    for t in range(N):
        y[t,target[t]] = 1
        
    return X, y
