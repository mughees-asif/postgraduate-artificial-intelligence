import numpy as np

def sigmoid(z):
    
    output = 0.0
    #########################################
    # Write your code here
    # modify this to return z passed through the sigmoid function
    output = 1 / (1 + np.exp(-z))
    ########################################/
    
    return output
