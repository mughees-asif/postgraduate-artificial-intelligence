import matplotlib.pyplot as plt
import os

def plot_cost_train_test(train_cost, test_cost, ax1):
    
    ax1.set_xlabel('Iterations')
    ax1.set_ylabel('Cost')
    ax1.plot(train_cost, c='red', label='train cost')
    ax1.plot(test_cost, c='blue', label='test cost')
    
    # add legend to the subplot
    ax1.legend()
