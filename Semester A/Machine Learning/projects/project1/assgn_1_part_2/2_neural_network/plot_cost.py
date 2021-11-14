import matplotlib.pyplot as plt
import os

def plot_cost(cost, ax1):
    
    ax1.set_xlabel('Iterations')
    ax1.set_ylabel('Cost')
    ax1.plot(cost)
