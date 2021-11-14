import matplotlib.pyplot as plt
import os

def plot_cost(cost):
    
    fig, ax1 = plt.subplots()
    ax1.set_xlabel('Iterations')
    ax1.set_ylabel('Cost')
    plt.plot(cost)
    fig.tight_layout()
    plot_filename = os.path.join(os.getcwd(), 'figures', 'cost.png')
    plt.savefig(plot_filename)
    plt.show()
