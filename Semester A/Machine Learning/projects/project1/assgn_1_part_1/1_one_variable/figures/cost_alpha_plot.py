import matplotlib.pyplot as plt

fig = plt.figure(1)  # identifies the figure
plt.title(r'Overall effect of $\alpha$ on the cost', fontsize='13', fontweight='bold')  # title
plt.plot([1e-6, 1e-4, 1e-2, 0, 1.00, 1.2, 1.4, 1.6],
         [31.85851, 17.36882, 5.67829, 32.07273, 172570.09522, 249519.29410, 340615.69111, 445859.28625])  # plot the points
plt.ticklabel_format(style='sci', axis='both', scilimits=(0,0))
plt.xlabel(r'Learning rate $\alpha$', fontsize='13')  # adds a label in the x axis
plt.ylabel("Cost", fontsize='13')  # adds a label in the y axis
plt.savefig('a_cost.png')  # saves the figure in the present directory
plt.grid()  # shows a grid under the plot
plt.show()


