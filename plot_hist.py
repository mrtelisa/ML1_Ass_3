import matplotlib.pyplot as plt
import numpy as np


def plot_hist(matr, cl, k):

    num_histograms = len(matr)

    # Creating the subplot gird
    rows = int(np.ceil(np.sqrt(num_histograms)))  
    cols = int(np.ceil(num_histograms / rows))  

    # Creating the subplots
    fig, axes = plt.subplots(rows, cols, figsize=(12, 8))

    # Appiattire gli assi per gestire la griglia
    axes = axes.flatten()

    # Plotting each list
    for i, data in enumerate(matr):
        labels = ["TP", "FP", "FN", "TN"]  
        axes[i].bar(labels, data, color=np.random.rand(3,)) 
        axes[i].set_title(f'Confusion Matrix of class {cl} with k = {k[i]}')
        axes[i].set_ylabel('cardinality')

    # Hide possible extra-subplot
    for j in range(num_histograms, len(axes)):
        fig.delaxes(axes[j])

    # Emproving the layout
    plt.tight_layout()
    plt.show()