import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

def plot_cm(matrix, ax, k):

    conf_matrix = np.array([
        [matrix[0], matrix[1]],
        [matrix[2], matrix[3]]
    ])

    labels = ["True Pos", "False Pos"]
    ticks = ["False Neg", "True Neg"]

    # Plot the confusion matrix on the provided `ax`
    custom_cmap = LinearSegmentedColormap.from_list("custom_blues", ["#FFFFFF", "#66B2FF", "#457b9d"])
    cax = ax.matshow(conf_matrix, cmap = custom_cmap)

    # Add values inside the matrix with reduced font size
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(conf_matrix[i, j]), va = 'center', ha = 'center', color = "black", fontsize = 8)

    # Customize the axes with reduced font size
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(labels, rotation = 0, ha = "center", fontsize = 8)
    ax.set_yticklabels(ticks, rotation = 90, ha = "center", fontsize = 8)
    ax.tick_params(axis = 'y', pad = 15)
    ax.set_title(f"k = {k}", fontsize = 10, pad = 3)

    
    # Add a colorbar to the parent figure
    if hasattr(ax, 'cax'):
        ax.figure.colorbar(cax, ax = ax, fraction = 0.046, pad = 0.04)



def plot_conf_matr(matr, cl, k):
    # Create a 3x3 grid of subplots
    
    fig, axes = plt.subplots(3, 3, figsize = (10, 10))  # Reduced figure size for compactness
    fig.suptitle(f"Confusion Matrices of {cl} class", fontsize = 14, y = 0.98)

    # Plot each confusion matrix in the corresponding subplot
    for i, ax in enumerate(axes.flat):
        if i < len(matr):
            plot_cm(matr[i], ax, k[i])
        else:
            # Hide unused subplots
            ax.axis('off')

    # Adjust spacing to bring plots closer together
    plt.subplots_adjust(wspace = 0.08, hspace = 0.3)  # Reduce horizontal and vertical spacing
    
    # Show the figure
    plt.show()
