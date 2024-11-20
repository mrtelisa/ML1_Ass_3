import matplotlib.pyplot as plt
import numpy as np

def plot_confusion_matr(matrices, class_label, k_values):
  
    num_matrices = len(matrices)

    # Create the subplot grid
    rows = int(np.ceil(np.sqrt(num_matrices)))
    cols = int(np.ceil(num_matrices / rows))

    fig, axes = plt.subplots(rows, cols, figsize=(15, 10))
    axes = axes.flatten()

    # Define colors
    label_color = "#FFFACD"  
    diagonal_color = "#ADD8E6"  
    off_diagonal_color = "#E0FFFF"  

    for i, (matrix, k) in enumerate(zip(matrices, k_values)):
        ax = axes[i]
        ax.axis("off")  # Turn off the axes

        # Define the table data
        table_data = [
            ["", "Predicted Positive", "Predicted Negative"],
            ["Actual Positive", matrix[0], matrix[2]],  # TP, FN
            ["Actual Negative", matrix[1], matrix[3]]   # FP, TN
        ]

        # Create the cell colors array
        cell_colors = [["white"] * len(row) for row in table_data]
        cell_colors[0][1:] = [label_color, label_color]  
        cell_colors[1][0] = label_color  
        cell_colors[2][0] = label_color  
        # Set colors for the confusion matrix values
        cell_colors[1][1] = diagonal_color  # TP
        cell_colors[1][2] = off_diagonal_color  # FN
        cell_colors[2][1] = off_diagonal_color  # FP
        cell_colors[2][2] = diagonal_color  # TN

        # Create the table with colored cells
        table = ax.table(cellText=table_data, loc="center", cellLoc="center", cellColours=cell_colors)
        table.scale(1, 2)

        # Remove the border of the top-left cell
        cell = table[0, 0]
        cell.set_edgecolor("white")
        cell.set_facecolor("white")

        ax.set_title(f"Confusion matrix of class {class_label} with k = {k}", pad=20)

    # Hide any unused subplots
    for j in range(len(matrices), len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.show()
