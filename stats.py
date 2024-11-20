from knn import *
from accuracy import *

import matplotlib.pyplot as plt

def stats(tr_feat, tr_cl, test_feat, test_cl, k, cl):

    for i in k:
        pred = knn_class(tr_feat, tr_cl, test_feat, i, cl)
        tp, fp, fn, tn = accuracy_class(pred, test_cl, cl)
        stats = []
        sensitivity = tp / (tp + fn)
        stats.append(sensitivity)
        specifity = tn / (tn + fp)
        stats.append(specifity)
        precision = tp / (tp + fp)
        stats.append(precision)
        f1_score = (2 * precision * sensitivity) / (precision + sensitivity)
        stats.append(f1_score)
    
    return stats

import matplotlib.pyplot as plt

def plot_stats(matrix):

    matr = [[round(el, 2) for el in riga] for riga in matrix]

    row_labels = ["Class 1", "Class 2", "Class 3"]
    col_labels = ["Sensitivity", "Specifity", "Precision", "F1_score"]

    fig, ax = plt.subplots(figsize=(len(col_labels) * 1.5, len(row_labels) * 0.8))
    ax.axis("tight")
    ax.axis("off")

    # Creating the table
    tabella = ax.table(
        cellText=matr,
        rowLabels=row_labels,
        colLabels=col_labels,
        loc="center",
        cellLoc="center",
    )

    # Lables of rows and columns 
    for (i, j), cell in tabella.get_celld().items():
        if i == 0 or j == -1:  # Etichette di intestazione o righe
            cell.set_text_props(color="black")
            cell.set_facecolor("#fffacd")  # Giallo chiaro

    tabella.auto_set_font_size(False)
    tabella.set_fontsize(12)
    tabella.auto_set_column_width(col_labels)
    ax.set_title("Statistics from the analysis", fontsize=14, weight="bold", pad=20)

    plt.show()

