from knn import *
from accuracy import *
import matplotlib.pyplot as plt

def stats(tr_feat, bin_tr, test_feat, bin_test, k):

    for i in k:
        pred = knn(tr_feat, bin_tr, test_feat, i)
        tp, fp, fn, tn = accuracy_class(pred, bin_test)
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

def compute_av_stats(st):

    sens, spec, prec, f1 = 0, 0, 0, 0
    for i in range(len(st)):
        sens += st[i][0]
        spec += st[i][1]
        prec += st[i][2]
        f1 += st[i][3]

    return [sens/len(st), spec/len(st), prec/len(st), f1/len(st)]

def plot_stats(matrix, str):

    matr = [[round(el, 2) for el in riga] for riga in matrix]

    row_labels = ["Class 0", "Class 1", "Class 2"]
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
    ax.set_title(f"{str} from the analysis", fontsize=14, weight="bold", pad=20)

    plt.show()

