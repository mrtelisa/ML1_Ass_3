from values import *
from knn import *
from accuracy import *
import matplotlib.pyplot as plt

# Calculating the accuracy for each class, obtaining tp, fp, fn, tn
def accuracy_class(pred, test_cl):

    tp, fp, fn, tn = 0, 0, 0, 0

    for i in range(len(pred)):
        if(pred[i] == 1):
            if(test_cl[i] == 1):
                tp += 1
            else:
                fp += 1
        else:
            if(test_cl[i] == 1):
                fn += 1
            else:
                tn += 1

    return tp, fp, fn, tn



# Computing statistics in different cases
def stats(tr_feat, bin_tr, test_feat, bin_test, k):

    matr = []
    for i in k:
        stats = []
        pred = knn(tr_feat, bin_tr, test_feat, i)
        tp, fp, fn, tn = accuracy_class(pred, bin_test)
        sensitivity = tp / (tp + fn)
        stats.append(sensitivity)
        specifity = tn / (tn + fp)
        stats.append(specifity)
        precision = tp / (tp + fp)
        stats.append(precision)
        f1_score = (2 * precision * sensitivity) / (precision + sensitivity)
        stats.append(f1_score)
        stats.append(calculate_accuracy(pred, bin_test))
        matr.append(stats)
    
        #print(f"matr for k = {i}:\n", matr)
    return matr[0]



def compute_stat_stat(st):

    matr = []
    matr.append(compute_av_stats(st))
    matr.append(compute_median(st))
    matr.append(compute_mode(st))
    stand = np.std(st, axis=0)
    perc25 = np.percentile(st, 25, axis=0)
    perc75 = np.percentile(st, 75, axis=0)
    
    matr.append(stand.tolist())
    matr.append(perc25.tolist())
    matr.append(perc75.tolist())

    return matr

def compute_acc_class(vec):
    st = []
    st.append(compute_average(vec))
    stand = np.std(vec)
    st.append(stand.tolist())

    return st

def compute_acc_k(matr):
    tran = list(map(list, zip(*matr)))
    st = []
    for i in range(len(tran)):
        a = []
        a.append(compute_average(tran[i]))
        a.append(np.std(tran[i]))
        st.append(a)

    return st



def plot_table(data, row_labels, tit=None):
    row_labels = row_labels
    column_labels = ["Average", "Std_dev"]

    fig, ax = plt.subplots()
    ax.axis('tight')
    ax.axis('off')

    data_rounded = [[round(value, 4) if isinstance(value, (int, float)) else value for value in row] for row in data]

    table = plt.table(
        cellText=data_rounded, 
        rowLabels=row_labels, 
        colLabels=column_labels, 
        loc='center', 
        cellLoc='center'
    )
    
    for (i, j), cell in table.get_celld().items():
        if i == 0 or j == -1: 
            cell.set_text_props(color="black")
            cell.set_facecolor("#fffacd")  

    table.auto_set_font_size(False)
    table.set_fontsize(10)
    for col in range(len(column_labels)):
        table.auto_set_column_width([col])
    ax.set_title(tit, fontsize=14, weight="bold", pad=20)

    plt.show()

