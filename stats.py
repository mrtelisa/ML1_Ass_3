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


