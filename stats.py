from knn import *
from accuracy import *

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