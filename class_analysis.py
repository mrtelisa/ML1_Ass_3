from knn import *
from accuracy import *

def class_analysis(tr_feat, tr_cl, test_feat, test_cl, k, cl):

    matr = []
    for i in k:
        pred = knn_class(tr_feat, tr_cl, test_feat, i, cl)
        tp, fp, fn, tn = accuracy_class(pred, test_cl, cl)
        matr.append([tp, fp, fn, tn])
    
    return matr