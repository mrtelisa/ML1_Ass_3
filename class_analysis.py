from knn import *
from accuracy import *

# Analysis for each class
def class_analysis(tr_feat, bin_tr, test_feat, bin_test, k):

    matr = []
    for i in k:
        pred = knn(tr_feat, bin_tr, test_feat, i)
        tp, fp, fn, tn = accuracy_class(pred, bin_test)
        matr.append([tp, fp, fn, tn])
    
    return matr