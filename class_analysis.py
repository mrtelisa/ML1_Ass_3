from knn import *
from accuracy import *
from plot_hist import *

def class_analysis(tr_feat, tr_cl, test_feat, test_cl, k, cl):

    matr = []
    for i in k:
        pred = knn_class(tr_feat, tr_cl, test_feat, i, cl)
        tp1, fp1, fn1, tn1 = accuracy_class(pred, test_cl, cl)
        matr.append([tp1, fp1, fn1, tn1])
    
    return matr