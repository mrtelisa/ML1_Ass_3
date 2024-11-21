from split_dataframe import *
from knn import *
from class_analysis import *
from stats import *
from compute_binary import *

def average_stats(classes, feat, k, cl, iter):

    statistics = []
    statistics_0 = []
    statistics_1 = []
    statistics_2 = []

    for i in range(iter):
        tr_cl, tr_feat, test_cl, test_feat = split_matrix_random(classes, feat)

        binary_tr = []
        binary_test =  []
        binary_tr = compute_binary(tr_cl, cl)
        binary_test = compute_binary(test_cl, cl)

        matr = []
        for i in range(len(cl)):
            matr.append(class_analysis(tr_feat, binary_tr[i], test_feat, binary_test[i], k))

        statistics_0.append(stats(tr_feat, binary_tr[0], test_feat, binary_test[0], k))
        statistics_1.append(stats(tr_feat, binary_tr[1], test_feat, binary_test[1], k))
        statistics_2.append(stats(tr_feat, binary_tr[2], test_feat, binary_test[2], k))
    
    statistics.append(compute_av_stats(statistics_0))
    statistics.append(compute_av_stats(statistics_1))
    statistics.append(compute_av_stats(statistics_2))

    return statistics