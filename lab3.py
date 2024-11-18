import pandas as pd
import numpy as np
from split_dataframe import *
from normalization import *
from knn import *
from accuracy import *
from plot_hist import *


# Loading the dataset
matrix = np.loadtxt("wine_data.txt", delimiter=",")

# Dividing matrix between the class and the features + normalization of the features
feat = matrix[:, 1:]
cl = matrix[:, 0]
    #print(feat)
norm_feat = normalization(feat)
    #print(norm_feat)

# Splitting into training set and test set (divided in classes and features)
tr_cl, tr_feat, test_cl, test_feat = split_matrix_random(cl, norm_feat)
    #print(test_feat)

# Initializing the number k needed in kNN
results = []
k = [1, 2, 3, 4, 5, 10, 15, 20, 30, 40, 50]

for i in k:
    predictions = knn(tr_feat, tr_cl, test_feat, i)
        #print(predictions)
    accuracy = calculate_accuracy(predictions, test_cl)
        #print(accuracy)

k1 = [3, 4, 5, 10]

for i in k1:
    # Control if the class is 1
        # funct con dentro knn ma anche classe
    pred1 = knn_class(tr_feat, tr_cl, test_feat, i, 1)
    tp1, fp1, fn1, tn1 = accuracy_class(pred1, test_cl, 1)
    conf_matr1 = [tp1, fp1, fn1, tn1]
        #plot_hist(tp1, fp1, fn1, tn1, 1, i)
    
    # Control if the class is 2
    pred2 = knn_class(tr_feat, tr_cl, test_feat, i, 2)
    tp2, fp2, fn2, tn2 = accuracy_class(pred2, test_cl, 2)
    conf_matr2 = [tp2, fp2, fn2, tn2]
        #plot_hist(tp2, fp2, fn2, tn2, 2, i)
    
    # Control if the class is 3
    pred3 = knn_class(tr_feat, tr_cl, test_feat, i, 3)
    tp3, fp3, fn3, tn3 = accuracy_class(pred3, test_cl, 3)
    conf_matr3 = [tp3, fp3, fn3, tn3]
    
    plot_hist(conf_matr3, 3, i)

        #print(pred3)


