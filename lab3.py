import pandas as pd
import numpy as np
from split_dataframe import *
from normalization import *
from knn import *
from accuracy import *
from plot_hist import *
from class_analysis import *
from stats import *
from plot_confusion_matr import *
from compute_binary import *

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
k = [1, 2, 3, 4, 5, 10, 15, 20, 30, 40, 50]

for i in k:
    predictions, accuracy = knn(tr_feat, tr_cl, test_feat, i, test_cl)
        #print(accuracy)
        #print(predictions)

k1 = [3, 5, 6, 7, 10, 30]
cl = [1, 2, 3]

# Computing a binary matrix with the classes
binary_tr = []
binary_test =  []
binary_tr = compute_binary(tr_cl, cl)
binary_test = compute_binary(test_cl, cl)
    #print("Binary test:\n", binary_test, "\nBinary training\n", binary_tr)

# Analyizing each class
matr1 = class_analysis(tr_feat, binary_tr[0], test_feat, binary_test[0], k1)
matr2 = class_analysis(tr_feat, binary_tr[1], test_feat, binary_test[1], k1)
matr3 = class_analysis(tr_feat, binary_tr[2], test_feat, binary_test[2], k1)

#print("\nMatr1:\n", matr1, "\nMatr2:\n", matr2, "\nMatr3:\n", matr3, "\n\n")

# Computing the values of the statistic requested
stat = []
stat.append(stats(tr_feat, binary_tr[0], test_feat, binary_test[0], k1))
stat.append(stats(tr_feat, binary_tr[1], test_feat, binary_test[1], k1))
stat.append(stats(tr_feat, binary_tr[2], test_feat, binary_test[2], k1))
plot_stats(stat)

# Plotting the histograms for each class for every k
"""plot_hist(matr1, cl[0], k1)
plot_hist(matr2, cl[1], k1)
plot_hist(matr3, cl[2], k1)"""

# Plotting the confusion matrices for each class for every k
"""plot_confusion_matr(matr1, cl[0], k1)
plot_confusion_matr(matr2, cl[1], k1)
plot_confusion_matr(matr3, cl[2], k1)"""
    



