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

k1 = [3, 5, 6, 7, 10, 30]
cl = [1, 2, 3]

# Analyizing each class
matr1 = class_analysis(tr_feat, tr_cl, test_feat, test_cl, k1, cl[0])
matr2 = class_analysis(tr_feat, tr_cl, test_feat, test_cl, k1, cl[1])
matr3 = class_analysis(tr_feat, tr_cl, test_feat, test_cl, k1, cl[2])

print("\nMatr1:\n", matr1, "\nMatr2:\n", matr2, "\nMatr3:\n", matr3)

# Computing the values of the statistic requested
stat = []
stat.append(stats(tr_feat, tr_cl, test_feat, test_cl, k1, cl[0]))
stat.append(stats(tr_feat, tr_cl, test_feat, test_cl, k1, cl[1]))
stat.append(stats(tr_feat, tr_cl, test_feat, test_cl, k1, cl[2]))

for i in range(len(stat)):
    print(f"sensitivity_{i+1} =", stat[i][0], f"specifity_{i+1} =", stat[i][1], f"precision_{i+1} =", stat[i][2], f"f1_score_{i+1} =", stat[i][3])

# Plotting the histograms for each class for every k
"""plot_hist(matr1, cl[0], k1)
plot_hist(matr2, cl[1], k1)
plot_hist(matr3, cl[2], k1)"""

# Supponendo che matr1, matr2, matr3 siano le matrici per le classi 1, 2 e 3
plot_confusion_matrices_as_tables(matr1, cl[0], k1)
plot_confusion_matrices_as_tables(matr2, cl[1], k1)
plot_confusion_matrices_as_tables(matr3, cl[2], k1)
    



