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

print("\nMatr1:\n", matr1, "\nMatr2:\n", matr2, "\nMatr3:\n", matr3, "\n\n")

# Computing the values of the statistic requested
stat = []
stat.append(stats(tr_feat, tr_cl, test_feat, test_cl, k1, cl[0]))
stat.append(stats(tr_feat, tr_cl, test_feat, test_cl, k1, cl[1]))
stat.append(stats(tr_feat, tr_cl, test_feat, test_cl, k1, cl[2]))

plot_stats(stat)

"""for i in range(len(stat)):
    print(f"Class {i+1}:\n sensitivity_{i+1} = {(stat[i][0]):.2f}, specifity_{i+1} = {(stat[i][1]):.2f}, precision_{i+1} = {(stat[i][2]):.2f}, f1_score_{i+1} = {(stat[i][3]):.2f}\n")
"""
# Plotting the histograms for each class for every k
"""plot_hist(matr1, cl[0], k1)
plot_hist(matr2, cl[1], k1)
plot_hist(matr3, cl[2], k1)"""

# Plotting the confusion matrices for each class for every k
plot_confusion_matr(matr1, cl[0], k1)
plot_confusion_matr(matr2, cl[1], k1)
plot_confusion_matr(matr3, cl[2], k1)
    



