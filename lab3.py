import pandas as pd
import numpy as np
from split_dataframe import *
from normalization import *
from knn import *
from accuracy import *
from plot_hist import *
from class_analysis import *
from stats import *
from plot_cm import *
from compute_binary import *
from check_k import *

# Loading the dataset
matrix = np.loadtxt("wine_data.txt", delimiter=",")

# Dividing matrix between the class and the features + normalization of the features
cl = matrix[:, 0]
feat = matrix[:, 1:]
norm_feat = normalization(feat)

# Splitting into training set and test set (divided in classes and features)
tr_cl, tr_feat, test_cl, test_feat = split_matrix_random(cl, norm_feat)

# Initializing the number k needed in kNN + checking on them
k = [1, 2, 3, 4, 5, 10, 15, 20, 30, 40, 50]
check_k(k, 3)

for i in k:
    predictions, accuracy = knn(tr_feat, tr_cl, test_feat, i, test_cl)

################################################################## TASK 2

# Computing a binary matrix with the classes
binary_tr = []
binary_test =  []
binary_tr = compute_binary(tr_cl, cl)
binary_test = compute_binary(test_cl, cl)

# Inizialising k1 + checking on its values
k1 = [1, 2, 3, 5, 6, 7, 10, 30, 49]
check_k(k1, 2)

# Creating a list containing the values of the different classes
cl = [1, 2, 3]

# Analyizing each class
matr = []
for i in range(len(cl)):
    matr.append(class_analysis(tr_feat, binary_tr[i], test_feat, binary_test[i], k1))
print("\nMatr1:\n", matr[0], "\nMatr2:\n", matr[1], "\nMatr3:\n", matr[2], "\n\n")

# Computing the values of the statistic requested
stat = []
for i in range(len(cl)):
    stat.append(stats(tr_feat, binary_tr[i], test_feat, binary_test[i], k1))
plot_stats(stat)

# Plotting the confusion matrices for each class for every k
for i in range(len(cl)):
    plot_conf_matr(matr[i], cl[i], k1)
    



