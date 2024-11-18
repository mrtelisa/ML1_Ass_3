import pandas as pd
import numpy as np
from split_dataframe import *
from normalization import *
from knn import *
from accuracy import *
from plot_hist import *
from class_analysis import *


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

k1 = [3, 5, 6, 7, 10, 25, 30, 50]
cl = [1, 2, 3]

matr1 = class_analysis(tr_feat, tr_cl, test_feat, test_cl, k1, cl[0])
matr2 = class_analysis(tr_feat, tr_cl, test_feat, test_cl, k1, cl[1])
matr3 = class_analysis(tr_feat, tr_cl, test_feat, test_cl, k1, cl[2])

print("\nMatr1:\n", matr1, "\nMatr2:\n", matr2, "\nMatr3:\n", matr3)

plot_hist(matr1, cl[0], k1)
plot_hist(matr2, cl[1], k1)
plot_hist(matr3, cl[2], k1)
    



