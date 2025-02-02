import pandas as pd
import numpy as np
from split_dataframe import *
from normalization import *
from knn import *
from plot_hist import *
from class_analysis import *
from stats import *
from plot_cm import *
from compute_binary import *
from check_k import *
from sklearn.datasets import load_wine
from av_stat import *

# Loading the dataset and dividing it between classes and features
wine = load_wine()
X, y = wine.data, wine.target
classes = np.array(y)
feat = np.array(X)
norm_feat = normalization(feat)

# Splitting into training set and test set (divided in classes and features)
tr_cl, tr_feat, test_cl, test_feat = split_matrix_random(classes, norm_feat)

# Initializing the number k needed in kNN + checking on them
k = [1, 2, 3, 4, 5, 10, 15, 20, 30, 40, 50]
check_k(k, 3)

accuracies = []
columns = []
for i in k:
    predictions, accuracy = knn(tr_feat, tr_cl, test_feat, i, test_cl)
    accuracies.append(accuracy)
    columns.append(f"k = {i}")

tbl = pd.DataFrame([accuracies], index=["Accuracy"], columns=columns)  
print("\nAccuracy of knn as k changes:\n\n", tbl, "\n\n")


################################################################## TASK 2
# Creating a list containing the values of the different classes
cl = [0, 1, 2]

# Computing a binary matrix with the classes
binary_tr = []
binary_test =  []
binary_tr = compute_binary(tr_cl, cl)
binary_test = compute_binary(test_cl, cl)

# Inizialising k1 + checking on its values
k1 = [1, 2, 3, 6, 7, 10, 30, 49]
check_k(k1, 2)

# Analyizing each class and predicting for each element to which class it belongs
matr = []
for i in range(len(cl)):
    matr.append(class_analysis(tr_feat, binary_tr[i], test_feat, binary_test[i], k1))
#print("\nMatr0:\n", matr[0], "\nMatr1:\n", matr[1], "\nMatr2:\n", matr[2], "\n\n")

# Plotting the confusion matrices for each class for every k
for i in range(len(cl)):
    plot_conf_matr(matr[i], cl[i], k1)

############################################################################################################################################
# Computing the statistic requested over n iterations of the code and plotting the results for each class for each k value in k1
iterations = 10
average_stats(classes, norm_feat, k1, cl, iterations)

############################################################################################################################################
# Calculating the average and the standard deviations for each class as k changes
acc = []
acc_stats = []

row_labels = []

for c in cl:
    row_labels.append(f"Class {c}")
    acc_cl = []
    for k in k1:
        acc_cl.append(knn_acc(tr_feat, binary_tr[c], test_feat, k, binary_test[c]))
    acc.append(acc_cl)
    acc_stats.append(compute_acc_class(acc_cl))
    
col_labels = []
for k in k1:
    col_labels.append(f"k = {k}")

table_cl = pd.DataFrame(acc, index=row_labels, columns=col_labels)
print("\n\nValues of accuracies:\n\n", table_cl)

plot_table(acc_stats, [f"Class {i}" for i in range(len(acc_stats))], "Average and Standard deviation of the accuracy computed on each class over all values of k")

acc_k = compute_acc_k(acc)
plot_table(acc_k, [f"k = {i}" for i in k1], "Average and Standard deviation of the accuracy computed on each k over all classes")

 