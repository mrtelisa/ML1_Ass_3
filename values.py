from knn import *
from accuracy import *
import numpy as np

# Calculate the accuracy of the predictions
def calculate_accuracy(predictions, test_cl):
    n = 0
    for i in range(len(test_cl)):
        if(predictions[i] == test_cl[i]):
            n += 1
    return (n/len(test_cl))

# Calculate the error rate of the predictions
def calculate_error(predictions, test_cl):
    n = 0
    for i in range(len(test_cl)):
        if(predictions[i] != test_cl[i]):
            n += 1
    return (n/len(test_cl))

# Calculate the average of the predictions
def compute_av_stats(st):

    sens, spec, prec, f1, aver = 0, 0, 0, 0, 0
    for i in range(len(st)):
        sens += st[i][0]
        spec += st[i][1]
        prec += st[i][2]
        f1 += st[i][3]
        aver += st[i][4]

    return [sens/len(st), spec/len(st), prec/len(st), f1/len(st), aver/len(st)]

# Calculate the median of the predictions
def compute_median(st):

    median = [0 for i in range(len(st[0]))]

    if(len(st)%2 == 0):
        for i in range(len(st[0])):
            median[i] = (st[int(len(st)/2)][i] + st[int((len(st)+1)/2)][i])/2

    else:
        for i in range(len(st[0])):
            median[i] = st[int(len(st))+1][i]

    return median

# Calculate the mode of the predictions
def compute_mode(st):
    
    mode = []
    for j in range(len(st[0])):
        unique = []
        a = 0
        for i in range(len(st)):
            if(st[i][j] not in unique):
                unique.append(st[i][j])
            count = np.zeros(len(unique))
            for k in range(len(unique)):
                if(st[i][j] == unique[k]):
                    count[k] += 1     
        for k in range(len(unique)):
            if (unique[k] > a):
                a = k
        mode.append(st[a][j])

    return mode



