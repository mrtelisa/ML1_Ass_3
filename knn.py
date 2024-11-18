import numpy as np
from collections import Counter

def k_sort_array(arr, k):
    indexed_arr = list(enumerate(arr))
    sorted_indexed_arr = sorted(indexed_arr, key=lambda x: x[1])
    indices = [x[0] for x in sorted_indexed_arr]
    ind = [indices[i] for i in range(k)]
    return ind


def knn(tr_feat, tr_cl, test_feat, k):
        #print(len(tr_feat), len(test_feat))
    if k <= 0 or k > len(tr_cl):
        raise ValueError("k deve essere > 0 e <= al numero di dati di training")

        #print(len(tr_feat))
    
    pred = []

    # For each point in the test set, compute the distance of all the points in the training
    for i in range(len(test_feat)):
        k_labels = []
        norm = []
        for j in range(len(tr_feat)):
                #print(type(test_feat), "\n", type(tr_feat))
                #print(type(test_feat[i]), "\n", type(tr_feat[j]))
            val = np.linalg.norm(np.array(test_feat[i]) - np.array(tr_feat[j]))
            norm.append(float(val))
            #print(len(norm))

        # Order the distances and takes the lable of k nearest points
        k_neighbors = k_sort_array(norm, k)

        for j in k_neighbors:
            k_labels.append(tr_cl[j])
            #print(k_labels)
        
        # Counting the cardinality of each class and adding in "pred" the mode.
        pred_label = np.bincount(k_labels).argmax()
        pred.append(int(pred_label))
        #print(pred)
    
    return pred

