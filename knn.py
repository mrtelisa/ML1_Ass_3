import numpy as np
from accuracy import *
# Reorder the elements in an array giving as output an index vector
def k_sort_array(arr, k):

    indexed_arr = list(enumerate(arr))
    sorted_indexed_arr = sorted(indexed_arr, key=lambda x: x[1])
    indices = [x[0] for x in sorted_indexed_arr]
    ind = [indices[i] for i in range(k)]

    return ind

# Knn classifier implementation
def knn(tr_feat, tr_cl, test_feat, k, test_cl=None):

    if tr_feat is None or tr_cl is None or test_feat is None or k is None:
        raise ValueError("Insufficient number of input! 4 arguments are required.")

    if len(tr_feat[0]) != len(test_feat[0]):
        raise ValueError("Il numero di colonne di train_set deve essere uguale al numero di colonne di test_set.")

    if k <= 0 or k > len(tr_cl):
        raise ValueError("Vector k containing incomputable values!")

    pred = []

    # For each point in the test set, compute the distance of all the points in the training
    for i in range(len(test_feat)):
        k_labels = []
        norm = []
        for j in range(len(tr_feat)):
            val = np.linalg.norm(np.array(test_feat[i]) - np.array(tr_feat[j]))
            norm.append(float(val))

        # Order the distances and takes the label of k nearest points
        k_neighbors = np.argsort(norm)[:k]  # Using numpy.argsort to sort indices
        for j in k_neighbors:
            k_labels.append(tr_cl[j])

        # Counting the cardinality of each class and adding in "pred" the mode.
        pred_label = np.bincount(k_labels).argmax()
        pred.append(int(pred_label))

    if test_cl is not None:
        accuracy = calculate_accuracy(pred, test_cl)
        return pred, accuracy

    return pred

def knn_acc(tr_feat, tr_cl, test_feat, k, test_cl):
    pred, acc = knn(tr_feat, tr_cl, test_feat, k, test_cl)
    return acc