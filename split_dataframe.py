import random
import math

def split_matrix_random(cl, feat):

    # Selecting 4 casual rows
    ind = random.sample(range(len(feat)), math.ceil(len(feat)*0.7))
    #print(ind)

    tr_feat = [feat[i] for i in ind]
    tr_cl = [cl[i] for i in ind]

    test_feat = [feat[i] for i in range(len(feat)) if i not in ind]
    test_cl = [cl[i] for i in range(len(feat)) if i not in ind]

    cleaned_tr_feat = [row.tolist() for row in tr_feat]
    cleaned_test_feat = [row.tolist() for row in test_feat]
    
    cleaned_tr_cl = [float(row) for row in tr_cl]
    cleaned_test_cl = [float(row) for row in test_cl]
    
    return cleaned_tr_cl, cleaned_tr_feat, cleaned_test_cl, cleaned_test_feat

def split_matrix(cl, feat):

    # Selecting specific rows
    ind = list(range(0, 47)) + list(range(99, 177))
    #print(ind)

    tr_feat = [feat[i] for i in ind]
    tr_cl = [cl[i] for i in ind]

    test_feat = [feat[i] for i in range(len(feat)) if i not in ind]
    test_cl = [cl[i] for i in range(len(feat)) if i not in ind]

    cleaned_tr_feat = [row.tolist() for row in tr_feat]
    cleaned_test_feat = [row.tolist() for row in test_feat]
    
    cleaned_tr_cl = [float(row) for row in tr_cl]
    cleaned_test_cl = [float(row) for row in test_cl]
    
    return cleaned_tr_cl, cleaned_tr_feat, cleaned_test_cl, cleaned_test_feat