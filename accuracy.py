
# Calculate the accuracy of the predictions
def calculate_accuracy(predictions, test_cl):
    n = 0
    for i in range(len(test_cl)):
        if(predictions[i] == test_cl[i]):
            n += 1
        else:
            n += 0
    acc = n/len(test_cl)

    return acc

# Calculating the accuracy for each class, obtaining tp, fp, fn, tn
def accuracy_class(pred, test_cl):

    tp, fp, fn, tn = 0, 0, 0, 0

    for i in range(len(pred)):
        if(pred[i] == 1):
            if(test_cl[i] == 1):
                tp += 1
            else:
                fp += 1
        else:
            if(test_cl[i] == 1):
                fn += 1
            else:
                tn += 1

    return tp, fp, fn, tn