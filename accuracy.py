
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