# Computes a binary matrix
def compute_binary(matr, cl):
    binary = []

    for i in range(len(cl)):
        b = []
        for j in range(len(matr)):  
            if matr[j] == cl[i]:
                b.append(1)
            else:
                b.append(0)
        binary.append(b)
        
    return binary