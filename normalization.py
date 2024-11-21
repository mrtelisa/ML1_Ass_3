# Normalization of the matrix in input
def normalization(matrix):    

    min_vals = matrix.min(axis=0)  # min in each column
    max_vals = matrix.max(axis=0)  # max in each column
    normalized_matrix = (matrix - min_vals) / (max_vals - min_vals)

    return normalized_matrix
