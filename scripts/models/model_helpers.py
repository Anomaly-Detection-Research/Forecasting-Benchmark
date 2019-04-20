import numpy as np

def MSE(y, y_hat):
    return np.square(y - y_hat).mean()

def get_min_matrix(matrix):
    min_colomns = []
    for row in matrix:
        min_colomns.append(np.argmin(row))
    min_row = 0 
    _min = matrix[min_row][min_colomns[min_row]]
    for row_index in range(0,len(matrix)):
        if matrix[row_index][min_colomns[row_index]] < _min:
            _min = matrix[row_index][min_colomns[row_index]]
            min_row = row_index
    return min_row,min_colomns[min_row]