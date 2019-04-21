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

def get_min_3dmatrix(matrix):
    min_in_z_dimension = []
    for z in range(0, matrix.shape[0]):
        min_row, min_colomn = get_min_matrix(matrix[z])
        min_in_z_dimension.append([min_row, min_colomn])

    min_z = 0 
    _min = matrix[min_z][min_in_z_dimension[min_z][0]][min_in_z_dimension[min_z][1]]
    for z_index in range(0,matrix.shape[0]):
        if matrix[z_index][min_in_z_dimension[z_index][0]][min_in_z_dimension[z_index][1]] < _min:
            _min = matrix[z_index][min_in_z_dimension[z_index][0]][min_in_z_dimension[z_index][1]]
            min_z = z_index
    return min_z, min_in_z_dimension[min_z][0], min_in_z_dimension[min_z][1]