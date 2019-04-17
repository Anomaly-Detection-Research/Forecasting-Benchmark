import numpy as np

def MSE(y, y_hat):
    return np.square(y - y_hat).mean()