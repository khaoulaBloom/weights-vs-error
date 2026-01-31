import numpy as np


def mse(y_true, y_pred):
    errors = y_true - y_pred 
    squared = errors ** 2
    mean_value = np.mean(squared)
    return mean_value

    