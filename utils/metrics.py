import numpy as np

def calculate_NSE(observed, predicted):
    numerator = np.sum((observed - predicted) ** 2)
    denominator = np.sum((observed - np.mean(observed)) ** 2)
    nse = 1 - (numerator / denominator)
    return nse

def calculate_MAE(observed, predicted):
    return np.mean(np.abs(observed - predicted))

def calculate_RMSE(observed, predicted):
    mse = np.mean((observed - predicted) ** 2)
    rmse = np.sqrt(mse)
    return rmse
