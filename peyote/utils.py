import numpy as np

def sampling_frequency(time_series):
    """
    Calculate sampling frequency from a time series
    """
    tol = 1e-10
    if np.ptp(np.diff(time_series)) > tol:
        raise ValueError("Your time series was not evenly sampled")
    else:
        return 1./(time_series[1] - time_series[0])
