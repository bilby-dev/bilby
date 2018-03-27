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


def ra_dec_to_theta_phi(ra, dec, gmst):
    '''
    Convert from RA and DEC to polar coordinates on celestial sphere
    Input:
    ra - right ascension in radians
    dec - declination in radians
    gmst - Greenwich mean sidereal time of arrival of the signal in radians
    Output:
    theta - zenith angle in radians
    phi - azimuthal angle in radians
    '''
    phi = ra - gmst
    theta = np.pi/2 - dec
    return theta, phi
