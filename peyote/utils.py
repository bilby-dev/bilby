import numpy as np
from astropy.time import Time


def sampling_frequency(time_series):
    """
    Calculate sampling frequency from a time series
    """
    tol = 1e-10
    if np.ptp(np.diff(time_series)) > tol:
        raise ValueError("Your time series was not evenly sampled")
    else:
        return 1. / (time_series[1] - time_series[0])


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
    theta = np.pi / 2 - dec
    return theta, phi


def gps_time_to_gmst(time):
    '''
    Convert gps time to Greenwich mean sidereal time in radians
    Input:
    time - gps time
    Output:
    gmst - Greenwich mean sidereal time in radians
    '''
    gps_time = Time(time, format='gps', scale='utc')
    gmst = gps_time.sidereal_time('mean', 'greenwich').value * np.pi / 12
    return gmst
