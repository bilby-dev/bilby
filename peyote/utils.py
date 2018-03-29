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


def create_time_series(sampling_frequency, duration, starting_time = 0.):
    return np.arange(starting_time, duration, 1./sampling_frequency)

def ra_dec_to_theta_phi(ra, dec, gmst):
    """
    Convert from RA and DEC to polar coordinates on celestial sphere
    Input:
    ra - right ascension in radians
    dec - declination in radians
    gmst - Greenwich mean sidereal time of arrival of the signal in radians
    Output:
    theta - zenith angle in radians
    phi - azimuthal angle in radians
    """
    phi = ra - gmst
    theta = np.pi / 2 - dec
    return theta, phi


def gps_time_to_gmst(time):
    """
    Convert gps time to Greenwich mean sidereal time in radians
    Input:
    time - gps time
    Output:
    gmst - Greenwich mean sidereal time in radians
    """
    gps_time = Time(time, format='gps', scale='utc')
    gmst = gps_time.sidereal_time('mean', 'greenwich').value * np.pi / 12
    return gmst


def create_white_noise(sampling_frequency, duration):
    """
    Create white_noise which is then coloured by a given PSD
    """

    number_of_samples = duration * sampling_frequency
    number_of_samples = int(np.round(number_of_samples))

    # prepare for FFT
    number_of_frequencies = (number_of_samples-1)//2
    delta_freq = 1./duration

    f = delta_freq * np.linspace(1, number_of_frequencies, number_of_frequencies)

    norm1 = 0.5*(1./delta_freq)**0.5
    re1 = np.random.normal(0, norm1, int(number_of_frequencies))
    im1 = np.random.normal(0, norm1, int(number_of_frequencies))
    z1 = re1 + 1j*im1


    # freq domain solution for htilde1, htilde2 in terms of z1, z2
    htilde1 = z1
    # convolve data with instrument transfer function
    otilde1 = htilde1 * 1.
    # set DC and Nyquist = 0
    # python: we are working entirely with positive frequencies
    if np.mod(number_of_samples, 2) == 0:
        otilde1 = np.concatenate(([0], otilde1, [0]))
        f = np.concatenate(([0], f, [sampling_frequency / 2.]))
    else:
        # no Nyquist frequency when N=odd
        otilde1 = np.concatenate(([0], otilde1))
        f = np.concatenate(([0], f))

    # normalise for positive frequencies and units of strain/rHz
    white_noise = otilde1
    # python: transpose for use with infft
    white_noise = np.transpose(white_noise)
    f = np.transpose(f)

    return white_noise, f


def nfft(ht, Fs):
    '''
    performs an FFT while keeping track of the frequency bins
    assumes input time series is real (positive frequencies only)

    ht = time series
    Fs = sampling frequency

    returns
    hf = single-sided FFT of ft normalised to units of strain / sqrt(Hz)
    f = frequencies associated with hf
    '''
    # add one zero padding if time series does not have even number of sampling times
    if np.mod(len(ht), 2) == 1:
        ht = np.append(ht, 0)
    LL = len(ht)
    # frequency range
    ff = Fs / 2 * np.linspace(0, 1, LL/2+1)

    # calculate FFT
    # rfft computes the fft for real inputs
    hf = np.fft.rfft(ht)

    # normalise to units of strain / sqrt(Hz)
    hf = hf / Fs

    return hf, ff


def infft(hf, Fs):
    '''
    inverse FFT for use in conjunction with nfft
    eric.thrane@ligo.org
    input:
    hf = single-side FFT calculated by fft_eht
    Fs = sampling frequency
    output:
    h = time series
    '''
    # use irfft to work with positive frequencies only
    h = np.fft.irfft(hf)
    # undo LAL/Lasky normalisation
    h = h*Fs

    return h
