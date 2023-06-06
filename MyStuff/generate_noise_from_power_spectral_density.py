import numpy as np
from bilby.gw.detector import PowerSpectralDensity

def generate_noise_from_power_spectral_density(psd, duration, sampling_frequency):
    """
    Generate a time-domain noise realization using a given power spectral density (PSD).

    Parameters
    ----------
    psd: bilby.gw.detector.PowerSpectralDensity
        Power spectral density object.
    duration: float
        Duration of the time series (in seconds).
    sampling_frequency: float
        Sampling frequency of the time series (in Hz).

    Returns
    -------
    array_like
        A time series of noise that follows the input PSD.
    """
    # Generate white noise with the desired length and sampling rate
    N = int(duration * sampling_frequency)
    white_noise = np.random.normal(size=N)

    # Get the frequency values for the PSD
    freqs = psd.sample_frequencies

    # Interpolate the PSD onto the frequency values
    psd_vals = psd.interpolated_values(freqs)

    # Calculate the amplitude spectral density (ASD) from the PSD
    asd_vals = np.sqrt(psd_vals)

    # Calculate the Fourier transform of the white noise
    white_noise_ft = np.fft.rfft(white_noise)

    # Scale the Fourier transform by the ASD values
    scaled_noise_ft = white_noise_ft * asd_vals

    # Calculate the inverse Fourier transform to get the time-domain noise
    noise_realization = np.fft.irfft(scaled_noise_ft)

    # Trim the noise realization to the desired length
    noise_realization = noise_realization[:N]

    return noise_realization
