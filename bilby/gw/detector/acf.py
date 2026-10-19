import numpy as np

from ...core.utils import logger


class AutoCovarianceFunction(object):

    def __init__(self, acf_array=None, sampling_frequency=None, minimum_frequency=None,
                 maximum_frequency=None, fill_value=None, fill_multiplier=None):
        """
        Instantiate a new AutoCovarianceFunction object.

        The autocovariance function (ACF) of stationary Gaussian noise gives the
        noise covariance matrix of N consecutive samples, which is a Toeplitz
        matrix with first row equal to the first N lags of the ACF.

        Examples
        ========
        From a one-sided power spectral density:

        .. code-block:: python

            >>> from bilby.gw.detector import PowerSpectralDensity
            >>> psd = PowerSpectralDensity.from_aligo()
            >>> acf = AutoCovarianceFunction.from_power_spectral_density_array(
            ...     psd.frequency_array, psd.psd_array, minimum_frequency=20,
            ...     maximum_frequency=896, sampling_frequency=2048)

        From a precomputed ACF:

        .. code-block:: python

            >>> acf_array = acf.get_acf_array(4096)
            >>> acf = AutoCovarianceFunction(acf_array=acf_array, sampling_frequency=2048)

        Parameters
        ==========
        acf_array: array_like
            The ACF at lags 0, 1 / sampling_frequency, 2 / sampling_frequency, ...
        sampling_frequency: float
            The sampling frequency (in Hz) of the lags; must equal that of the data
        minimum_frequency: float, optional
            The lower edge (in Hz) of the frequency band used to compute the ACF
        maximum_frequency: float, optional
            The upper edge (in Hz) of the frequency band used to compute the ACF
        fill_value: float, optional
            The PSD value used outside the band
        fill_multiplier: float, optional
            The multiple of the maximum in-band PSD used outside the band

        """
        self.acf_array = None if acf_array is None else np.asarray(acf_array, dtype=float)
        self.sampling_frequency = None if sampling_frequency is None else float(sampling_frequency)
        self.minimum_frequency = minimum_frequency
        self.maximum_frequency = maximum_frequency
        self.fill_value = fill_value
        self.fill_multiplier = fill_multiplier

    @property
    def meta_data(self):
        return dict(
            sampling_frequency=self.sampling_frequency,
            minimum_frequency=self.minimum_frequency,
            maximum_frequency=self.maximum_frequency,
            fill_value=self.fill_value,
            fill_multiplier=self.fill_multiplier,
        )

    @classmethod
    def from_power_spectral_density_array(
            cls, frequency_array, psd_array, minimum_frequency, maximum_frequency,
            sampling_frequency=None, fill_value=None, fill_multiplier=1e4):
        """ Compute the ACF from a one-sided power spectral density (PSD).

        The PSD outside [`minimum_frequency`, `maximum_frequency`] is set to
        `fill_value`, or to `fill_multiplier` times its maximum in the band, and
        the ACF is its inverse Fourier transform (Wiener-Khinchin theorem).

        Parameters
        ==========
        frequency_array: array_like
            Uniformly spaced frequencies (in Hz) up to the Nyquist frequency
        psd_array: array_like
            The one-sided PSD at `frequency_array`
        minimum_frequency: float
            The lower edge of the frequency band (in Hz)
        maximum_frequency: float
            The upper edge of the frequency band (in Hz)
        sampling_frequency: float, optional
            The sampling frequency of the data (in Hz); a warning is logged if it
            is not twice the highest frequency in `frequency_array`
        fill_value: float, optional
            The PSD value used outside the band; overrides `fill_multiplier`
        fill_multiplier: float
            The multiple of the maximum in-band PSD used outside the band

        Returns
        =======
        AutoCovarianceFunction

        """
        frequency_array = np.asarray(frequency_array, dtype=float)
        psd_array = np.asarray(psd_array, dtype=float)
        delta_f = frequency_array[1] - frequency_array[0]
        frequency_values_missing = int(round(frequency_array[0] / delta_f))
        if frequency_values_missing > 0:
            frequency_array = np.concatenate(
                [np.arange(frequency_values_missing) * delta_f, frequency_array])
            psd_array = np.concatenate([np.zeros(frequency_values_missing), psd_array])

        psd_sampling_frequency = 2 * frequency_array[-1]
        if sampling_frequency is not None and not np.isclose(
                sampling_frequency, psd_sampling_frequency, rtol=1e-9, atol=0):
            logger.warning(
                f"The sampling frequency given ({sampling_frequency:g} Hz) is not equal to the one "
                f"computed from the PSD ({psd_sampling_frequency:g} Hz); using "
                f"{psd_sampling_frequency:g} Hz. An ACF computed with {sampling_frequency:g} Hz "
                "would be wrongly scaled and its lags wrongly spaced, giving wrong likelihood "
                "and SNR values.")

        band = (frequency_array >= minimum_frequency) & (frequency_array <= maximum_frequency)
        if fill_value is None:
            fill_value = fill_multiplier * np.max(psd_array[band])
        else:
            fill_multiplier = None
        psd_array = psd_array.copy()
        psd_array[~band] = fill_value
        acf_array = 0.5 * np.fft.irfft(psd_array) * psd_sampling_frequency
        return cls(acf_array=acf_array, sampling_frequency=psd_sampling_frequency,
                   minimum_frequency=minimum_frequency, maximum_frequency=maximum_frequency,
                   fill_value=float(fill_value), fill_multiplier=fill_multiplier)

    def get_acf_array(self, number_of_samples):
        """ The first `number_of_samples` lags of the ACF

        Parameters
        ==========
        number_of_samples: int
            The number of samples in the analysis window

        Returns
        =======
        array_like: The first row of the noise covariance matrix

        """
        if number_of_samples > len(self.acf_array):
            raise ValueError(
                f"The analysis segment ({number_of_samples} samples) is longer than the ACF "
                f"({len(self.acf_array)} samples)")
        return self.acf_array[:number_of_samples]


__all__ = ["AutoCovarianceFunction"]
