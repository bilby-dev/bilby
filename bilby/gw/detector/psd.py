import os

import array_api_compat as aac
import array_api_extra as xpx
import numpy as np

from ...core.utils.calculus import interp1d
from ...core import utils
from ...core.utils import logger
from .strain_data import InterferometerStrainData


class PowerSpectralDensity(object):

    # A precomputed autocovariance function, see from_autocovariance_array
    autocovariance_array = None
    autocovariance_sampling_frequency = None

    def __init__(self, frequency_array=None, psd_array=None, asd_array=None,
                 psd_file=None, asd_file=None):
        """
        Instantiate a new PowerSpectralDensity object.

        Examples
        ========
        Using the `from` method directly (here `psd_file` is a string
        containing the path to the file to load):

        .. code-block:: python

            >>> psd_file = "aLIGO_ZERO_DET_high_P_psd.txt"
            >>> power_spectral_density = PowerSpectralDensity.from_power_spectral_density_file(psd_file)

        Alternatively (and equivalently) setting the psd_file directly:

        .. code-block:: python

            >>> power_spectral_density = PowerSpectralDensity(psd_file=psd_file)

        Attributes
        ==========
        asd_array: array_like
            Array representation of the ASD
        asd_file: str
            Name of the ASD file
        frequency_array: array_like
            Array containing the frequencies of the ASD/PSD values
        psd_array: array_like
            Array representation of the PSD
        psd_file: str
            Name of the PSD file
        power_spectral_density_interpolated: scipy.interpolated.interp1d
            Interpolated function of the PSD

        """
        self._cache = dict(
            frequency_array=np.array([]), psd_array=None, asd_array=None)
        self.frequency_array = np.array(frequency_array)
        if psd_array is not None:
            self.psd_array = psd_array
        if asd_array is not None:
            self.asd_array = asd_array
        self.psd_file = psd_file
        self.asd_file = asd_file

    def _update_cache(self, frequency_array):
        psd_array = self.power_spectral_density_interpolated(frequency_array)
        self._cache['psd_array'] = psd_array
        self._cache['asd_array'] = psd_array**0.5
        self._cache['frequency_array'] = frequency_array

    def __eq__(self, other):
        if self.psd_file == other.psd_file \
                and self.asd_file == other.asd_file \
                and np.array_equal(self.frequency_array, other.frequency_array) \
                and np.array_equal(self.psd_array, other.psd_array) \
                and np.array_equal(self.asd_array, other.asd_array):
            return True
        return False

    def __repr__(self):
        if self.asd_file is not None or self.psd_file is not None:
            return self.__class__.__name__ + '(psd_file=\'{}\', asd_file=\'{}\')' \
                .format(self.psd_file, self.asd_file)
        else:
            return self.__class__.__name__ + '(frequency_array={}, psd_array={}, asd_array={})' \
                .format(self.frequency_array, self.psd_array, self.asd_array)

    @staticmethod
    def from_amplitude_spectral_density_file(asd_file):
        """ Set the amplitude spectral density from a given file

        Parameters
        ==========
        asd_file: str
            File containing amplitude spectral density, format 'f h_f'

        """
        return PowerSpectralDensity(asd_file=asd_file)

    @staticmethod
    def from_power_spectral_density_file(psd_file):
        """ Set the power spectral density from a given file

        Parameters
        ==========
        psd_file: str, optional
            File containing power spectral density, format 'f h_f'

        """
        return PowerSpectralDensity(psd_file=psd_file)

    @staticmethod
    def from_frame_file(frame_file, psd_start_time, psd_duration,
                        fft_length=4, sampling_frequency=4096, roll_off=0.2,
                        overlap=0, channel=None, name=None, outdir=None,
                        analysis_segment_start_time=None):
        """ Generate power spectral density from a frame file

        Parameters
        ==========
        frame_file: str, optional
            Frame file to read data from.
        psd_start_time: float
            Beginning of segment to analyse.
        psd_duration: float, optional
            Duration of data (in seconds) to generate PSD from.
        fft_length: float, optional
            Number of seconds in a single fft.
        sampling_frequency: float, optional
            Sampling frequency for time series.
            This is twice the maximum frequency.
        roll_off: float, optional
            Rise time in seconds of tukey window.
        overlap: float,
            Number of seconds of overlap between FFTs.
        channel: str, optional
            Name of channel to use to generate PSD.
        name, outdir: str, optional
            Name (and outdir) of the detector for which a PSD is to be
            generated.
        analysis_segment_start_time: float, optional
            The start time of the analysis segment, if given, this data will
            be removed before creating the PSD.

        """
        strain = InterferometerStrainData(roll_off=roll_off)
        strain.set_from_frame_file(
            frame_file, start_time=psd_start_time, duration=psd_duration,
            channel=channel, sampling_frequency=sampling_frequency)
        frequency_array, psd_array = strain.create_power_spectral_density(
            fft_length=fft_length, name=name, outdir=outdir, overlap=overlap,
            analysis_segment_start_time=analysis_segment_start_time)
        return PowerSpectralDensity(frequency_array=frequency_array, psd_array=psd_array)

    @staticmethod
    def from_channel_name(channel, psd_start_time, psd_duration,
                          fft_length=4, sampling_frequency=4096, roll_off=0.2,
                          overlap=0, name=None, outdir=None,
                          analysis_segment_start_time=None):
        """ Generate power spectral density from a given channel name
        by loading data using `strain_data.set_from_channel_name`

        Parameters
        ==========
        psd_start_time: float
            Beginning of segment to analyse.
        psd_duration: float, optional
            Duration of data (in seconds) to generate PSD from.
        fft_length: float, optional
            Number of seconds in a single fft.
        sampling_frequency: float, optional
            Sampling frequency for time series.
            This is twice the maximum frequency.
        roll_off: float, optional
            Rise time in seconds of tukey window.
        overlap: float,
            Number of seconds of overlap between FFTs.
        channel: str
            Name of channel to use to generate PSD in the format
            `IFO:Channel`
        name, outdir: str, optional
            Name (and outdir) of the detector for which a PSD is to be
            generated.
        analysis_segment_start_time: float, optional
            The start time of the analysis segment, if given, this data will
            be removed before creating the PSD.

        """
        strain = InterferometerStrainData(roll_off=roll_off)
        strain.set_from_channel_name(
            channel, duration=psd_duration, start_time=psd_start_time,
            sampling_frequency=sampling_frequency)
        frequency_array, psd_array = strain.create_power_spectral_density(
            fft_length=fft_length, name=name, outdir=outdir, overlap=overlap,
            analysis_segment_start_time=analysis_segment_start_time)
        return PowerSpectralDensity(frequency_array=frequency_array, psd_array=psd_array)

    @staticmethod
    def from_amplitude_spectral_density_array(frequency_array, asd_array):
        return PowerSpectralDensity(frequency_array=frequency_array, asd_array=asd_array)

    @staticmethod
    def from_power_spectral_density_array(frequency_array, psd_array):
        return PowerSpectralDensity(frequency_array=frequency_array, psd_array=psd_array)

    @staticmethod
    def from_autocovariance_array(autocovariance_array, sampling_frequency):
        """ A noise model given by a precomputed autocovariance function (ACF), for
        time-domain likelihoods. It has no power spectral density.

        Parameters
        ==========
        autocovariance_array: array_like
            The ACF at lags 0, 1 / sampling_frequency, 2 / sampling_frequency, ...
        sampling_frequency: float
            The sampling frequency (in Hz) of the lags

        Examples
        ========
        .. code-block:: python

            >>> ifo.power_spectral_density = PowerSpectralDensity.from_autocovariance_array(
            ...     np.loadtxt("acf_H1.txt"), 2048)
            >>> ifo.set_autocovariance_function_from_power_spectral_density()
        """
        psd = PowerSpectralDensity()
        psd.autocovariance_array = np.asarray(autocovariance_array, dtype=float)
        psd.autocovariance_sampling_frequency = float(sampling_frequency)
        return psd

    @staticmethod
    def from_aligo():
        logger.info("No power spectral density provided, using aLIGO,"
                    "zero detuning, high power.")
        return PowerSpectralDensity.from_power_spectral_density_file(psd_file='aLIGO_ZERO_DET_high_P_psd.txt')

    @property
    def psd_array(self):
        return self._psd_array

    @psd_array.setter
    def psd_array(self, psd_array):
        self._check_frequency_array_matches_density_array(psd_array)
        self._psd_array = np.array(psd_array)
        self._asd_array = psd_array ** 0.5
        self._interpolate_power_spectral_density()

    @property
    def asd_array(self):
        return self._asd_array

    @asd_array.setter
    def asd_array(self, asd_array):
        self._check_frequency_array_matches_density_array(asd_array)
        self._asd_array = np.array(asd_array)
        self._psd_array = asd_array ** 2
        self._interpolate_power_spectral_density()

    def _check_frequency_array_matches_density_array(self, density_array):
        if len(self.frequency_array) != len(density_array):
            raise ValueError('Provided spectral density does not match frequency array. Not updating.\n'
                             'Length spectral density {}\n Length frequency array {}\n'
                             .format(density_array, self.frequency_array))

    def _interpolate_power_spectral_density(self):
        """Interpolate the loaded power spectral density so it can be resampled
           for arbitrary frequency arrays.
        """
        self._power_spectral_density_interpolated = interp1d(self.frequency_array,
                                                             self.psd_array,
                                                             bounds_error=False,
                                                             fill_value=np.inf)
        self._update_cache(self.frequency_array)

    def get_power_spectral_density_array(self, frequency_array):
        if aac.is_jax_array(frequency_array):
            return self.power_spectral_density_interpolated(frequency_array)
        if not np.array_equal(frequency_array, self._cache['frequency_array']):
            self._update_cache(frequency_array=frequency_array)
        return self._cache['psd_array']

    def get_autocovariance_array(
            self, number_of_samples, sampling_frequency, minimum_frequency,
            maximum_frequency, duration_factor=16, fill_value=None,
            fill_multiplier=100):
        """ Compute the autocovariance function (ACF) from this PSD.

        The ACF is the inverse Fourier transform of the (one-sided) PSD
        (Wiener-Khinchin theorem). The PSD is evaluated on a frequency grid
        whose length is the next power of two at or above
        ``duration_factor * number_of_samples``, so that the ACF has the
        correct sampling frequency. If the PSD was created with
        :code:`from_autocovariance_array`, its first `number_of_samples` lags are
        returned and the band and fill arguments are not used.

        Parameters
        ==========
        number_of_samples: int
            The number of samples in the analysis window
        sampling_frequency: float
            The sampling frequency of the data (in Hz)
        minimum_frequency: float
            The lower edge of the frequency band (in Hz)
        maximum_frequency: float
            The upper edge of the frequency band (in Hz)
        duration_factor: float
            The minimum ratio of the ACF length to the analysis window length
        fill_value: float or tuple, optional
            The PSD value used outside the band, or the values (below, above) the band;
            overrides `fill_multiplier`
        fill_multiplier: float
            The PSD below (above) the band is set to this multiple of its maximum there;
            see :code:`power_spectral_density_to_autocovariance`

        Returns
        =======
        acf_array: numpy.ndarray
            The first `number_of_samples` lags of the ACF
        fill_value: tuple or None
            The PSD values used (below, above) the band (None for a precomputed ACF)
        fill_multiplier: float or None
            The multiplier used (None if `fill_value` was given explicitly or for a
            precomputed ACF)

        """
        if self.autocovariance_array is not None:
            if not np.isclose(sampling_frequency, self.autocovariance_sampling_frequency, rtol=1e-9, atol=0):
                raise ValueError(
                    f"The sampling frequency of the autocovariance function "
                    f"({self.autocovariance_sampling_frequency:g} Hz) differs from {sampling_frequency:g} Hz")
            acf_array, fill_value, fill_multiplier = self.autocovariance_array, None, None
        else:
            fft_length = 2 ** int(np.ceil(np.log2(duration_factor * number_of_samples)))
            frequency_array = np.fft.rfftfreq(fft_length, 1 / sampling_frequency)
            with np.errstate(invalid="ignore", divide="ignore"):
                psd_array = self.get_power_spectral_density_array(frequency_array)
            acf_array, fill_value, fill_multiplier = power_spectral_density_to_autocovariance(
                frequency_array, psd_array, minimum_frequency, maximum_frequency,
                sampling_frequency=sampling_frequency, fill_value=fill_value,
                fill_multiplier=fill_multiplier)
        if number_of_samples > len(acf_array):
            raise ValueError(
                f"The analysis segment ({number_of_samples} samples) is longer "
                f"than the ACF ({len(acf_array)} samples)")
        return acf_array[:number_of_samples], fill_value, fill_multiplier

    def get_amplitude_spectral_density_array(self, frequency_array):
        if aac.is_jax_array(frequency_array):
            return self.power_spectral_density_interpolated(frequency_array)**0.5
        if not np.array_equal(frequency_array, self._cache['frequency_array']):
            self._update_cache(frequency_array=frequency_array)
        return self._cache['asd_array']

    @property
    def power_spectral_density_interpolated(self):
        return self._power_spectral_density_interpolated

    @property
    def asd_file(self):
        return self._asd_file

    @asd_file.setter
    def asd_file(self, asd_file):
        asd_file = self._validate_file_name(file=asd_file)
        self._asd_file = asd_file
        if asd_file is not None:
            self._import_amplitude_spectral_density()
            self._check_file_was_asd_file()

    def _check_file_was_asd_file(self):
        if min(self.asd_array) < 1e-30:
            logger.warning("You specified an amplitude spectral density file.")
            logger.warning("{} WARNING {}".format("*" * 30, "*" * 30))
            logger.warning("The minimum of the provided curve is {:.2e}.".format(min(self.asd_array)))
            logger.warning("You may have intended to provide this as a power spectral density.")

    @property
    def psd_file(self):
        return self._psd_file

    @psd_file.setter
    def psd_file(self, psd_file):
        psd_file = self._validate_file_name(file=psd_file)
        self._psd_file = psd_file
        if psd_file is not None:
            self._import_power_spectral_density()
            self._check_file_was_psd_file()

    def _check_file_was_psd_file(self):
        if min(self.psd_array) > 1e-30:
            logger.warning("You specified a power spectral density file.")
            logger.warning("{} WARNING {}".format("*" * 30, "*" * 30))
            logger.warning("The minimum of the provided curve is {:.2e}.".format(min(self.psd_array)))
            logger.warning("You may have intended to provide this as an amplitude spectral density.")

    @staticmethod
    def _validate_file_name(file):
        """
        Test if the file exists or is available in the default directory.

        Parameters
        ==========
        file: str, None
            A string pointing either to a PSD file, or the name of a psd file
            in the default directory. If none, no check is performed.

        Returns
        =======
        file: str
            The path to the PSD file to use

        Raises
        ======
        ValueError:
            If the PSD file cannot be located

        """
        if file is None:
            logger.debug("PSD file set to None")
            return None
        elif os.path.isfile(file):
            logger.debug("PSD file {} exists".format(file))
            return file
        else:
            file_in_default_directory = (
                os.path.join(os.path.dirname(__file__), 'noise_curves', file))
            if os.path.isfile(file_in_default_directory):
                logger.debug("PSD file {} exists in default dir.".format(file))
                return file_in_default_directory
            else:
                raise ValueError(
                    "Unable to locate PSD file {} locally or in the default dir"
                    .format(file))
        return file

    def _import_amplitude_spectral_density(self):
        """ Automagically load an amplitude spectral density curve """
        self.frequency_array, self.asd_array = np.genfromtxt(self.asd_file).T

    def _import_power_spectral_density(self):
        """ Automagically load a power spectral density curve """
        self.frequency_array, self.psd_array = np.genfromtxt(self.psd_file).T

    def get_noise_realisation(self, number_of_samples, duration, *, random_state=None):
        """
        Generate frequency Gaussian noise scaled to the power spectral density.

        Parameters
        ==========
        sampling_frequency: float
            sampling frequency of noise
        duration: float
            duration of noise

        Returns
        =======
        array_like: frequency domain strain of this noise realisation
        array_like: frequencies related to the frequency domain strain

        """
        white_noise, frequencies = utils.safe_white_noise(number_of_samples, duration, random_state=random_state)
        with np.errstate(invalid="ignore"):
            frequency_domain_strain = self._power_spectral_density_interpolated(frequencies) ** 0.5 * white_noise
        xp = aac.array_namespace(frequency_domain_strain)
        out_of_bounds = (
            (frequencies < self.frequency_array.min())
            | (frequencies > self.frequency_array.max())
        )
        frequency_domain_strain = xpx.at(frequency_domain_strain, out_of_bounds).set(0j)
        return xp.nan_to_num(frequency_domain_strain), xp.asarray(frequencies)

    def set_array_backend(self, xp):
        """ Set the array backend for the cached arrays

        Parameters
        ==========
        xp: module
            The array backend to use for the cached arrays

        """
        self.frequency_array = xp.asarray(self.frequency_array)
        self._asd_array = xp.asarray(self._asd_array)
        self._psd_array = xp.asarray(self._psd_array)
        self._cache['frequency_array'] = xp.asarray(self._cache['frequency_array'])
        self._cache['psd_array'] = xp.asarray(self._cache['psd_array'])
        self._cache['asd_array'] = xp.asarray(self._cache['asd_array'])
        self._interpolate_power_spectral_density()


def power_spectral_density_to_autocovariance(
        frequency_array, psd_array, minimum_frequency, maximum_frequency,
        sampling_frequency=None, fill_value=None, fill_multiplier=100):
    """ Compute the autocovariance function from a one-sided power spectral density.

    The PSD below and above [`minimum_frequency`, `maximum_frequency`], and where
    it is not defined, is replaced by a constant, and the ACF is its inverse Fourier
    transform (Wiener-Khinchin theorem). By default, the PSD below (above) the band
    is set to `fill_multiplier` times the maximum of the PSD below (above) the band,
    or times the maximum of the whole PSD if it is not defined there.

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
    fill_value: float or tuple, optional
        The PSD value used outside the band, or the values (below, above) the band;
        overrides `fill_multiplier`
    fill_multiplier: float
        The PSD below (above) the band is set to this multiple of its maximum there

    Returns
    =======
    acf_array: numpy.ndarray
        The autocovariance function at lags 0, 1/f_s, 2/f_s, ...
    fill_value: tuple
        The PSD values used (below, above) the band
    fill_multiplier: float or None
        The multiplier used (None if `fill_value` was given explicitly)

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

    requested = (frequency_array >= minimum_frequency) & (frequency_array <= maximum_frequency)
    with np.errstate(invalid="ignore"):
        defined = np.isfinite(psd_array) & (psd_array > 0)
    band = requested & defined
    if not band.any():
        raise ValueError(
            f"The PSD is not defined anywhere in the band {minimum_frequency:g} - {maximum_frequency:g} Hz")
    if (requested & ~defined).any():
        logger.warning(
            f"The PSD is not defined at {np.sum(requested & ~defined)} frequencies in the band "
            f"{minimum_frequency:g} - {maximum_frequency:g} Hz (e.g. outside its range); these are "
            "filled like the frequencies outside the band, so the band used is effectively "
            f"{frequency_array[band].min():g} - {frequency_array[band].max():g} Hz")
    regions = dict(below=frequency_array < frequency_array[band].min(),
                   above=frequency_array > frequency_array[band].max())
    if fill_value is None:
        fill_value = list()
        for name, region in regions.items():
            measured = region & defined & (frequency_array > 0)
            if measured.any():
                fill_value.append(fill_multiplier * np.max(psd_array[measured]))
            else:
                fill_value.append(fill_multiplier * np.max(psd_array[defined & (frequency_array > 0)]))
                if (region & (frequency_array > 0)).any():
                    logger.warning(
                        f"The PSD is not defined {name} the band, so the fill value cannot be checked "
                        f"against the noise there; it must exceed the noise in the data {name} the band")
    else:
        fill_multiplier = None
        fill_value = np.broadcast_to(fill_value, 2)
        for (name, region), value in zip(regions.items(), fill_value):
            measured = region & defined & (frequency_array > 0)
            if measured.any() and value < np.max(psd_array[measured]):
                logger.warning(
                    f"The fill value {name} the band ({value:.2e}) is below the PSD there (maximum "
                    f"{np.max(psd_array[measured]):.2e}): the noise model underestimates the noise "
                    "there. This mainly biases the noise log-likelihood and evidences.")
    fill_value = tuple(float(value) for value in fill_value)
    psd_array = psd_array.copy()
    psd_array[regions["below"]] = fill_value[0]
    psd_array[regions["above"]] = fill_value[1]
    psd_array[~band & ~regions["below"] & ~regions["above"]] = max(fill_value)  # not defined inside the band
    acf_array = 0.5 * np.fft.irfft(psd_array) * psd_sampling_frequency
    return acf_array, fill_value, fill_multiplier
