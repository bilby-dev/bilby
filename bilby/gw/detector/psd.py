import os

import numpy as np
from scipy.interpolate import interp1d

from ...core import utils
from ...core.utils import logger
from .strain_data import InterferometerStrainData


class PowerSpectralDensity(object):

    def __init__(self, frequency_array=None, psd_array=None, asd_array=None,
                 psd_file=None, asd_file=None):
        """
        Instantiate a new PowerSpectralDensity object.

        Examples
        ========
        Using the `from` method directly (here `psd_file` is a string
        containing the path to the file to load):

        .. code-block:: python

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
    def from_aligo():
        logger.info("No power spectral density provided, using aLIGO,"
                    "zero detuning, high power.")
        return PowerSpectralDensity.from_power_spectral_density_file(psd_file='aLIGO_ZERO_DET_high_P_psd.txt')

    @property
    def psd_array(self):
        return self.__psd_array

    @psd_array.setter
    def psd_array(self, psd_array):
        self.__check_frequency_array_matches_density_array(psd_array)
        self.__psd_array = np.array(psd_array)
        self.__asd_array = psd_array ** 0.5
        self.__interpolate_power_spectral_density()

    @property
    def asd_array(self):
        return self.__asd_array

    @asd_array.setter
    def asd_array(self, asd_array):
        self.__check_frequency_array_matches_density_array(asd_array)
        self.__asd_array = np.array(asd_array)
        self.__psd_array = asd_array ** 2
        self.__interpolate_power_spectral_density()

    def __check_frequency_array_matches_density_array(self, density_array):
        if len(self.frequency_array) != len(density_array):
            raise ValueError('Provided spectral density does not match frequency array. Not updating.\n'
                             'Length spectral density {}\n Length frequency array {}\n'
                             .format(density_array, self.frequency_array))

    def __interpolate_power_spectral_density(self):
        """Interpolate the loaded power spectral density so it can be resampled
           for arbitrary frequency arrays.
        """
        self.__power_spectral_density_interpolated = interp1d(self.frequency_array,
                                                              self.psd_array,
                                                              bounds_error=False,
                                                              fill_value=np.inf)
        self._update_cache(self.frequency_array)

    def get_power_spectral_density_array(self, frequency_array):
        if not np.array_equal(frequency_array, self._cache['frequency_array']):
            self._update_cache(frequency_array=frequency_array)
        return self._cache['psd_array']

    def get_amplitude_spectral_density_array(self, frequency_array):
        if not np.array_equal(frequency_array, self._cache['frequency_array']):
            self._update_cache(frequency_array=frequency_array)
        return self._cache['asd_array']

    @property
    def power_spectral_density_interpolated(self):
        return self.__power_spectral_density_interpolated

    @property
    def asd_file(self):
        return self._asd_file

    @asd_file.setter
    def asd_file(self, asd_file):
        asd_file = self.__validate_file_name(file=asd_file)
        self._asd_file = asd_file
        if asd_file is not None:
            self.__import_amplitude_spectral_density()
            self.__check_file_was_asd_file()

    def __check_file_was_asd_file(self):
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
        psd_file = self.__validate_file_name(file=psd_file)
        self._psd_file = psd_file
        if psd_file is not None:
            self.__import_power_spectral_density()
            self.__check_file_was_psd_file()

    def __check_file_was_psd_file(self):
        if min(self.psd_array) > 1e-30:
            logger.warning("You specified a power spectral density file.")
            logger.warning("{} WARNING {}".format("*" * 30, "*" * 30))
            logger.warning("The minimum of the provided curve is {:.2e}.".format(min(self.psd_array)))
            logger.warning("You may have intended to provide this as an amplitude spectral density.")

    @staticmethod
    def __validate_file_name(file):
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

    def __import_amplitude_spectral_density(self):
        """ Automagically load an amplitude spectral density curve """
        self.frequency_array, self.asd_array = np.genfromtxt(self.asd_file).T

    def __import_power_spectral_density(self):
        """ Automagically load a power spectral density curve """
        self.frequency_array, self.psd_array = np.genfromtxt(self.psd_file).T

    def get_noise_realisation(self, sampling_frequency, duration):
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
        white_noise, frequencies = utils.create_white_noise(sampling_frequency, duration)
        with np.errstate(invalid="ignore"):
            frequency_domain_strain = self.__power_spectral_density_interpolated(frequencies) ** 0.5 * white_noise
        out_of_bounds = (frequencies < min(self.frequency_array)) | (frequencies > max(self.frequency_array))
        frequency_domain_strain[out_of_bounds] = 0 * (1 + 1j)
        return frequency_domain_strain, frequencies
