
import math

import numpy as np

from .base import GravitationalWaveTransient
from ...core.utils import (
    logger, speed_of_light, solar_mass, radius_of_earth,
    gravitational_constant, round_up_to_power_of_two
)
from ..prior import CBCPriorDict


class MBGravitationalWaveTransient(GravitationalWaveTransient):
    """A multi-banded likelihood object

    This uses the method described in S. Morisaki, 2021, arXiv: 2104.07813.

    Parameters
    ----------
    interferometers: list, bilby.gw.detector.InterferometerList
        A list of `bilby.detector.Interferometer` instances - contains the detector data and power spectral densities
    waveform_generator: `bilby.waveform_generator.WaveformGenerator`
        An object which computes the frequency-domain strain of the signal, given some set of parameters
    reference_chirp_mass: float, optional
        A reference chirp mass for determining the frequency banding. This is set to prior minimum of chirp mass if
        not specified. Hence a CBCPriorDict object needs to be passed to priors when this parameter is not specified.
    highest_mode: int, optional
        The maximum magnetic number of gravitational-wave moments. Default is 2
    linear_interpolation: bool, optional
        If True, the linear-interpolation method is used for the computation of (h, h). If False, the IFFT-FFT method
        is used. Default is True.
    accuracy_factor: float, optional
        A parameter to determine the accuracy of multi-banding. The larger this factor is, the more accurate the
        approximation is. This corresponds to L in the paper. Default is 5.
    time_offset: float, optional
        (end time of data) - (maximum arrival time). If None, it is inferred from the prior of geocent time.
    delta_f_end: float, optional
        The frequency scale with which waveforms at the high-frequency end are smoothed. If None, it is determined from
        the prior of geocent time.
    maximum_banding_frequency: float, optional
        A maximum frequency for multi-banding. If specified, the low-frequency limit of a band does not exceed it.
    minimum_banding_duration: float, optional
        A minimum duration for multi-banding. If specified, the duration of a band is not smaller than it.
    distance_marginalization: bool, optional
        If true, marginalize over distance in the likelihood. This uses a look up table calculated at run time. The
        distance prior is set to be a delta function at the minimum distance allowed in the prior being marginalised
        over.
    phase_marginalization: bool, optional
        If true, marginalize over phase in the likelihood. This is done analytically using a Bessel function. The phase
        prior is set to be a delta function at phase=0.
    priors: dict, bilby.prior.PriorDict
        A dictionary of priors containing at least the geocent_time prior
    distance_marginalization_lookup_table: (dict, str), optional
        If a dict, dictionary containing the lookup_table, distance_array, (distance) prior_array, and
        reference_distance used to construct the table. If a string the name of a file containing these quantities. The
        lookup table is stored after construction in either the provided string or a default location:
        '.distance_marginalization_lookup_dmin{}_dmax{}_n{}.npz'
    reference_frame: (str, bilby.gw.detector.InterferometerList, list), optional
        Definition of the reference frame for the sky location.
        - "sky": sample in RA/dec, this is the default
        - e.g., "H1L1", ["H1", "L1"], InterferometerList(["H1", "L1"]):
          sample in azimuth and zenith, `azimuth` and `zenith` defined in the frame where the z-axis is aligned the the
          vector connecting H1 and L1.
    time_reference: str, optional
        Name of the reference for the sampled time parameter.
        - "geocent"/"geocenter": sample in the time at the Earth's center, this is the default
        - e.g., "H1": sample in the time of arrival at H1

    Returns
    -------
    Likelihood: `bilby.core.likelihood.Likelihood`
        A likelihood object, able to compute the likelihood of the data given some model parameters

    """
    def __init__(
            self, interferometers, waveform_generator, reference_chirp_mass=None, highest_mode=2,
            linear_interpolation=True, accuracy_factor=5, time_offset=None, delta_f_end=None,
            maximum_banding_frequency=None, minimum_banding_duration=0., distance_marginalization=False,
            phase_marginalization=False, priors=None, distance_marginalization_lookup_table=None,
            reference_frame="sky", time_reference="geocenter"
    ):
        super(MBGravitationalWaveTransient, self).__init__(
            interferometers=interferometers, waveform_generator=waveform_generator, priors=priors,
            distance_marginalization=distance_marginalization, phase_marginalization=phase_marginalization,
            time_marginalization=False, distance_marginalization_lookup_table=distance_marginalization_lookup_table,
            jitter_time=False, reference_frame=reference_frame, time_reference=time_reference
        )
        self.reference_chirp_mass = reference_chirp_mass
        self.highest_mode = highest_mode
        self.linear_interpolation = linear_interpolation
        self.accuracy_factor = accuracy_factor
        self.time_offset = time_offset
        self.delta_f_end = delta_f_end
        self.minimum_frequency = np.min([i.minimum_frequency for i in self.interferometers])
        self.maximum_frequency = np.max([i.maximum_frequency for i in self.interferometers])
        self.maximum_banding_frequency = maximum_banding_frequency
        self.minimum_banding_duration = minimum_banding_duration
        self.setup_multibanding()

    @property
    def reference_chirp_mass(self):
        return self._reference_chirp_mass

    @property
    def reference_chirp_mass_in_second(self):
        return gravitational_constant * self._reference_chirp_mass * solar_mass / speed_of_light**3.

    @reference_chirp_mass.setter
    def reference_chirp_mass(self, reference_chirp_mass):
        if isinstance(reference_chirp_mass, int) or isinstance(reference_chirp_mass, float):
            self._reference_chirp_mass = reference_chirp_mass
        else:
            logger.info(
                "No int or float number has been passed to reference_chirp_mass. "
                "Checking prior minimum of chirp mass ..."
            )
            if not isinstance(self.priors, CBCPriorDict):
                raise TypeError(
                    f"priors: {self.priors} is not CBCPriorDict. Prior minimum of chirp mass can not be obtained."
                )
            self._reference_chirp_mass = self.priors.minimum_chirp_mass
            if self._reference_chirp_mass is None:
                raise Exception(
                    "Prior minimum of chirp mass can not be determined as priors does not contain necessary mass "
                    "parameters."
                )
            logger.info(
                "reference_chirp_mass is automatically set to prior minimum of chirp mass: "
                f"{self._reference_chirp_mass}."
            )

    @property
    def highest_mode(self):
        return self._highest_mode

    @highest_mode.setter
    def highest_mode(self, highest_mode):
        if isinstance(highest_mode, int) or isinstance(highest_mode, float):
            self._highest_mode = highest_mode
        else:
            raise TypeError("highest_mode must be a number")

    @property
    def linear_interpolation(self):
        return self._linear_interpolation

    @linear_interpolation.setter
    def linear_interpolation(self, linear_interpolation):
        if isinstance(linear_interpolation, bool):
            self._linear_interpolation = linear_interpolation
        else:
            raise TypeError("linear_interpolation must be a bool")

    @property
    def accuracy_factor(self):
        return self._accuracy_factor

    @accuracy_factor.setter
    def accuracy_factor(self, accuracy_factor):
        if isinstance(accuracy_factor, int) or isinstance(accuracy_factor, float):
            self._accuracy_factor = accuracy_factor
        else:
            raise TypeError("accuracy_factor must be a number")

    @property
    def time_offset(self):
        return self._time_offset

    @time_offset.setter
    def time_offset(self, time_offset):
        """
        This sets the time offset assumed when frequency bands are constructed. The default value is (the
        maximum offset of geocent time in the prior range) +  (light-traveling time of the Earth). If the
        prior does not contain 'geocent_time', 2.12 seconds is used. It is calculated assuming that the
        maximum offset of geocent time is 2.1 seconds, which is the value for the standard prior used by
        LIGO-Virgo-KAGRA.
        """
        time_parameter = self.time_reference + "_time"
        if time_parameter == "geocent_time":
            safety = radius_of_earth / speed_of_light
        else:
            safety = 2 * radius_of_earth / speed_of_light
        if time_offset is not None:
            if isinstance(time_offset, int) or isinstance(time_offset, float):
                self._time_offset = time_offset
            else:
                raise TypeError("time_offset must be a number")
        elif self.priors is not None and time_parameter in self.priors:
            self._time_offset = (
                self.interferometers.start_time + self.interferometers.duration
                - self.priors[time_parameter].minimum + safety
            )
        else:
            self._time_offset = 2.12
            logger.warning("time offset can not be inferred. Use the standard time offset of {} seconds.".format(
                self._time_offset))

    @property
    def delta_f_end(self):
        return self._delta_f_end

    @delta_f_end.setter
    def delta_f_end(self, delta_f_end):
        """
        This sets the frequency scale of tapering the high-frequency end of waveform, to avoid the issues of
        abrupt termination of waveform described in Sec. 2. F of arXiv: 2104.07813. This needs to be much
        larger than the inverse of the minimum time offset, and the default value is 100 times of that. If
        the prior does not contain 'geocent_time' and the minimum time offset can not be computed, 53Hz is
        used. It is computed assuming that the minimum offset of geocent time is 1.9 seconds, which is the
        value for the standard prior used by LIGO-Virgo-KAGRA.
        """
        time_parameter = self.time_reference + "_time"
        if time_parameter == "geocent_time":
            safety = radius_of_earth / speed_of_light
        else:
            safety = 2 * radius_of_earth / speed_of_light
        if delta_f_end is not None:
            if isinstance(delta_f_end, int) or isinstance(delta_f_end, float):
                self._delta_f_end = delta_f_end
            else:
                raise TypeError("delta_f_end must be a number")
        elif self.priors is not None and time_parameter in self.priors:
            self._delta_f_end = 100 / (
                self.interferometers.start_time + self.interferometers.duration
                - self.priors[time_parameter].maximum - safety
            )
        else:
            self._delta_f_end = 53.
            logger.warning("delta_f_end can not be inferred. Use the standard delta_f_end of {} Hz.".format(
                self._delta_f_end))

    @property
    def maximum_banding_frequency(self):
        return self._maximum_banding_frequency

    @maximum_banding_frequency.setter
    def maximum_banding_frequency(self, maximum_banding_frequency):
        r"""
        This sets the upper limit on a starting frequency of a band. The default value is the frequency at
        which f - 1 / \sqrt(- d\tau / df) starts to decrease, because the bisection search of the starting
        frequency does not work from that frequency. The stationary phase approximation is not valid at such
        a high frequency, which can break down the approximation. It is calculated from the 0PN formula of
        time-to-merger \tau(f). The user-specified frequency is used if it is lower than that frequency.
        """
        fmax_tmp = (
            (15 / 968)**(3 / 5) * (self.highest_mode / (2 * np.pi))**(8 / 5)
            / self.reference_chirp_mass_in_second
        )
        if maximum_banding_frequency is not None:
            if isinstance(maximum_banding_frequency, int) or isinstance(maximum_banding_frequency, float):
                if maximum_banding_frequency < fmax_tmp:
                    fmax_tmp = maximum_banding_frequency
                else:
                    logger.warning("The input maximum_banding_frequency is too large."
                                   "It is set to be {} Hz.".format(fmax_tmp))
            else:
                raise TypeError("maximum_banding_frequency must be a number")
        self._maximum_banding_frequency = fmax_tmp

    @property
    def minimum_banding_duration(self):
        return self._minimum_banding_duration

    @minimum_banding_duration.setter
    def minimum_banding_duration(self, minimum_banding_duration):
        if isinstance(minimum_banding_duration, int) or isinstance(minimum_banding_duration, float):
            self._minimum_banding_duration = minimum_banding_duration
        else:
            raise TypeError("minimum_banding_duration must be a number")

    def setup_multibanding(self):
        """Set up frequency bands and coefficients needed for likelihood evaluations"""
        self._setup_frequency_bands()
        self._setup_integers()
        self._setup_waveform_frequency_points()
        self._setup_linear_coefficients()
        if self.linear_interpolation:
            self._setup_quadratic_coefficients_linear_interp()
        else:
            self._setup_quadratic_coefficients_ifft_fft()

    def _tau(self, f):
        """Compute time-to-merger from the input frequency. This uses the 0PN formula.

        Parameters
        ----------
        f: float
            input frequency

        Returns
        -------
        tau: float
            time-to-merger

        """
        f_22 = 2 * f / self.highest_mode
        return (
            5 / 256 * self.reference_chirp_mass_in_second
            * (np.pi * self.reference_chirp_mass_in_second * f_22) ** (-8 / 3)
        )

    def _dtaudf(self, f):
        """Compute the derivative of time-to-merger with respect to a starting frequency. This uses the 0PN formula.

        Parameters
        ----------
        f: float
            input frequency

        Returns
        -------
        dtaudf: float
            derivative of time-to-merger

        """
        f_22 = 2 * f / self.highest_mode
        return (
            -5 / 96 * self.reference_chirp_mass_in_second
            * (np.pi * self.reference_chirp_mass_in_second * f_22) ** (-8. / 3.) / f
        )

    def _find_starting_frequency(self, duration, fnow):
        """Find the starting frequency of the next band satisfying (10) and
        (51) of arXiv: 2104.07813.

        Parameters
        ----------
        duration: float
            duration of the next band
        fnow: float
            starting frequency of the current band

        Returns
        -------
        fnext: float or None
            starting frequency of the next band. None if a frequency satisfying the conditions does not exist.
        dfnext: float or None
            frequency scale with which waveforms are smoothed. None if a frequency satisfying the conditions does not
            exist.

        """
        def _is_above_fnext(f):
            """This function returns True if f > fnext"""
            cond1 = (
                duration - self.time_offset - self._tau(f)
                - self.accuracy_factor * np.sqrt(-self._dtaudf(f))
            ) > 0
            cond2 = f - 1. / np.sqrt(-self._dtaudf(f)) - fnow > 0
            return cond1 and cond2
        # Bisection search for fnext
        fmin, fmax = fnow, self.maximum_banding_frequency
        if not _is_above_fnext(fmax):
            return None, None
        while fmax - fmin > 1e-2 / duration:
            f = (fmin + fmax) / 2.
            if _is_above_fnext(f):
                fmax = f
            else:
                fmin = f
        return f, 1. / np.sqrt(-self._dtaudf(f))

    def _setup_frequency_bands(self):
        r"""Set up frequency bands. The durations of bands geometrically decrease T, T/2. T/4, ..., where T is the
        original duration. This sets the following instance variables.

        durations: durations of bands (T^(b) in the paper)
        fb_dfb: the list of tuples, which contain starting frequencies (f^(b) in the paper) and frequency scales for
        smoothing waveforms (\Delta f^(b) in the paper) of bands

        """
        self.durations = [self.interferometers.duration]
        self.fb_dfb = [(self.minimum_frequency, 0.)]
        dnext = self.interferometers.duration / 2
        while dnext > max(self.time_offset, self.minimum_banding_duration):
            fnow, _ = self.fb_dfb[-1]
            fnext, dfnext = self._find_starting_frequency(dnext, fnow)
            if fnext is not None and fnext < min(self.maximum_frequency, self.maximum_banding_frequency):
                self.durations.append(dnext)
                self.fb_dfb.append((fnext, dfnext))
                dnext /= 2
            else:
                break
        self.fb_dfb.append((self.maximum_frequency + self.delta_f_end, self.delta_f_end))
        logger.info("The total frequency range is divided into {} bands with frequency intervals of {}.".format(
            len(self.durations), ", ".join(["1/{} Hz".format(d) for d in self.durations])))

    def _setup_integers(self):
        """Set up integers needed for likelihood evaluations. This sets the following instance variables.

        Nbs: the numbers of samples of downsampled data (N^(b) in the paper)
        Mbs: the numbers of samples of shortened data (M^(b) in the paper)
        Ks_Ke: start and end frequency indices of bands (K^(b)_s and K^(b)_e in the paper)

        """
        self.Nbs = []
        self.Mbs = []
        self.Ks_Ke = []
        for b in range(len(self.durations)):
            dnow = self.durations[b]
            fnow, dfnow = self.fb_dfb[b]
            fnext, _ = self.fb_dfb[b + 1]
            Nb = max(round_up_to_power_of_two(2. * (fnext * self.interferometers.duration + 1.)), 2**b)
            self.Nbs.append(Nb)
            self.Mbs.append(Nb // 2**b)
            self.Ks_Ke.append((math.ceil((fnow - dfnow) * dnow), math.floor(fnext * dnow)))

    def _setup_waveform_frequency_points(self):
        """Set up frequency points where waveforms are evaluated. Frequency points are reordered because some waveform
        models raise an error if the input frequencies are not increasing. This adds frequency_points into the
        waveform_arguments of waveform_generator. This sets the following instance variables.

        banded_frequency_points: ndarray of total banded frequency points
        start_end_idxs: list of tuples containing start and end indices of each band
        unique_to_original_frequencies: indices converting unique frequency
        points into the original duplicated banded frequencies

        """
        self.banded_frequency_points = np.array([])
        self.start_end_idxs = []
        start_idx = 0
        for i in range(len(self.fb_dfb) - 1):
            d = self.durations[i]
            Ks, Ke = self.Ks_Ke[i]
            self.banded_frequency_points = np.append(self.banded_frequency_points, np.arange(Ks, Ke + 1) / d)
            end_idx = start_idx + Ke - Ks
            self.start_end_idxs.append((start_idx, end_idx))
            start_idx = end_idx + 1
        unique_frequencies, idxs = np.unique(self.banded_frequency_points, return_inverse=True)
        self.waveform_generator.waveform_arguments['frequencies'] = unique_frequencies
        self.unique_to_original_frequencies = idxs
        logger.info("The number of frequency points where waveforms are evaluated is {}.".format(
            len(unique_frequencies)))
        logger.info("The speed-up gain of multi-banding is {}.".format(
            (self.maximum_frequency - self.minimum_frequency) * self.interferometers.duration /
            len(unique_frequencies)))

    def _window(self, f, b):
        """Compute window function in the b-th band

        Parameters
        ----------
        f: float or ndarray
            frequency at which the window function is computed
        b: int

        Returns
        -------
        window: float
            window function at f
        """
        fnow, dfnow = self.fb_dfb[b]
        fnext, dfnext = self.fb_dfb[b + 1]

        @np.vectorize
        def _vectorized_window(f):
            if fnow - dfnow < f < fnow:
                return (1. + np.cos(np.pi * (f - fnow) / dfnow)) / 2.
            elif fnow <= f <= fnext - dfnext:
                return 1.
            elif fnext - dfnext < f < fnext:
                return (1. - np.cos(np.pi * (f - fnext) / dfnext)) / 2.
            else:
                return 0.

        return _vectorized_window(f)

    def _setup_linear_coefficients(self):
        """Set up coefficients by which waveforms are multiplied to compute (d, h)"""
        self.linear_coeffs = dict((ifo.name, np.array([])) for ifo in self.interferometers)
        N = self.Nbs[-1]
        for ifo in self.interferometers:
            logger.info("Pre-computing linear coefficients for {}".format(ifo.name))
            fddata = np.zeros(N // 2 + 1, dtype=complex)
            fddata[:len(ifo.frequency_domain_strain)][ifo.frequency_mask] += \
                ifo.frequency_domain_strain[ifo.frequency_mask] / ifo.power_spectral_density_array[ifo.frequency_mask]
            for b in range(len(self.fb_dfb) - 1):
                start_idx, end_idx = self.start_end_idxs[b]
                windows = self._window(self.banded_frequency_points[start_idx:end_idx + 1], b)
                fddata_in_ith_band = np.copy(fddata[:int(self.Nbs[b] / 2 + 1)])
                fddata_in_ith_band[-1] = 0.  # zeroing data at the Nyquist frequency
                tddata = np.fft.irfft(fddata_in_ith_band)[-self.Mbs[b]:]
                Ks, Ke = self.Ks_Ke[b]
                fddata_in_ith_band = np.fft.rfft(tddata)[Ks:Ke + 1]
                self.linear_coeffs[ifo.name] = np.append(
                    self.linear_coeffs[ifo.name], (4. / self.durations[b]) * windows * np.conj(fddata_in_ith_band))

    def _setup_quadratic_coefficients_linear_interp(self):
        """Set up coefficients by which the squares of waveforms are multiplied to compute (h, h) for the
        linear-interpolation algorithm"""
        logger.info("Linear-interpolation algorithm is used for (h, h).")
        self.quadratic_coeffs = dict((ifo.name, np.array([])) for ifo in self.interferometers)
        N = self.Nbs[-1]
        for ifo in self.interferometers:
            logger.info("Pre-computing quadratic coefficients for {}".format(ifo.name))
            full_frequencies = np.arange(N // 2 + 1) / ifo.duration
            full_inv_psds = np.zeros(N // 2 + 1)
            full_inv_psds[:len(ifo.power_spectral_density_array)][ifo.frequency_mask] = \
                1. / ifo.power_spectral_density_array[ifo.frequency_mask]
            for i in range(len(self.fb_dfb) - 1):
                start_idx, end_idx = self.start_end_idxs[i]
                banded_frequencies = self.banded_frequency_points[start_idx:end_idx + 1]
                coeffs = np.zeros(len(banded_frequencies))
                for k in range(len(coeffs) - 1):
                    if k == 0:
                        start_idx_in_sum = 0
                    else:
                        start_idx_in_sum = math.ceil(ifo.duration * banded_frequencies[k])
                    if k == len(coeffs) - 2:
                        end_idx_in_sum = len(full_frequencies) - 1
                    else:
                        end_idx_in_sum = math.ceil(ifo.duration * banded_frequencies[k + 1]) - 1
                    window_over_psd = (
                        full_inv_psds[start_idx_in_sum:end_idx_in_sum + 1]
                        * self._window(full_frequencies[start_idx_in_sum:end_idx_in_sum + 1], i)
                    )
                    frequencies_in_sum = full_frequencies[start_idx_in_sum:end_idx_in_sum + 1]
                    coeffs[k] += 4 * self.durations[i] / ifo.duration * np.sum(
                        (banded_frequencies[k + 1] - frequencies_in_sum) * window_over_psd
                    )
                    coeffs[k + 1] += 4 * self.durations[i] / ifo.duration * np.sum(
                        (frequencies_in_sum - banded_frequencies[k]) * window_over_psd
                    )
                self.quadratic_coeffs[ifo.name] = np.append(self.quadratic_coeffs[ifo.name], coeffs)

    def _setup_quadratic_coefficients_ifft_fft(self):
        """Set up coefficients needed for the IFFT-FFT algorithm to compute (h, h)"""
        logger.info("IFFT-FFT algorithm is used for (h, h).")
        N = self.Nbs[-1]
        # variables defined below correspond to \hat{N}^(b), \hat{T}^(b), \tilde{I}^(b)_{c, k}, h^(b)_{c, m} and
        # \sqrt{w^(b)(f^(b)_k)} \tilde{h}(f^(b)_k) in the paper
        Nhatbs = [min(2 * Mb, Nb) for Mb, Nb in zip(self.Mbs, self.Nbs)]
        self.Tbhats = [self.interferometers.duration * Nbhat / Nb for Nb, Nbhat in zip(self.Nbs, Nhatbs)]
        self.Ibcs = dict((ifo.name, []) for ifo in self.interferometers)
        self.hbcs = dict((ifo.name, []) for ifo in self.interferometers)
        self.wths = dict((ifo.name, []) for ifo in self.interferometers)
        for ifo in self.interferometers:
            logger.info("Pre-computing quadratic coefficients for {}".format(ifo.name))
            full_inv_psds = np.zeros(N // 2 + 1)
            full_inv_psds[:len(ifo.power_spectral_density_array)][ifo.frequency_mask] = (
                1 / ifo.power_spectral_density_array[ifo.frequency_mask]
            )
            for b in range(len(self.fb_dfb) - 1):
                Imb = np.fft.irfft(full_inv_psds[:self.Nbs[b] // 2 + 1])
                half_length = Nhatbs[b] // 2
                Imbc = np.append(Imb[:half_length + 1], Imb[-(Nhatbs[b] - half_length - 1):])
                self.Ibcs[ifo.name].append(np.fft.rfft(Imbc))
                # Allocate arrays for IFFT-FFT operations
                self.hbcs[ifo.name].append(np.zeros(Nhatbs[b]))
                self.wths[ifo.name].append(np.zeros(self.Mbs[b] // 2 + 1, dtype=complex))
        # precompute windows and their squares
        self.windows = np.array([])
        self.square_root_windows = np.array([])
        for b in range(len(self.fb_dfb) - 1):
            start, end = self.start_end_idxs[b]
            ws = self._window(self.banded_frequency_points[start:end + 1], b)
            self.windows = np.append(self.windows, ws)
            self.square_root_windows = np.append(self.square_root_windows, np.sqrt(ws))

    def calculate_snrs(self, waveform_polarizations, interferometer):
        """
        Compute the snrs for multi-banding

        Parameters
        ----------
        waveform_polarizations: waveform
        interferometer: bilby.gw.detector.Interferometer

        Returns
        -------
        snrs: named tuple of snrs

        """
        strain = np.zeros(len(self.banded_frequency_points), dtype=complex)
        for mode in waveform_polarizations:
            response = interferometer.antenna_response(
                self.parameters['ra'], self.parameters['dec'],
                self.parameters['geocent_time'], self.parameters['psi'],
                mode
            )
            strain += waveform_polarizations[mode][self.unique_to_original_frequencies] * response

        dt = interferometer.time_delay_from_geocenter(
            self.parameters['ra'], self.parameters['dec'],
            self.parameters['geocent_time'])
        dt_geocent = self.parameters['geocent_time'] - interferometer.strain_data.start_time
        ifo_time = dt_geocent + dt

        calib_factor = interferometer.calibration_model.get_calibration_factor(
            self.banded_frequency_points, prefix='recalib_{}_'.format(interferometer.name), **self.parameters)

        strain *= np.exp(-1j * 2. * np.pi * self.banded_frequency_points * ifo_time)
        strain *= calib_factor

        d_inner_h = np.dot(strain, self.linear_coeffs[interferometer.name])

        if self.linear_interpolation:
            optimal_snr_squared = np.vdot(
                np.real(strain * np.conjugate(strain)),
                self.quadratic_coeffs[interferometer.name]
            )
        else:
            optimal_snr_squared = 0.
            for b in range(len(self.fb_dfb) - 1):
                Ks, Ke = self.Ks_Ke[b]
                start_idx, end_idx = self.start_end_idxs[b]
                Mb = self.Mbs[b]
                if b == 0:
                    optimal_snr_squared += (4. / self.interferometers.duration) * np.vdot(
                        np.real(strain[start_idx:end_idx + 1] * np.conjugate(strain[start_idx:end_idx + 1])),
                        interferometer.frequency_mask[Ks:Ke + 1] * self.windows[start_idx:end_idx + 1]
                        / interferometer.power_spectral_density_array[Ks:Ke + 1])
                else:
                    self.wths[interferometer.name][b][Ks:Ke + 1] = (
                        self.square_root_windows[start_idx:end_idx + 1] * strain[start_idx:end_idx + 1]
                    )
                    self.hbcs[interferometer.name][b][-Mb:] = np.fft.irfft(self.wths[interferometer.name][b])
                    thbc = np.fft.rfft(self.hbcs[interferometer.name][b])
                    optimal_snr_squared += (4. / self.Tbhats[b]) * np.vdot(
                        np.real(thbc * np.conjugate(thbc)), self.Ibcs[interferometer.name][b])

        complex_matched_filter_snr = d_inner_h / (optimal_snr_squared**0.5)

        return self._CalculatedSNRs(
            d_inner_h=d_inner_h, optimal_snr_squared=optimal_snr_squared,
            complex_matched_filter_snr=complex_matched_filter_snr,
            d_inner_h_squared_tc_array=None,
            d_inner_h_array=None,
            optimal_snr_squared_array=None)

    def _rescale_signal(self, signal, new_distance):
        for mode in signal:
            signal[mode] *= self._ref_dist / new_distance
