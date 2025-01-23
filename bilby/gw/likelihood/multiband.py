
import math
import numbers

import numpy as np

from .base import GravitationalWaveTransient
from ...core.utils import (
    logger, speed_of_light, solar_mass, radius_of_earth,
    gravitational_constant, round_up_to_power_of_two,
    recursively_load_dict_contents_from_group,
    recursively_save_dict_contents_to_group
)
from ..prior import CBCPriorDict
from ..utils import ln_i0


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
    weights: str or dict, optional
        Pre-computed multiband weights for calculating inner products.
    distance_marginalization: bool, optional
        If true, marginalize over distance in the likelihood. This uses a look up table calculated at run time. The
        distance prior is set to be a delta function at the minimum distance allowed in the prior being marginalised
        over.
    phase_marginalization: bool, optional
        If true, marginalize over phase in the likelihood. This is done analytically using a Bessel function. The phase
        prior is set to be a delta function at phase=0.
    priors: dict, bilby.prior.PriorDict
        A dictionary of priors containing at least the geocent_time prior
    time_marginalization: bool, optional
        If true, marginalize over time in the likelihood.
        If using time marginalisation and jitter_time is True a "jitter"
        parameter is added to the prior which modifies the position of the
        grid of times.
    jitter_time: bool, optional
        Whether to introduce a `time_jitter` parameter. This avoids either
        missing the likelihood peak, or introducing biases in the
        reconstructed time posterior due to an insufficient sampling frequency.
        Default is True.
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
            maximum_banding_frequency=None, minimum_banding_duration=0., weights=None,
            distance_marginalization=False, phase_marginalization=False, priors=None,
            time_marginalization=False, jitter_time=True, distance_marginalization_lookup_table=None,
            reference_frame="sky", time_reference="geocenter"
    ):
        super(MBGravitationalWaveTransient, self).__init__(
            interferometers=interferometers, waveform_generator=waveform_generator, priors=priors,
            distance_marginalization=distance_marginalization, phase_marginalization=phase_marginalization,
            time_marginalization=time_marginalization,
            distance_marginalization_lookup_table=distance_marginalization_lookup_table,
            jitter_time=jitter_time, reference_frame=reference_frame, time_reference=time_reference
        )
        if weights is None:
            self.reference_chirp_mass = reference_chirp_mass
            self.highest_mode = highest_mode
            self.linear_interpolation = linear_interpolation
            self.accuracy_factor = accuracy_factor
            self.time_offset = time_offset
            self.delta_f_end = delta_f_end
            self.maximum_banding_frequency = maximum_banding_frequency
            self.minimum_banding_duration = minimum_banding_duration
            self.setup_multibanding()
        else:
            if isinstance(weights, str):
                import h5py
                logger.info(f"Loading multiband weights from {weights}.")
                with h5py.File(weights, 'r') as f:
                    weights = recursively_load_dict_contents_from_group(f, '/')
            self.setup_multibanding_from_weights(weights)
        if self.time_marginalization:
            self._setup_time_marginalization_multiband()

    @property
    def reference_chirp_mass(self):
        return self._reference_chirp_mass

    @property
    def reference_chirp_mass_in_second(self):
        return gravitational_constant * self._reference_chirp_mass * solar_mass / speed_of_light**3.

    @reference_chirp_mass.setter
    def reference_chirp_mass(self, reference_chirp_mass):
        if isinstance(reference_chirp_mass, numbers.Number):
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
        if isinstance(highest_mode, numbers.Number):
            self._highest_mode = highest_mode
        else:
            raise TypeError("highest_mode must be a number")

    @property
    def linear_interpolation(self):
        return self._linear_interpolation

    @linear_interpolation.setter
    def linear_interpolation(self, linear_interpolation):
        if isinstance(linear_interpolation, bool) or isinstance(linear_interpolation, np.bool_):
            self._linear_interpolation = linear_interpolation
        else:
            raise TypeError("linear_interpolation must be a bool")

    @property
    def accuracy_factor(self):
        return self._accuracy_factor

    @accuracy_factor.setter
    def accuracy_factor(self, accuracy_factor):
        if isinstance(accuracy_factor, numbers.Number):
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
            if isinstance(time_offset, numbers.Number):
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
            if isinstance(delta_f_end, numbers.Number):
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
            if isinstance(maximum_banding_frequency, numbers.Number):
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
        if isinstance(minimum_banding_duration, numbers.Number):
            self._minimum_banding_duration = minimum_banding_duration
        else:
            raise TypeError("minimum_banding_duration must be a number")

    @property
    def minimum_frequency(self):
        return np.min([i.minimum_frequency for i in self.interferometers])

    @property
    def maximum_frequency(self):
        return np.max([i.maximum_frequency for i in self.interferometers])

    @property
    def number_of_bands(self):
        return len(self.durations)

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
        fb_dfb: 2-dimensional ndarray, which contain starting frequencies (f^(b) in the paper) and frequency scales for
        smoothing waveforms (\Delta f^(b) in the paper) of bands

        """
        self.durations = np.array([self.interferometers.duration])
        self.fb_dfb = [[self.minimum_frequency, 0.]]
        dnext = self.interferometers.duration / 2
        while dnext > max(self.time_offset, self.minimum_banding_duration):
            fnow, _ = self.fb_dfb[-1]
            fnext, dfnext = self._find_starting_frequency(dnext, fnow)
            if fnext is not None and fnext < min(self.maximum_frequency, self.maximum_banding_frequency):
                self.durations = np.append(self.durations, dnext)
                self.fb_dfb.append([fnext, dfnext])
                dnext /= 2
            else:
                break
        self.fb_dfb.append([self.maximum_frequency + self.delta_f_end, self.delta_f_end])
        self.fb_dfb = np.array(self.fb_dfb)
        logger.info("The total frequency range is divided into {} bands with frequency intervals of {}.".format(
            self.number_of_bands, ", ".join(["1/{} Hz".format(d) for d in self.durations])))

    def _setup_integers(self):
        """Set up integers needed for likelihood evaluations. This sets the following instance variables.

        Nbs: the numbers of samples of downsampled data (N^(b) in the paper)
        Mbs: the numbers of samples of shortened data (M^(b) in the paper)
        Ks_Ke: start and end frequency indices of bands (K^(b)_s and K^(b)_e in the paper)

        """
        self.Nbs = np.array([], dtype=int)
        self.Mbs = np.array([], dtype=int)
        self.Ks_Ke = []
        for b in range(self.number_of_bands):
            dnow = self.durations[b]
            fnow, dfnow = self.fb_dfb[b]
            fnext, _ = self.fb_dfb[b + 1]
            Nb = max(round_up_to_power_of_two(2. * (fnext * self.interferometers.duration + 1.)), 2**b)
            self.Nbs = np.append(self.Nbs, Nb)
            self.Mbs = np.append(self.Mbs, Nb // 2**b)
            self.Ks_Ke.append([math.ceil((fnow - dfnow) * dnow), math.floor(fnext * dnow)])
        self.Ks_Ke = np.array(self.Ks_Ke)

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
        for i in range(self.number_of_bands):
            d = self.durations[i]
            Ks, Ke = self.Ks_Ke[i]
            self.banded_frequency_points = np.append(self.banded_frequency_points, np.arange(Ks, Ke + 1) / d)
            end_idx = start_idx + Ke - Ks
            self.start_end_idxs.append([start_idx, end_idx])
            start_idx = end_idx + 1
        self.start_end_idxs = np.array(self.start_end_idxs)
        unique_frequencies, idxs = np.unique(self.banded_frequency_points, return_inverse=True)
        self.waveform_generator.waveform_arguments['frequencies'] = unique_frequencies
        self.unique_to_original_frequencies = idxs
        logger.info("The number of frequency points where waveforms are evaluated is {}.".format(
            len(unique_frequencies)))
        logger.info("The speed-up gain of multi-banding is {}.".format(
            (self.maximum_frequency - self.minimum_frequency) * self.interferometers.duration /
            len(unique_frequencies)))

    def _get_window_sequence(self, delta_f, start_idx, length, b):
        """Compute window function on frequencies with a fixed frequency interval

        Parameters
        ----------
        delta_f: float
            frequency interval
        start_idx: int
            starting frequency per delta_f
        length: int
            number of frequencies
        b: int
            band number

        Returns
        -------
        window_sequence: array

        """
        fnow, dfnow = self.fb_dfb[b]
        fnext, dfnext = self.fb_dfb[b + 1]

        window_sequence = np.zeros(length)
        increase_start = np.clip(
            math.floor((fnow - dfnow) / delta_f) - start_idx + 1, 0, length
        )
        unity_start = np.clip(math.ceil(fnow / delta_f) - start_idx, 0, length)
        decrease_start = np.clip(
            math.floor((fnext - dfnext) / delta_f) - start_idx + 1, 0, length
        )
        decrease_stop = np.clip(math.ceil(fnext / delta_f) - start_idx, 0, length)

        window_sequence[unity_start:decrease_start] = 1.

        # this if statement avoids overflow caused by vanishing dfnow
        if increase_start < unity_start:
            frequencies = (np.arange(increase_start, unity_start) + start_idx) * delta_f
            window_sequence[increase_start:unity_start] = (
                1. + np.cos(np.pi * (frequencies - fnow) / dfnow)
            ) / 2.

        if decrease_start < decrease_stop:
            frequencies = (np.arange(decrease_start, decrease_stop) + start_idx) * delta_f
            window_sequence[decrease_start:decrease_stop] = (
                1. - np.cos(np.pi * (frequencies - fnext) / dfnext)
            ) / 2.

        return window_sequence

    def _setup_linear_coefficients(self):
        """Set up coefficients by which waveforms are multiplied to compute (d, h)"""
        self.linear_coeffs = dict((ifo.name, np.array([])) for ifo in self.interferometers)
        N = self.Nbs[-1]
        for ifo in self.interferometers:
            logger.info("Pre-computing linear coefficients for {}".format(ifo.name))
            fddata = np.zeros(N // 2 + 1, dtype=complex)
            fddata[:len(ifo.frequency_domain_strain)][ifo.frequency_mask[:len(fddata)]] += \
                ifo.frequency_domain_strain[ifo.frequency_mask] / ifo.power_spectral_density_array[ifo.frequency_mask]
            for b in range(self.number_of_bands):
                Ks, Ke = self.Ks_Ke[b]
                windows = self._get_window_sequence(1. / self.durations[b], Ks, Ke - Ks + 1, b)
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
        original_duration = self.interferometers.duration

        for b in range(self.number_of_bands):
            logger.info(f"Pre-computing quadratic coefficients for the {b}-th band")
            _start, _end = self.start_end_idxs[b]
            banded_frequencies = self.banded_frequency_points[_start:_end + 1]
            prefactor = 4 * self.durations[b] / original_duration

            # precompute window values
            _fnow, _dfnow = self.fb_dfb[b]
            _fnext, _ = self.fb_dfb[b + 1]
            start_idx_in_band = math.ceil((_fnow - _dfnow) * original_duration)
            window_sequence = self._get_window_sequence(
                1 / original_duration,
                start_idx_in_band,
                math.floor(_fnext * original_duration) - start_idx_in_band + 1,
                b
            )

            for ifo in self.interferometers:
                end_idx_in_band = min(
                    start_idx_in_band + len(window_sequence) - 1,
                    len(ifo.power_spectral_density_array) - 1
                )
                _frequency_mask = ifo.frequency_mask[start_idx_in_band:end_idx_in_band + 1]
                window_over_psd = np.zeros(end_idx_in_band + 1 - start_idx_in_band)
                window_over_psd[_frequency_mask] = \
                    1. / ifo.power_spectral_density_array[start_idx_in_band:end_idx_in_band + 1][_frequency_mask]
                window_over_psd *= window_sequence[:len(window_over_psd)]

                coeffs = np.zeros(len(banded_frequencies))
                for k in range(len(coeffs) - 1):
                    if k == 0:
                        start_idx_in_sum = start_idx_in_band
                    else:
                        start_idx_in_sum = max(
                            start_idx_in_band,
                            math.ceil(original_duration * banded_frequencies[k])
                        )
                    if k == len(coeffs) - 2:
                        end_idx_in_sum = end_idx_in_band
                    else:
                        end_idx_in_sum = min(
                            end_idx_in_band,
                            math.ceil(original_duration * banded_frequencies[k + 1]) - 1
                        )
                    frequencies_in_sum = np.arange(start_idx_in_sum, end_idx_in_sum + 1) / original_duration
                    coeffs[k] += prefactor * np.sum(
                        (banded_frequencies[k + 1] - frequencies_in_sum) *
                        window_over_psd[start_idx_in_sum - start_idx_in_band:end_idx_in_sum - start_idx_in_band + 1]
                    )
                    coeffs[k + 1] += prefactor * np.sum(
                        (frequencies_in_sum - banded_frequencies[k]) *
                        window_over_psd[start_idx_in_sum - start_idx_in_band:end_idx_in_sum - start_idx_in_band + 1]
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
            full_inv_psds[:len(ifo.power_spectral_density_array)][ifo.frequency_mask[:len(full_inv_psds)]] = (
                1 / ifo.power_spectral_density_array[ifo.frequency_mask]
            )
            for b in range(self.number_of_bands):
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
        for b in range(self.number_of_bands):
            Ks, Ke = self.Ks_Ke[b]
            ws = self._get_window_sequence(1. / self.durations[b], Ks, Ke - Ks + 1, b)
            self.windows = np.append(self.windows, ws)
            self.square_root_windows = np.append(self.square_root_windows, np.sqrt(ws))

    @property
    def weights(self):
        _weights = {}
        for key in [
            "reference_chirp_mass", "highest_mode", "linear_interpolation",
            "accuracy_factor", "time_offset", "delta_f_end",
            "maximum_banding_frequency", "minimum_banding_duration",
            "durations", "fb_dfb", "Nbs", "Mbs", "Ks_Ke",
            "banded_frequency_points", "start_end_idxs",
            "unique_to_original_frequencies", "linear_coeffs"
        ]:
            _weights[key] = getattr(self, key)
        _weights["waveform_frequencies"] = \
            self.waveform_generator.waveform_arguments['frequencies']
        if self.linear_interpolation:
            _weights["quadratic_coeffs"] = self.quadratic_coeffs
        else:
            for key in ["Tbhats", "windows", "square_root_windows"]:
                _weights[key] = getattr(self, key)
            for key in ["wths", "hbcs", "Ibcs"]:
                _weights[key] = {}
                value = getattr(self, key)
                for ifo_name, data in value.items():
                    _weights[key][ifo_name] = dict((str(b), v) for b, v in enumerate(data))
        return _weights

    def save_weights(self, filename):
        """
        Save multiband weights into a .hdf5 file.

        Parameters
        ==========
        filename : str

        """
        import h5py
        if not filename.endswith(".hdf5"):
            filename += ".hdf5"
        logger.info(f"Saving multiband weights to {filename}")
        with h5py.File(filename, 'w') as f:
            recursively_save_dict_contents_to_group(f, '/', self.weights)

    def setup_multibanding_from_weights(self, weights):
        """
        Set multiband weights from dictionary-like weights

        Parameters
        ==========
        weights : dict

        """
        keys = list(weights.keys())
        # reference_chirp_mass needs to be set first as it is required for the setter of maximum_banding_frequency
        self.reference_chirp_mass = weights["reference_chirp_mass"]
        keys.remove("reference_chirp_mass")
        for key in keys:
            value = weights[key]
            if key in ["wths", "hbcs", "Ibcs"]:
                to_set = {}
                for ifo_name, data in value.items():
                    to_set[ifo_name] = [data[str(b)] for b in range(len(data.keys()))]
                setattr(self, key, to_set)
            elif key == "waveform_frequencies":
                self.waveform_generator.waveform_arguments['frequencies'] = weights["waveform_frequencies"]
            else:
                setattr(self, key, value)

    def _setup_time_marginalization_multiband(self):
        """This overwrites attributes set by _setup_time_marginalization of the base likelihood class"""
        N = self.Nbs[-1] // 2
        self._delta_tc = self.durations[0] / N
        self._times = \
            self.interferometers.start_time + np.arange(N) * self._delta_tc
        self.time_prior_array = \
            self.priors['geocent_time'].prob(self._times) * self._delta_tc
        # allocate array which is FFTed at each likelihood evaluation
        self._full_d_h = np.zeros(N, dtype=complex)
        # idxs to convert full frequency points to banded frequency points, used for filling _full_d_h.
        self._full_to_multiband = [int(f * self.durations[0]) for f in self.banded_frequency_points]
        self._beam_pattern_reference_time = (
            self.priors['geocent_time'].minimum + self.priors['geocent_time'].maximum
        ) / 2

    def calculate_snrs(self, waveform_polarizations, interferometer, return_array=True):
        """
        Compute the snrs

        Parameters
        ----------
        waveform_polarizations: dict
            A dictionary of waveform polarizations and the corresponding array
        interferometer: bilby.gw.detector.Interferometer
            The bilby interferometer object
        return_array: bool
            If true, calculate and return internal array objects
            (d_inner_h_array and optimal_snr_squared_array), otherwise
            these are returned as None.

        Returns
        -------
        calculated_snrs: _CalculatedSNRs
            An object containing the SNR quantities.

        """
        if self.time_marginalization:
            time_ref = self._beam_pattern_reference_time
        else:
            time_ref = self.parameters['geocent_time']

        strain = np.zeros(len(self.banded_frequency_points), dtype=complex)
        for mode in waveform_polarizations:
            response = interferometer.antenna_response(
                self.parameters['ra'], self.parameters['dec'],
                time_ref, self.parameters['psi'], mode
            )
            strain += waveform_polarizations[mode][self.unique_to_original_frequencies] * response

        dt = interferometer.time_delay_from_geocenter(
            self.parameters['ra'], self.parameters['dec'], time_ref)
        dt_geocent = self.parameters['geocent_time'] - interferometer.strain_data.start_time
        ifo_time = dt_geocent + dt
        strain *= np.exp(-1j * 2. * np.pi * self.banded_frequency_points * ifo_time)

        strain *= interferometer.calibration_model.get_calibration_factor(
            self.banded_frequency_points, prefix='recalib_{}_'.format(interferometer.name), **self.parameters)

        d_inner_h = np.conj(np.dot(strain, self.linear_coeffs[interferometer.name]))

        if self.linear_interpolation:
            optimal_snr_squared = np.vdot(
                np.real(strain * np.conjugate(strain)),
                self.quadratic_coeffs[interferometer.name]
            )
        else:
            optimal_snr_squared = 0.
            for b in range(self.number_of_bands):
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

        if return_array and self.time_marginalization:
            self._full_d_h[self._full_to_multiband] *= 0
            for b in range(self.number_of_bands):
                start_idx, end_idx = self.start_end_idxs[b]
                self._full_d_h[self._full_to_multiband[start_idx:end_idx + 1]] += \
                    strain[start_idx:end_idx + 1] * self.linear_coeffs[interferometer.name][start_idx:end_idx + 1]
            d_inner_h_array = np.fft.fft(self._full_d_h)
        else:
            d_inner_h_array = None

        return self._CalculatedSNRs(
            d_inner_h=d_inner_h,
            optimal_snr_squared=optimal_snr_squared.real,
            complex_matched_filter_snr=complex_matched_filter_snr,
            d_inner_h_array=d_inner_h_array,
        )

    def _rescale_signal(self, signal, new_distance):
        for mode in signal:
            signal[mode] *= self._ref_dist / new_distance

    def generate_time_sample_from_marginalized_likelihood(self, signal_polarizations=None):
        self.parameters.update(self.get_sky_frame_parameters())
        if signal_polarizations is None:
            signal_polarizations = \
                self.waveform_generator.frequency_domain_strain(self.parameters)

        snrs = self._CalculatedSNRs()

        for interferometer in self.interferometers:
            snrs += self.calculate_snrs(
                waveform_polarizations=signal_polarizations,
                interferometer=interferometer
            )
        d_inner_h = snrs.d_inner_h_array
        h_inner_h = snrs.optimal_snr_squared

        if self.distance_marginalization:
            time_log_like = self.distance_marginalized_likelihood(
                d_inner_h, h_inner_h)
        elif self.phase_marginalization:
            time_log_like = ln_i0(abs(d_inner_h)) - h_inner_h.real / 2
        else:
            time_log_like = (d_inner_h.real - h_inner_h.real / 2)

        times = self._times
        if self.jitter_time:
            times = times + self.parameters["time_jitter"]
        time_prior_array = self.priors['geocent_time'].prob(times)
        time_post = np.exp(time_log_like - max(time_log_like)) * time_prior_array
        time_post /= np.sum(time_post)
        return np.random.choice(times, p=time_post)
