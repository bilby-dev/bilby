
import json

import numpy as np

from .base import GravitationalWaveTransient
from ...core.utils import BilbyJsonEncoder, decode_bilby_json
from ...core.utils import (
    logger, create_frequency_series, speed_of_light, radius_of_earth
)
from ..prior import CBCPriorDict
from ..utils import build_roq_weights


class ROQGravitationalWaveTransient(GravitationalWaveTransient):
    """A reduced order quadrature likelihood object

    This uses the method described in Smith et al., (2016) Phys. Rev. D 94,
    044031. A public repository of the ROQ data is available from
    https://git.ligo.org/lscsoft/ROQ_data.

    Parameters
    ==========
    interferometers: list, bilby.gw.detector.InterferometerList
        A list of `bilby.detector.Interferometer` instances - contains the
        detector data and power spectral densities
    waveform_generator: `bilby.waveform_generator.WaveformGenerator`
        An object which computes the frequency-domain strain of the signal,
        given some set of parameters
    linear_matrix: str, array_like
        Either a string point to the file from which to load the linear_matrix
        array, or the array itself.
    quadratic_matrix: str, array_like
        Either a string point to the file from which to load the
        quadratic_matrix array, or the array itself.
    roq_params: str, array_like
        Parameters describing the domain of validity of the ROQ basis.
    roq_params_check: bool
        If true, run tests using the roq_params to check the prior and data are
        valid for the ROQ
    roq_scale_factor: float
        The ROQ scale factor used.
    priors: dict, bilby.prior.PriorDict
        A dictionary of priors containing at least the geocent_time prior
        Warning: when using marginalisation the dict is overwritten which will change the
        the dict you are passing in. If this behaviour is undesired, pass `priors.copy()`.
    time_marginalization: bool, optional
        If true, marginalize over time in the likelihood.
        The spacing of time samples can be specified through delta_tc.
        If using time marginalisation and jitter_time is True a "jitter"
        parameter is added to the prior which modifies the position of the
        grid of times.
    jitter_time: bool, optional
        Whether to introduce a `time_jitter` parameter. This avoids either
        missing the likelihood peak, or introducing biases in the
        reconstructed time posterior due to an insufficient sampling frequency.
        Default is False, however using this parameter is strongly encouraged.
    delta_tc: float, optional
        The spacing of time samples for time marginalization. If not specified,
        it is determined based on the signal-to-noise ratio of signal.
    distance_marginalization_lookup_table: (dict, str), optional
        If a dict, dictionary containing the lookup_table, distance_array,
        (distance) prior_array, and reference_distance used to construct
        the table.
        If a string the name of a file containing these quantities.
        The lookup table is stored after construction in either the
        provided string or a default location:
        '.distance_marginalization_lookup_dmin{}_dmax{}_n{}.npz'
    reference_frame: (str, bilby.gw.detector.InterferometerList, list), optional
        Definition of the reference frame for the sky location.
        - "sky": sample in RA/dec, this is the default
        - e.g., "H1L1", ["H1", "L1"], InterferometerList(["H1", "L1"]):
          sample in azimuth and zenith, `azimuth` and `zenith` defined in the
          frame where the z-axis is aligned the the vector connecting H1
          and L1.
    time_reference: str, optional
        Name of the reference for the sampled time parameter.
        - "geocent"/"geocenter": sample in the time at the Earth's center,
          this is the default
        - e.g., "H1": sample in the time of arrival at H1

    """
    def __init__(
            self, interferometers, waveform_generator, priors,
            weights=None, linear_matrix=None, quadratic_matrix=None,
            roq_params=None, roq_params_check=True, roq_scale_factor=1,
            distance_marginalization=False, phase_marginalization=False,
            time_marginalization=False, jitter_time=True, delta_tc=None,
            distance_marginalization_lookup_table=None,
            reference_frame="sky", time_reference="geocenter"

    ):
        self._delta_tc = delta_tc
        super(ROQGravitationalWaveTransient, self).__init__(
            interferometers=interferometers,
            waveform_generator=waveform_generator, priors=priors,
            distance_marginalization=distance_marginalization,
            phase_marginalization=phase_marginalization,
            time_marginalization=time_marginalization,
            distance_marginalization_lookup_table=distance_marginalization_lookup_table,
            jitter_time=jitter_time,
            reference_frame=reference_frame,
            time_reference=time_reference
        )

        self.roq_params_check = roq_params_check
        self.roq_scale_factor = roq_scale_factor
        if isinstance(roq_params, np.ndarray) or roq_params is None:
            self.roq_params = roq_params
        elif isinstance(roq_params, str):
            self.roq_params_file = roq_params
            self.roq_params = np.genfromtxt(roq_params, names=True)
        else:
            raise TypeError("roq_params should be array or str")
        if isinstance(weights, dict):
            self.weights = weights
        elif isinstance(weights, str):
            self.weights = self.load_weights(weights)
        else:
            self.weights = dict()
            if isinstance(linear_matrix, str):
                logger.info(
                    "Loading linear matrix from {}".format(linear_matrix))
                linear_matrix = np.load(linear_matrix).T
            if isinstance(quadratic_matrix, str):
                logger.info(
                    "Loading quadratic_matrix from {}".format(quadratic_matrix))
                quadratic_matrix = np.load(quadratic_matrix).T
            self._set_weights(linear_matrix=linear_matrix,
                              quadratic_matrix=quadratic_matrix)
        self.frequency_nodes_linear = \
            waveform_generator.waveform_arguments['frequency_nodes_linear']
        self.frequency_nodes_quadratic = \
            waveform_generator.waveform_arguments['frequency_nodes_quadratic']

    def _setup_time_marginalization(self):
        if self._delta_tc is None:
            self._delta_tc = self._get_time_resolution()
        tcmin = self.priors['geocent_time'].minimum
        tcmax = self.priors['geocent_time'].maximum
        number_of_time_samples = int(np.ceil((tcmax - tcmin) / self._delta_tc))
        # adjust delta tc so that the last time sample has an equal weight
        self._delta_tc = (tcmax - tcmin) / number_of_time_samples
        logger.info(
            "delta tc for time marginalization = {} seconds.".format(self._delta_tc))
        self._times = tcmin + self._delta_tc / 2. + np.arange(number_of_time_samples) * self._delta_tc
        self._beam_pattern_reference_time = (tcmin + tcmax) / 2.

    def calculate_snrs(self, waveform_polarizations, interferometer):
        """
        Compute the snrs for ROQ

        Parameters
        ==========
        waveform_polarizations: waveform
        interferometer: bilby.gw.detector.Interferometer

        """

        if self.time_marginalization:
            time_ref = self._beam_pattern_reference_time
        else:
            time_ref = self.parameters['geocent_time']

        h_linear = np.zeros(len(self.frequency_nodes_linear), dtype=complex)
        h_quadratic = np.zeros(len(self.frequency_nodes_quadratic), dtype=complex)
        for mode in waveform_polarizations['linear']:
            response = interferometer.antenna_response(
                self.parameters['ra'], self.parameters['dec'],
                self.parameters['geocent_time'], self.parameters['psi'],
                mode
            )
            h_linear += waveform_polarizations['linear'][mode] * response
            h_quadratic += waveform_polarizations['quadratic'][mode] * response

        calib_linear = interferometer.calibration_model.get_calibration_factor(
            self.frequency_nodes_linear,
            prefix='recalib_{}_'.format(interferometer.name), **self.parameters)
        calib_quadratic = interferometer.calibration_model.get_calibration_factor(
            self.frequency_nodes_quadratic,
            prefix='recalib_{}_'.format(interferometer.name), **self.parameters)

        h_linear *= calib_linear
        h_quadratic *= calib_quadratic

        optimal_snr_squared = \
            np.vdot(np.abs(h_quadratic)**2, self.weights[interferometer.name + '_quadratic'])

        dt = interferometer.time_delay_from_geocenter(
            self.parameters['ra'], self.parameters['dec'], time_ref)

        if not self.time_marginalization:
            dt_geocent = self.parameters['geocent_time'] - interferometer.strain_data.start_time
            ifo_time = dt_geocent + dt

            indices, in_bounds = self._closest_time_indices(
                ifo_time, self.weights['time_samples'])
            if not in_bounds:
                logger.debug("SNR calculation error: requested time at edge of ROQ time samples")
                return self._CalculatedSNRs(
                    d_inner_h=np.nan_to_num(-np.inf), optimal_snr_squared=0,
                    complex_matched_filter_snr=np.nan_to_num(-np.inf),
                    d_inner_h_squared_tc_array=None,
                    d_inner_h_array=None,
                    optimal_snr_squared_array=None)

            d_inner_h_tc_array = np.einsum(
                'i,ji->j', np.conjugate(h_linear),
                self.weights[interferometer.name + '_linear'][indices])

            d_inner_h = self._interp_five_samples(
                self.weights['time_samples'][indices], d_inner_h_tc_array, ifo_time)

            with np.errstate(invalid="ignore"):
                complex_matched_filter_snr = d_inner_h / (optimal_snr_squared**0.5)

            d_inner_h_array = None

        else:
            ifo_times = self._times - interferometer.strain_data.start_time + dt
            if self.jitter_time:
                ifo_times += self.parameters['time_jitter']
            d_inner_h_array = self._calculate_d_inner_h_array(ifo_times, h_linear, interferometer.name)

            d_inner_h = 0.
            complex_matched_filter_snr = 0.

        return self._CalculatedSNRs(
            d_inner_h=d_inner_h, optimal_snr_squared=optimal_snr_squared,
            complex_matched_filter_snr=complex_matched_filter_snr,
            d_inner_h_squared_tc_array=None,
            d_inner_h_array=d_inner_h_array,
            optimal_snr_squared_array=None)

    @staticmethod
    def _closest_time_indices(time, samples):
        """
        Get the closest five times

        Parameters
        ==========
        time: float
            Time to check
        samples: array-like
            Available times

        Returns
        =======
        indices: list
            Indices nearest to time
        in_bounds: bool
            Whether the indices are for valid times
        """
        closest = int((time - samples[0]) / (samples[1] - samples[0]))
        indices = [closest + ii for ii in [-2, -1, 0, 1, 2]]
        in_bounds = (indices[0] >= 0) & (indices[-1] < samples.size)
        return indices, in_bounds

    @staticmethod
    def _interp_five_samples(time_samples, values, time):
        """
        Interpolate a function of time with its values at the closest five times.
        The algorithm is explained in https://dcc.ligo.org/T2100224.

        Parameters
        ==========
        time_samples: array-like
            Closest 5 times
        values: array-like
            The values of the function at closest 5 times
        time: float
            Time at which the function is calculated

        Returns
        =======
        value: float
            The value of the function at the input time
        """
        r1 = (-values[0] + 8. * values[1] - 14. * values[2] + 8. * values[3] - values[4]) / 4.
        r2 = values[2] - 2. * values[3] + values[4]
        a = (time_samples[3] - time) / (time_samples[1] - time_samples[0])
        b = 1. - a
        c = (a**3. - a) / 6.
        d = (b**3. - b) / 6.
        return a * values[2] + b * values[3] + c * r1 + d * r2

    def _calculate_d_inner_h_array(self, times, h_linear, ifo_name):
        """
        Calculate d_inner_h at regularly-spaced time samples. Each value is
        interpolated from the nearest 5 samples with the algorithm explained in
        https://dcc.ligo.org/T2100224.

        Parameters
        ==========
        times: array-like
            Regularly-spaced time samples at which d_inner_h are calculated.
        h_linear: array-like
            Waveforms at linear frequency nodes
        ifo_name: str

        Returns
        =======
        d_inner_h_array: array-like
        """
        roq_time_space = self.weights['time_samples'][1] - self.weights['time_samples'][0]
        times_per_roq_time_space = (times - self.weights['time_samples'][0]) / roq_time_space
        closest_idxs = np.floor(times_per_roq_time_space).astype(int)
        # Get the nearest 5 samples of d_inner_h. Calculate only the required d_inner_h values if the time
        # spacing is larger than 5 times the ROQ time spacing.
        weights_linear = self.weights[ifo_name + '_linear']
        h_linear_conj = np.conjugate(h_linear)
        if (times[1] - times[0]) / roq_time_space > 5:
            d_inner_h_m2 = np.dot(weights_linear[closest_idxs - 2], h_linear_conj)
            d_inner_h_m1 = np.dot(weights_linear[closest_idxs - 1], h_linear_conj)
            d_inner_h_0 = np.dot(weights_linear[closest_idxs], h_linear_conj)
            d_inner_h_p1 = np.dot(weights_linear[closest_idxs + 1], h_linear_conj)
            d_inner_h_p2 = np.dot(weights_linear[closest_idxs + 2], h_linear_conj)
        else:
            d_inner_h_at_roq_time_samples = np.dot(weights_linear, h_linear_conj)
            d_inner_h_m2 = d_inner_h_at_roq_time_samples[closest_idxs - 2]
            d_inner_h_m1 = d_inner_h_at_roq_time_samples[closest_idxs - 1]
            d_inner_h_0 = d_inner_h_at_roq_time_samples[closest_idxs]
            d_inner_h_p1 = d_inner_h_at_roq_time_samples[closest_idxs + 1]
            d_inner_h_p2 = d_inner_h_at_roq_time_samples[closest_idxs + 2]
        # quantities required for spline interpolation
        b = times_per_roq_time_space - closest_idxs
        a = 1. - b
        c = (a**3. - a) / 6.
        d = (b**3. - b) / 6.
        r1 = (-d_inner_h_m2 + 8. * d_inner_h_m1 - 14. * d_inner_h_0 + 8. * d_inner_h_p1 - d_inner_h_p2) / 4.
        r2 = d_inner_h_0 - 2. * d_inner_h_p1 + d_inner_h_p2
        return a * d_inner_h_0 + b * d_inner_h_p1 + c * r1 + d * r2

    def perform_roq_params_check(self, ifo=None):
        """ Perform checking that the prior and data are valid for the ROQ

        Parameters
        ==========
        ifo: bilby.gw.detector.Interferometer
            The interferometer
        """
        if self.roq_params_check is False:
            logger.warning("No ROQ params checking performed")
            return
        else:
            if getattr(self, "roq_params_file", None) is not None:
                msg = ("Check ROQ params {} with roq_scale_factor={}"
                       .format(self.roq_params_file, self.roq_scale_factor))
            else:
                msg = ("Check ROQ params with roq_scale_factor={}"
                       .format(self.roq_scale_factor))
            logger.info(msg)

        roq_params = self.roq_params
        roq_minimum_frequency = roq_params['flow'] * self.roq_scale_factor
        roq_maximum_frequency = roq_params['fhigh'] * self.roq_scale_factor
        roq_segment_length = roq_params['seglen'] / self.roq_scale_factor
        roq_minimum_chirp_mass = roq_params['chirpmassmin'] / self.roq_scale_factor
        roq_maximum_chirp_mass = roq_params['chirpmassmax'] / self.roq_scale_factor
        roq_minimum_component_mass = roq_params['compmin'] / self.roq_scale_factor

        if ifo.maximum_frequency > roq_maximum_frequency:
            raise BilbyROQParamsRangeError(
                "Requested maximum frequency {} larger than ROQ basis fhigh {}"
                .format(ifo.maximum_frequency, roq_maximum_frequency)
            )
        if ifo.minimum_frequency < roq_minimum_frequency:
            raise BilbyROQParamsRangeError(
                "Requested minimum frequency {} lower than ROQ basis flow {}"
                .format(ifo.minimum_frequency, roq_minimum_frequency)
            )
        if ifo.strain_data.duration != roq_segment_length:
            raise BilbyROQParamsRangeError(
                "Requested duration differs from ROQ basis seglen")

        priors = self.priors
        if isinstance(priors, CBCPriorDict) is False:
            logger.warning("Unable to check ROQ parameter bounds: priors not understood")
            return

        if priors.minimum_chirp_mass is None:
            logger.warning("Unable to check minimum chirp mass ROQ bounds")
        elif priors.minimum_chirp_mass < roq_minimum_chirp_mass:
            raise BilbyROQParamsRangeError(
                "Prior minimum chirp mass {} less than ROQ basis bound {}"
                .format(priors.minimum_chirp_mass, roq_minimum_chirp_mass)
            )

        if priors.maximum_chirp_mass is None:
            logger.warning("Unable to check maximum_chirp mass ROQ bounds")
        elif priors.maximum_chirp_mass > roq_maximum_chirp_mass:
            raise BilbyROQParamsRangeError(
                "Prior maximum chirp mass {} greater than ROQ basis bound {}"
                .format(priors.maximum_chirp_mass, roq_maximum_chirp_mass)
            )

        if priors.minimum_component_mass is None:
            logger.warning("Unable to check minimum component mass ROQ bounds")
        elif priors.minimum_component_mass < roq_minimum_component_mass:
            raise BilbyROQParamsRangeError(
                "Prior minimum component mass {} less than ROQ basis bound {}"
                .format(priors.minimum_component_mass, roq_minimum_component_mass)
            )

    def _set_weights(self, linear_matrix, quadratic_matrix):
        """
        Setup the time-dependent ROQ weights.
        See https://dcc.ligo.org/LIGO-T2100125 for the detail of how to compute them.

        Parameters
        ==========
        linear_matrix, quadratic_matrix: array_like
            Arrays of the linear and quadratic basis

        """

        time_space = self._get_time_resolution()
        number_of_time_samples = int(self.interferometers.duration / time_space)
        try:
            import pyfftw
            ifft_input = pyfftw.empty_aligned(number_of_time_samples, dtype=complex)
            ifft_output = pyfftw.empty_aligned(number_of_time_samples, dtype=complex)
            ifft = pyfftw.FFTW(ifft_input, ifft_output, direction='FFTW_BACKWARD')
        except ImportError:
            pyfftw = None
            logger.warning("You do not have pyfftw installed, falling back to numpy.fft.")
            ifft_input = np.zeros(number_of_time_samples, dtype=complex)
            ifft = np.fft.ifft
        earth_light_crossing_time = 2 * radius_of_earth / speed_of_light + 5 * time_space
        start_idx = max(
            0,
            int(np.floor((
                self.priors['{}_time'.format(self.time_reference)].minimum
                - earth_light_crossing_time
                - self.interferometers.start_time
            ) / time_space))
        )
        end_idx = min(
            number_of_time_samples - 1,
            int(np.ceil((
                self.priors['{}_time'.format(self.time_reference)].maximum
                + earth_light_crossing_time
                - self.interferometers.start_time
            ) / time_space))
        )
        self.weights['time_samples'] = np.arange(start_idx, end_idx + 1) * time_space
        logger.info("Using {} ROQ time samples".format(len(self.weights['time_samples'])))

        for ifo in self.interferometers:
            if self.roq_params is not None:
                self.perform_roq_params_check(ifo)
                # Get scaled ROQ quantities
                roq_scaled_minimum_frequency = self.roq_params['flow'] * self.roq_scale_factor
                roq_scaled_maximum_frequency = self.roq_params['fhigh'] * self.roq_scale_factor
                roq_scaled_segment_length = self.roq_params['seglen'] / self.roq_scale_factor
                # Generate frequencies for the ROQ
                roq_frequencies = create_frequency_series(
                    sampling_frequency=roq_scaled_maximum_frequency * 2,
                    duration=roq_scaled_segment_length)
                roq_mask = roq_frequencies >= roq_scaled_minimum_frequency
                roq_frequencies = roq_frequencies[roq_mask]
                overlap_frequencies, ifo_idxs, roq_idxs = np.intersect1d(
                    ifo.frequency_array[ifo.frequency_mask], roq_frequencies,
                    return_indices=True)
            else:
                overlap_frequencies = ifo.frequency_array[ifo.frequency_mask]
                roq_idxs = np.arange(linear_matrix.shape[0], dtype=int)
                ifo_idxs = np.arange(sum(ifo.frequency_mask))
                if len(ifo_idxs) != len(roq_idxs):
                    raise ValueError(
                        "Mismatch between ROQ basis and frequency array for "
                        "{}".format(ifo.name))
            logger.info(
                "Building ROQ weights for {} with {} frequencies between {} "
                "and {}.".format(
                    ifo.name, len(overlap_frequencies),
                    min(overlap_frequencies), max(overlap_frequencies)))

            ifft_input[:] *= 0.
            self.weights[ifo.name + '_linear'] = \
                np.zeros((len(self.weights['time_samples']), linear_matrix.shape[1]), dtype=complex)
            data_over_psd = (
                ifo.frequency_domain_strain[ifo.frequency_mask][ifo_idxs]
                / ifo.power_spectral_density_array[ifo.frequency_mask][ifo_idxs]
            )
            nonzero_idxs = ifo_idxs + int(ifo.frequency_array[ifo.frequency_mask][0] * self.interferometers.duration)
            for i, basis_element in enumerate(linear_matrix[roq_idxs].T):
                ifft_input[nonzero_idxs] = data_over_psd * np.conj(basis_element)
                self.weights[ifo.name + '_linear'][:, i] = ifft(ifft_input)[start_idx:end_idx + 1]
            self.weights[ifo.name + '_linear'] *= 4. * number_of_time_samples / self.interferometers.duration

            self.weights[ifo.name + '_quadratic'] = build_roq_weights(
                1 /
                ifo.power_spectral_density_array[ifo.frequency_mask][ifo_idxs],
                quadratic_matrix[roq_idxs].real,
                1 / ifo.strain_data.duration)

            logger.info("Finished building weights for {}".format(ifo.name))

        if pyfftw is not None:
            pyfftw.forget_wisdom()

    def save_weights(self, filename, format='npz'):
        if format not in filename:
            filename += "." + format
        logger.info("Saving ROQ weights to {}".format(filename))
        if format == 'json':
            with open(filename, 'w') as file:
                json.dump(self.weights, file, indent=2, cls=BilbyJsonEncoder)
        elif format == 'npz':
            np.savez(filename, **self.weights)

    @staticmethod
    def load_weights(filename, format=None):
        if format is None:
            format = filename.split(".")[-1]
        if format not in ["json", "npz"]:
            raise IOError("Format {} not recognized.".format(format))
        logger.info("Loading ROQ weights from {}".format(filename))
        if format == "json":
            with open(filename, 'r') as file:
                weights = json.load(file, object_hook=decode_bilby_json)
        elif format == "npz":
            # Wrap in dict to load data into memory
            weights = dict(np.load(filename))
        return weights

    def _get_time_resolution(self):
        """
        This method estimates the time resolution given the optimal SNR of the
        signal in the detector. This is then used when constructing the weights
        for the ROQ.

        A minimum resolution is set by assuming the SNR in each detector is at
        least 10. When the SNR is not available the SNR is assumed to be 30 in
        each detector.

        Returns
        =======
        delta_t: float
            Time resolution
        """

        def calc_fhigh(freq, psd, scaling=20.):
            """

            Parameters
            ==========
            freq: array-like
                Frequency array
            psd: array-like
                Power spectral density
            scaling: float
                SNR dependent scaling factor

            Returns
            =======
            f_high: float
                The maximum frequency which must be considered
            """
            from scipy.integrate import simps
            integrand1 = np.power(freq, -7. / 3) / psd
            integral1 = simps(integrand1, freq)
            integrand3 = np.power(freq, 2. / 3.) / (psd * integral1)
            f_3_bar = simps(integrand3, freq)

            f_high = scaling * f_3_bar**(1 / 3)

            return f_high

        def c_f_scaling(snr):
            return (np.pi**2 * snr**2 / 6)**(1 / 3)

        inj_snr_sq = 0
        for ifo in self.interferometers:
            inj_snr_sq += max(10, ifo.meta_data.get('optimal_SNR', 30))**2

        psd = ifo.power_spectral_density_array[ifo.frequency_mask]
        freq = ifo.frequency_array[ifo.frequency_mask]
        fhigh = calc_fhigh(freq, psd, scaling=c_f_scaling(inj_snr_sq**0.5))

        delta_t = fhigh**-1

        # Apply a safety factor to ensure the time step is short enough
        delta_t = delta_t / 5

        # duration / delta_t needs to be a power of 2 for IFFT
        number_of_time_samples = max(
            self.interferometers.duration / delta_t,
            self.interferometers.frequency_array[-1] * self.interferometers.duration + 1)
        number_of_time_samples = int(2**np.ceil(np.log2(number_of_time_samples)))
        delta_t = self.interferometers.duration / number_of_time_samples
        logger.info("ROQ time-step = {}".format(delta_t))
        return delta_t

    def _rescale_signal(self, signal, new_distance):
        for kind in ['linear', 'quadratic']:
            for mode in signal[kind]:
                signal[kind][mode] *= self._ref_dist / new_distance


class BilbyROQParamsRangeError(Exception):
    pass
