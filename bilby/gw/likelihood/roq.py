
import json

import numpy as np

from .base import GravitationalWaveTransient
from ...core.utils import BilbyJsonEncoder, decode_bilby_json
from ...core.utils import (
    logger, create_frequency_series, speed_of_light, radius_of_earth
)
from ..prior import CBCPriorDict
from ..utils import ln_i0


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
    parameter_conversion: func, optional
        Function to update self.parameters before bases are selected based on
        the values of self.parameters. This enables a user to switch bases
        based on the values of parameters which are not directly used for
        sampling.
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
            reference_frame="sky", time_reference="geocenter",
            parameter_conversion=None

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
            is_hdf5_linear = isinstance(linear_matrix, str) and linear_matrix.endswith('.hdf5')
            linear_matrix = self._parse_basis(linear_matrix, 'linear')
            is_hdf5_quadratic = isinstance(quadratic_matrix, str) and quadratic_matrix.endswith('.hdf5')
            quadratic_matrix = self._parse_basis(quadratic_matrix, 'quadratic')
            # retrieve roq params from a basis file if it is .hdf5
            if self.roq_params is None:
                if is_hdf5_linear:
                    self.roq_params = np.array(
                        [(linear_matrix['minimum_frequency_hz'][()],
                          linear_matrix['maximum_frequency_hz'][()],
                          linear_matrix['duration_s'][()])],
                        dtype=[('flow', float), ('fhigh', float), ('seglen', float)]
                    )
                if is_hdf5_quadratic:
                    if self.roq_params is None:
                        self.roq_params = np.array(
                            [(quadratic_matrix['minimum_frequency_hz'][()],
                              quadratic_matrix['maximum_frequency_hz'][()],
                              quadratic_matrix['duration_s'][()])],
                            dtype=[('flow', float), ('fhigh', float), ('seglen', float)]
                        )
                    else:
                        self.roq_params['flow'] = max(
                            self.roq_params['flow'], quadratic_matrix['minimum_frequency_hz'][()]
                        )
                        self.roq_params['fhigh'] = min(
                            self.roq_params['fhigh'], quadratic_matrix['maximum_frequency_hz'][()]
                        )
                        self.roq_params['seglen'] = min(
                            self.roq_params['seglen'], quadratic_matrix['duration_s'][()]
                        )
            if self.roq_params is not None:
                for ifo in self.interferometers:
                    self.perform_roq_params_check(ifo)

            self.weights = dict()
            self._set_weights(linear_matrix=linear_matrix, quadratic_matrix=quadratic_matrix)
            if is_hdf5_linear:
                linear_matrix.close()
            if is_hdf5_quadratic:
                quadratic_matrix.close()

        self.number_of_bases_linear = len(self.weights[f'{self.interferometers[0].name}_linear'])
        self.number_of_bases_quadratic = len(self.weights[f'{self.interferometers[0].name}_quadratic'])
        self._cache = dict(parameters=None, basis_number_linear=None, basis_number_quadratic=None)
        self.parameter_conversion = parameter_conversion

        for basis_type in ['linear', 'quadratic']:
            number_of_bases = getattr(self, f'number_of_bases_{basis_type}')
            if number_of_bases > 1:
                self._verify_numbers_of_prior_ranges_and_frequency_nodes(basis_type)
            else:
                self._check_frequency_nodes_exist_for_single_basis(basis_type)
            self._verify_prior_ranges(basis_type)

        self._set_unique_frequency_nodes_and_inverse()
        # need to fill waveform_arguments here if single basis is used, as they will never be updated.
        if self.number_of_bases_linear == 1 and self.number_of_bases_quadratic == 1:
            frequency_nodes, linear_indices, quadratic_indices = \
                self._unique_frequency_nodes_and_inverse[0][0]
            self._waveform_generator.waveform_arguments['frequency_nodes'] = frequency_nodes
            self._waveform_generator.waveform_arguments['linear_indices'] = linear_indices
            self._waveform_generator.waveform_arguments['quadratic_indices'] = quadratic_indices

    def _verify_numbers_of_prior_ranges_and_frequency_nodes(self, basis_type):
        """
        Check if self.weights contains lists of prior ranges and frequency nodes, and their sizes are equal to the
        number of bases.

        Parameters
        ==========
        basis_type: str

        """
        number_of_bases = getattr(self, f'number_of_bases_{basis_type}')
        key = f'prior_range_{basis_type}'
        try:
            prior_ranges = self.weights[key]
        except KeyError:
            raise AttributeError(
                f'For the use of multiple {basis_type} ROQ bases, weights should contain "{key}".')
        else:
            for param_name in prior_ranges:
                if len(prior_ranges[param_name]) != number_of_bases:
                    raise ValueError(
                        f'The number of prior ranges for "{param_name}" does not '
                        f'match the number of {basis_type} bases')
        key = f'frequency_nodes_{basis_type}'
        try:
            frequency_nodes = self.weights[key]
        except KeyError:
            raise AttributeError(
                f'For the use of multiple {basis_type} ROQ bases, weights should contain "{key}".')
        else:
            if len(frequency_nodes) != number_of_bases:
                raise ValueError(
                    f'The number of arrays of frequency nodes does not match the number of {basis_type} bases')

    def _verify_prior_ranges(self, basis_type):
        """Check if the union of prior ranges is within the ROQ basis bounds.

        Parameters
        ==========
        basis_type: str

        """
        key = f'prior_range_{basis_type}'
        if key not in self.weights:
            return
        prior_ranges = self.weights[key]
        for param_name, prior_ranges_of_this_param in prior_ranges.items():
            prior_minimum = self.priors[param_name].minimum
            basis_minimum = np.min(prior_ranges_of_this_param[:, 0])
            if prior_minimum < basis_minimum:
                raise BilbyROQParamsRangeError(
                    f"Prior minimum of {param_name} {prior_minimum} less "
                    f"than ROQ basis bound {basis_minimum}"
                )

            prior_maximum = self.priors[param_name].maximum
            basis_maximum = np.max(prior_ranges_of_this_param[:, 1])
            if prior_maximum > basis_maximum:
                raise BilbyROQParamsRangeError(
                    f"Prior maximum of {param_name} {prior_maximum} greater "
                    f"than ROQ basis bound {basis_maximum}"
                )

    def _check_frequency_nodes_exist_for_single_basis(self, basis_type):
        """
        For a single-basis case, frequency nodes should be contained in self._waveform_generator.waveform_arguments or
        self.weights. This method checks if it is the case and raise AttributeError if not. This method also adds
        frequency nodes to self._waveform_generator.waveform_arguments or self.weights from the other.

        Parameters
        ==========
        basis_type: str

        """
        key = f'frequency_nodes_{basis_type}'
        if not (key in self.weights or key in self._waveform_generator.waveform_arguments):
            raise AttributeError(f'{key} should be contained in weights or waveform arguments.')
        elif key not in self._waveform_generator.waveform_arguments:
            self._waveform_generator.waveform_arguments[key] = self.weights[key][0]
        elif key not in self.weights:
            self.weights[key] = [self._waveform_generator.waveform_arguments[key]]

    def _set_unique_frequency_nodes_and_inverse(self):
        """Set unique frequency nodes and indices to recover linear and quadratic frequency nodes for each combination
        of linear and quadratic bases
        """
        self._unique_frequency_nodes_and_inverse = []
        for idx_linear in range(self.number_of_bases_linear):
            tmp = []
            frequency_nodes_linear = self.weights['frequency_nodes_linear'][idx_linear]
            size_linear = len(frequency_nodes_linear)
            for idx_quadratic in range(self.number_of_bases_quadratic):
                frequency_nodes_quadratic = self.weights['frequency_nodes_quadratic'][idx_quadratic]
                frequency_nodes_unique, original_indices = np.unique(
                    np.hstack((frequency_nodes_linear, frequency_nodes_quadratic)),
                    return_inverse=True
                )
                linear_indices = original_indices[:size_linear]
                quadratic_indices = original_indices[size_linear:]
                tmp.append(
                    (frequency_nodes_unique, linear_indices, quadratic_indices)
                )
            self._unique_frequency_nodes_and_inverse.append(tmp)

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

    @staticmethod
    def _parse_basis(basis, basis_type):
        """
        Parse basis and format it to an hdf5-like object

        Parameters
        ----------
        basis : array-like or str
            array-like basis or path to file
        basis_type : str
            'linear' or 'quadratic'

        Returns
        -------
        basis : hdf5-like object

        """
        if basis_type not in ['linear', 'quadratic']:
            raise ValueError(f'basis_type {basis_type} not recognized')
        if isinstance(basis, str):
            logger.info(f'Loading {basis_type}_matrix from {basis}')
            format = basis.split('.')[-1]
            if format == 'npy':
                basis = {f'basis_{basis_type}': {'0': {'basis': np.load(basis)}}}
            elif format == 'hdf5':
                import h5py
                basis = h5py.File(basis, 'r')
            else:
                raise IOError(f'Format {format} not recognized.')
        elif isinstance(basis, np.ndarray):
            basis = {f'basis_{basis_type}': {'0': {'basis': basis.T}}}
        else:
            raise TypeError('basis needs to be str or np.ndarray')
        return basis

    def _select_prior_ranges(self, prior_ranges):
        """
        Select prior ranges which have intersection with self.priors

        Parameters
        ----------
        prior_ranges : dict
            dictionary whose keys are parameter names and values are ndarray of
            their prior ranges

        Returns
        -------
        idxs_in_prior_range : ndarray
            indexes of selected prior ranges
        selected_prior_ranges : dict

        """
        param_names = list(prior_ranges.keys())
        number_of_prior_ranges = len(prior_ranges[param_names[0]])
        in_prior_range = np.ones(number_of_prior_ranges, dtype=bool)
        for param_name in param_names:
            try:
                prior = self.priors[param_name]
            except KeyError:
                continue
            prior_ranges_of_this_param = prior_ranges[param_name]
            in_prior_range *= \
                (prior_ranges_of_this_param[:, 1] >= prior.minimum) * \
                (prior_ranges_of_this_param[:, 0] <= prior.maximum)
        idxs_in_prior_range = np.arange(number_of_prior_ranges)[in_prior_range]
        return idxs_in_prior_range, \
            dict((param_name, prior_ranges[param_name][idxs_in_prior_range])
                 for param_name in param_names)

    def _update_basis(self):
        """
        Update basis and frequency nodes depending on the curret values of parameters

        This updates
        - self._cache
        - frequency_nodes_linear/quadratic in self._waveform_generator.waveform_arguments

        """
        parameters = self.parameters.copy()
        if self.parameter_conversion is not None:
            parameters = self.parameter_conversion(parameters)
        for basis_type, number_of_bases in zip(
            ['linear', 'quadratic'], [self.number_of_bases_linear, self.number_of_bases_quadratic]
        ):
            basis_number_key = f'basis_number_{basis_type}'
            if number_of_bases == 1:
                self._cache[basis_number_key] = 0
                continue
            in_prior_range = np.ones(number_of_bases, dtype=bool)
            prior_range_key = f'prior_range_{basis_type}'
            for param_name in self.weights[prior_range_key]:
                if param_name not in parameters:
                    continue
                in_prior_range *= \
                    (self.weights[prior_range_key][param_name][:, 0] <= parameters[param_name]) * \
                    (self.weights[prior_range_key][param_name][:, 1] >= parameters[param_name])
            self._cache[basis_number_key] = np.arange(number_of_bases)[in_prior_range][0]
        basis_number_linear = self._cache['basis_number_linear']
        basis_number_quadratic = self._cache['basis_number_quadratic']
        frequency_nodes, linear_indices, quadratic_indices = \
            self._unique_frequency_nodes_and_inverse[basis_number_linear][basis_number_quadratic]
        self._waveform_generator.waveform_arguments['frequency_nodes'] = frequency_nodes
        self._waveform_generator.waveform_arguments['linear_indices'] = linear_indices
        self._waveform_generator.waveform_arguments['quadratic_indices'] = quadratic_indices
        self._cache['parameters'] = self.parameters.copy()

    @property
    def basis_number_linear(self):
        if self.number_of_bases_linear > 1 or self.number_of_bases_quadratic > 1:
            if self.parameters != self._cache['parameters']:
                self._update_basis()
            return self._cache['basis_number_linear']
        else:
            return 0

    @property
    def basis_number_quadratic(self):
        if self.number_of_bases_linear > 1 or self.number_of_bases_quadratic > 1:
            if self.parameters != self._cache['parameters']:
                self._update_basis()
            return self._cache['basis_number_quadratic']
        else:
            return 0

    @property
    def waveform_generator(self):
        if getattr(self, 'number_of_bases_linear', 1) > 1 or getattr(self, 'number_of_bases_quadratic', 1) > 1:
            if self.parameters != self._cache['parameters']:
                self._update_basis()
        return self._waveform_generator

    @waveform_generator.setter
    def waveform_generator(self, waveform_generator):
        self._waveform_generator = waveform_generator

    def calculate_snrs(self, waveform_polarizations, interferometer, return_array=True):
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

        frequency_nodes = self.waveform_generator.waveform_arguments['frequency_nodes']
        linear_indices = self.waveform_generator.waveform_arguments['linear_indices']
        quadratic_indices = self.waveform_generator.waveform_arguments['quadratic_indices']
        size_linear = len(linear_indices)
        size_quadratic = len(quadratic_indices)
        h_linear = np.zeros(size_linear, dtype=complex)
        h_quadratic = np.zeros(size_quadratic, dtype=complex)
        for mode in waveform_polarizations['linear']:
            response = interferometer.antenna_response(
                self.parameters['ra'], self.parameters['dec'],
                time_ref,
                self.parameters['psi'],
                mode
            )
            h_linear += waveform_polarizations['linear'][mode] * response
            h_quadratic += waveform_polarizations['quadratic'][mode] * response

        calib_factor = interferometer.calibration_model.get_calibration_factor(
            frequency_nodes, prefix='recalib_{}_'.format(interferometer.name), **self.parameters)
        h_linear *= calib_factor[linear_indices]
        h_quadratic *= calib_factor[quadratic_indices]

        optimal_snr_squared = np.vdot(
            np.abs(h_quadratic)**2,
            self.weights[interferometer.name + '_quadratic'][self.basis_number_quadratic]
        )

        dt = interferometer.time_delay_from_geocenter(
            self.parameters['ra'], self.parameters['dec'], time_ref)
        dt_geocent = self.parameters['geocent_time'] - interferometer.strain_data.start_time
        ifo_time = dt_geocent + dt

        indices, in_bounds = self._closest_time_indices(
            ifo_time, self.weights['time_samples'])
        if not in_bounds:
            logger.debug("SNR calculation error: requested time at edge of ROQ time samples")
            d_inner_h = -np.inf
            complex_matched_filter_snr = -np.inf
        else:
            d_inner_h_tc_array = np.einsum(
                'i,ji->j', np.conjugate(h_linear),
                self.weights[interferometer.name + '_linear'][self.basis_number_linear][indices])

            d_inner_h = self._interp_five_samples(
                self.weights['time_samples'][indices], d_inner_h_tc_array, ifo_time)

            with np.errstate(invalid="ignore"):
                complex_matched_filter_snr = d_inner_h / (optimal_snr_squared**0.5)

        if return_array and self.time_marginalization:
            ifo_times = self._times - interferometer.strain_data.start_time
            ifo_times += dt
            if self.jitter_time:
                ifo_times += self.parameters['time_jitter']
            d_inner_h_array = self._calculate_d_inner_h_array(ifo_times, h_linear, interferometer.name)
        else:
            d_inner_h_array = None

        return self._CalculatedSNRs(
            d_inner_h=d_inner_h,
            optimal_snr_squared=optimal_snr_squared.real,
            complex_matched_filter_snr=complex_matched_filter_snr,
            d_inner_h_array=d_inner_h_array,
        )

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
        weights_linear = self.weights[ifo_name + '_linear'][self.basis_number_linear]
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
        try:
            roq_minimum_chirp_mass = roq_params['chirpmassmin'] / self.roq_scale_factor
        except ValueError:
            roq_minimum_chirp_mass = None
        try:
            roq_maximum_chirp_mass = roq_params['chirpmassmax'] / self.roq_scale_factor
        except ValueError:
            roq_maximum_chirp_mass = None
        try:
            roq_minimum_component_mass = roq_params['compmin'] / self.roq_scale_factor
        except ValueError:
            roq_minimum_component_mass = None

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

        if roq_minimum_chirp_mass is not None:
            if priors.minimum_chirp_mass is None:
                logger.warning("Unable to check minimum chirp mass ROQ bounds")
            elif priors.minimum_chirp_mass < roq_minimum_chirp_mass:
                raise BilbyROQParamsRangeError(
                    "Prior minimum chirp mass {} less than ROQ basis bound {}"
                    .format(priors.minimum_chirp_mass, roq_minimum_chirp_mass)
                )

        if roq_maximum_chirp_mass is not None:
            if priors.maximum_chirp_mass is None:
                logger.warning("Unable to check maximum_chirp mass ROQ bounds")
            elif priors.maximum_chirp_mass > roq_maximum_chirp_mass:
                raise BilbyROQParamsRangeError(
                    "Prior maximum chirp mass {} greater than ROQ basis bound {}"
                    .format(priors.maximum_chirp_mass, roq_maximum_chirp_mass)
                )

        if roq_minimum_component_mass is not None:
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

        Parameters
        ==========
        linear_matrix, quadratic_matrix: dictionary or h5py.File
            linear and quadratic basis

        """
        time_space = self._get_time_resolution()
        number_of_time_samples = int(self.interferometers.duration / time_space)
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

        # select bases to be used, set prior ranges and frequency nodes if exist
        idxs_in_prior_range = dict()
        for basis_type, matrix in zip(['linear', 'quadratic'], [linear_matrix, quadratic_matrix]):
            key = f'prior_range_{basis_type}'
            if key in matrix:
                prior_ranges = {}
                for param_name in matrix[key]:
                    if 'roq_scale_power' in matrix[key][param_name].attrs:
                        roq_scale_factor = self.roq_scale_factor**matrix[key][param_name].attrs['roq_scale_power']
                    else:
                        roq_scale_factor = 1.
                    prior_ranges[param_name] = matrix[key][param_name][()] * roq_scale_factor
                selected_idxs, selected_prior_ranges = self._select_prior_ranges(prior_ranges)
                if len(selected_idxs) == 0:
                    raise BilbyROQParamsRangeError(f"There are no {basis_type} ROQ bases within the prior range.")
                self.weights[key] = selected_prior_ranges
                idxs_in_prior_range[basis_type] = selected_idxs
            else:
                idxs_in_prior_range[basis_type] = [0]
            if 'frequency_nodes' in matrix[f'basis_{basis_type}'][str(idxs_in_prior_range[basis_type][0])]:
                self.weights[f'frequency_nodes_{basis_type}'] = [
                    matrix[f'basis_{basis_type}'][str(i)]['frequency_nodes'][()] * self.roq_scale_factor
                    for i in idxs_in_prior_range[basis_type]]

        if 'multiband_linear' in linear_matrix:
            multiband_linear = linear_matrix['multiband_linear'][()]
        else:
            multiband_linear = False
        if 'multiband_quadratic' in quadratic_matrix:
            multiband_quadratic = quadratic_matrix['multiband_quadratic'][()]
        else:
            multiband_quadratic = False

        # Get intersection between ifo and ROQ frequency samples. Required only for non-multibanded basis.
        if not (multiband_linear and multiband_quadratic):
            roq_idxs = {}
            ifo_idxs = {}
            for ifo in self.interferometers:
                if self.roq_params is not None:
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
                    overlap_frequencies, ifo_idxs_this_ifo, roq_idxs_this_ifo = np.intersect1d(
                        ifo.frequency_array[ifo.frequency_mask], roq_frequencies,
                        return_indices=True)
                else:
                    overlap_frequencies = ifo.frequency_array[ifo.frequency_mask]
                    roq_idxs_this_ifo = np.arange(
                        linear_matrix['basis_linear'][str(idxs_in_prior_range['linear'][0])]['basis'].shape[1],
                        dtype=int)
                    ifo_idxs_this_ifo = np.arange(sum(ifo.frequency_mask))
                    if len(ifo_idxs_this_ifo) != len(roq_idxs_this_ifo):
                        raise ValueError(
                            "Mismatch between ROQ basis and frequency array for "
                            "{}".format(ifo.name))
                logger.info(
                    "Building ROQ weights for {} with {} frequencies between {} "
                    "and {}.".format(
                        ifo.name, len(overlap_frequencies),
                        min(overlap_frequencies), max(overlap_frequencies)))
                roq_idxs[ifo.name] = roq_idxs_this_ifo
                ifo_idxs[ifo.name] = ifo_idxs_this_ifo

        if multiband_linear:
            self._set_weights_linear_multiband(linear_matrix, idxs_in_prior_range['linear'])
        else:
            self._set_weights_linear(linear_matrix, idxs_in_prior_range['linear'], roq_idxs, ifo_idxs)

        if multiband_quadratic:
            self._set_weights_quadratic_multiband(quadratic_matrix, idxs_in_prior_range['quadratic'])
        else:
            self._set_weights_quadratic(quadratic_matrix, idxs_in_prior_range['quadratic'], roq_idxs, ifo_idxs)

    def _set_weights_linear(self, linear_matrix, basis_idxs, roq_idxs, ifo_idxs):
        """
        Setup the time-dependent linear ROQ weights. See https://dcc.ligo.org/LIGO-T2100125 for the detail of how to
        compute them.

        Parameters
        ==========
        linear_matrix : dictionary or h5py.File
            linear basis
        basis_idxs : array-like
            indexes of bases used for a run
        roq_idxs : dictionary
            dictionary whose keys are interferometer names and values are indexes of basis components intersecting
            frequency-domain data
        ifo_idxs : dictionary
            dictionary whose keys are interferometer names and values are indexes of frequency-domain data intersecting
            basis components

        """
        for ifo in self.interferometers:
            self.weights[ifo.name + '_linear'] = []
        time_space = self.weights['time_samples'][1] - self.weights['time_samples'][0]
        number_of_time_samples = int(self.interferometers.duration / time_space)
        start_idx = int(self.weights['time_samples'][0] / time_space)
        end_idx = int(self.weights['time_samples'][-1] / time_space)
        nonzero_idxs = {}
        data_over_psd = {}
        for ifo in self.interferometers:
            nonzero_idxs[ifo.name] = ifo_idxs[ifo.name] + int(
                ifo.frequency_array[ifo.frequency_mask][0] * self.interferometers.duration)
            data_over_psd[ifo.name] = ifo.frequency_domain_strain[ifo.frequency_mask][ifo_idxs[ifo.name]] / \
                ifo.power_spectral_density_array[ifo.frequency_mask][ifo_idxs[ifo.name]]
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
        for basis_idx in basis_idxs:
            logger.info(f"Building linear ROQ weights for the {basis_idx}-th basis.")
            linear_matrix_single = linear_matrix['basis_linear'][str(basis_idx)]['basis']
            basis_size = linear_matrix_single.shape[0]
            for ifo in self.interferometers:
                ifft_input[:] *= 0.
                linear_weights = \
                    np.zeros((len(self.weights['time_samples']), basis_size), dtype=complex)
                for i in range(basis_size):
                    basis_element = linear_matrix_single[i][roq_idxs[ifo.name]]
                    ifft_input[nonzero_idxs[ifo.name]] = data_over_psd[ifo.name] * np.conj(basis_element)
                    linear_weights[:, i] = ifft(ifft_input)[start_idx:end_idx + 1]
                linear_weights *= 4. * number_of_time_samples / self.interferometers.duration
                self.weights[ifo.name + '_linear'].append(linear_weights)
        if pyfftw is not None:
            pyfftw.forget_wisdom()

    def _set_weights_linear_multiband(self, linear_matrix, basis_idxs):
        """
        Setup the time-dependent linear ROQ weights from multibanded basis

        Parameters
        ==========
        linear_matrix : dictionary or h5py.File
            linear basis
        basis_idxs : array-like
            indexes of bases used for a run

        """
        for ifo in self.interferometers:
            self.weights[ifo.name + '_linear'] = []
        Tbs = linear_matrix['durations_s_linear'][()] / self.roq_scale_factor
        start_end_frequency_bins = linear_matrix['start_end_frequency_bins_linear'][()]
        basis_dimension = np.sum(start_end_frequency_bins[:, 1] - start_end_frequency_bins[:, 0] + 1)
        fhigh_basis = np.max(start_end_frequency_bins[:, 1] / Tbs)
        # prepare time-shifted data, which is multiplied by basis
        tc_shifted_data = dict()
        for ifo in self.interferometers:
            over_whitened_frequency_data = np.zeros(int(fhigh_basis * ifo.duration) + 1, dtype=complex)
            over_whitened_frequency_data[np.arange(len(ifo.frequency_domain_strain))[ifo.frequency_mask]] = \
                ifo.frequency_domain_strain[ifo.frequency_mask] / ifo.power_spectral_density_array[ifo.frequency_mask]
            over_whitened_time_data = np.fft.irfft(over_whitened_frequency_data)
            tc_shifted_data[ifo.name] = np.zeros((basis_dimension, len(self.weights['time_samples'])), dtype=complex)
            start_idx_of_band = 0
            for b, Tb in enumerate(Tbs):
                start_frequency_bin, end_frequency_bin = start_end_frequency_bins[b]
                fs = np.arange(start_frequency_bin, end_frequency_bin + 1) / Tb
                Db = np.fft.rfft(
                    over_whitened_time_data[-int(2. * fhigh_basis * Tb):]
                )[start_frequency_bin:end_frequency_bin + 1]
                start_idx_of_next_band = start_idx_of_band + end_frequency_bin - start_frequency_bin + 1
                tc_shifted_data[ifo.name][start_idx_of_band:start_idx_of_next_band] = 4. / Tb * Db[:, None] * np.exp(
                    2. * np.pi * 1j * fs[:, None] * (self.weights['time_samples'][None, :] - ifo.duration + Tb))
                start_idx_of_band = start_idx_of_next_band
        # compute inner products
        for basis_idx in basis_idxs:
            logger.info(f"Building linear ROQ weights for the {basis_idx}-th basis.")
            linear_matrix_single = linear_matrix['basis_linear'][str(basis_idx)]['basis'][()]
            for ifo in self.interferometers:
                self.weights[ifo.name + '_linear'].append(
                    np.dot(np.conj(linear_matrix_single), tc_shifted_data[ifo.name]).T)

    def _set_weights_quadratic(self, quadratic_matrix, basis_idxs, roq_idxs, ifo_idxs):
        """
        Setup the quadratic ROQ weights

        Parameters
        ==========
        quadratic_matrix : dictionary or h5py.File
            quadratic basis
        basis_idxs : array-like
            indexes of bases used for a run
        roq_idxs : dictionary
            dictionary whose keys are interferometer names and values are indexes of basis components intersecting
            frequency-domain data
        ifo_idxs : dictionary
            dictionary whose keys are interferometer names and values are indexes of frequency-domain data intersecting
            basis components

        """
        for ifo in self.interferometers:
            self.weights[ifo.name + '_quadratic'] = []
        for basis_idx in basis_idxs:
            logger.info(f"Building quadratic ROQ weights for the {basis_idx}-th basis.")
            quadratic_matrix_single = quadratic_matrix['basis_quadratic'][str(basis_idx)]['basis'][()].real
            for ifo in self.interferometers:
                self.weights[ifo.name + '_quadratic'].append(
                    4. / ifo.strain_data.duration * np.dot(
                        quadratic_matrix_single[:, roq_idxs[ifo.name]],
                        1 / ifo.power_spectral_density_array[ifo.frequency_mask][ifo_idxs[ifo.name]]))
            del quadratic_matrix_single

    def _set_weights_quadratic_multiband(self, quadratic_matrix, basis_idxs):
        """
        Setup the quadratic ROQ weights from multibanded basis

        Parameters
        ==========
        quadratic_matrix : dictionary or h5py.File
            quadratic basis
        basis_idxs : array-like
            indexes of bases used for a run

        """
        for ifo in self.interferometers:
            self.weights[ifo.name + '_quadratic'] = []
        Tbs = quadratic_matrix['durations_s_quadratic'][()] / self.roq_scale_factor
        start_end_frequency_bins = quadratic_matrix['start_end_frequency_bins_quadratic'][()]
        basis_dimension = np.sum(start_end_frequency_bins[:, 1] - start_end_frequency_bins[:, 0] + 1)
        fhigh_basis = np.max(start_end_frequency_bins[:, 1] / Tbs)
        # prepare coefficients multiplied by basis
        multibanded_inverse_psd = dict()
        for ifo in self.interferometers:
            inverse_psd_frequency = np.zeros(int(fhigh_basis * ifo.duration) + 1)
            inverse_psd_frequency[np.arange(len(ifo.power_spectral_density_array))[ifo.frequency_mask]] = \
                1. / ifo.power_spectral_density_array[ifo.frequency_mask]
            inverse_psd_time = np.fft.irfft(inverse_psd_frequency)
            multibanded_inverse_psd[ifo.name] = np.zeros(basis_dimension)
            start_idx_of_band = 0
            for b, Tb in enumerate(Tbs):
                start_frequency_bin, end_frequency_bin = start_end_frequency_bins[b]
                number_of_samples_half = int(fhigh_basis * Tb)
                start_idx_of_next_band = start_idx_of_band + end_frequency_bin - start_frequency_bin + 1
                multibanded_inverse_psd[ifo.name][start_idx_of_band:start_idx_of_next_band] = 4. / Tb * np.fft.rfft(
                    np.append(inverse_psd_time[:number_of_samples_half], inverse_psd_time[-number_of_samples_half:])
                )[start_frequency_bin:end_frequency_bin + 1].real
                start_idx_of_band = start_idx_of_next_band
        # compute inner products
        for basis_idx in basis_idxs:
            logger.info(f"Building quadratic ROQ weights for the {basis_idx}-th basis.")
            quadratic_matrix_single = quadratic_matrix['basis_quadratic'][str(basis_idx)]['basis'][()].real
            for ifo in self.interferometers:
                self.weights[ifo.name + '_quadratic'].append(
                    np.dot(quadratic_matrix_single, multibanded_inverse_psd[ifo.name]))

    def save_weights(self, filename, format='hdf5'):
        """
        Save ROQ weights into a single file. format should be npz, or hdf5.
        For weights from multiple bases, hdf5 is only the possible option.
        Support for json format is deprecated as of :code:`v2.1` and will be
        removed in :code:`v2.2`, another method should be used by default.

        Parameters
        ==========
        filename : str
            The name of the file to save the weights to.
        format : str
            The format to save the data to, this should be one of
            :code:`"hdf5"`, :code:`"npz"`, default=:code:`"hdf5"`.
        """
        if format not in ['json', 'npz', 'hdf5']:
            raise IOError(f"Format {format} not recognized.")
        if format == "json":
            import warnings

            warnings.warn(
                "json format for ROQ weights is deprecated, use hdf5 instead.",
                DeprecationWarning
            )
        if format not in filename:
            filename += "." + format
        logger.info(f"Saving ROQ weights to {filename}")
        if format == 'json' or format == 'npz':
            if self.number_of_bases_linear > 1 or self.number_of_bases_quadratic > 1:
                raise ValueError(f'Format {format} not compatible with multiple bases')
            weights = dict()
            weights['time_samples'] = self.weights['time_samples']
            for basis_type in ['linear', 'quadratic']:
                for ifo in self.interferometers:
                    key = f'{ifo.name}_{basis_type}'
                    weights[key] = self.weights[key][0]
            if format == 'json':
                with open(filename, 'w') as file:
                    json.dump(weights, file, indent=2, cls=BilbyJsonEncoder)
            else:
                np.savez(filename, **weights)
        else:
            import h5py
            with h5py.File(filename, 'w') as f:
                f.create_dataset('time_samples',
                                 data=self.weights['time_samples'])
                for basis_type in ['linear', 'quadratic']:
                    key = f'prior_range_{basis_type}'
                    if key in self.weights:
                        grp = f.create_group(key)
                        for param_name in self.weights[key]:
                            grp.create_dataset(
                                param_name, data=self.weights[key][param_name])
                    key = f'frequency_nodes_{basis_type}'
                    if key in self.weights:
                        grp = f.create_group(key)
                        for i in range(len(self.weights[key])):
                            grp.create_dataset(
                                str(i), data=self.weights[key][i])
                    for ifo in self.interferometers:
                        key = f"{ifo.name}_{basis_type}"
                        grp = f.create_group(key)
                        for i in range(len(self.weights[key])):
                            grp.create_dataset(
                                str(i), data=self.weights[key][i])

    def load_weights(self, filename, format=None):
        """
        Load ROQ weights. format should be json, npz, or hdf5.
        json or npz file is assumed to contain weights from a single basis.
        Support for json format is deprecated as of :code:`v2.1` and will be
        removed in :code:`v2.2`, another method should be used by default.

        Parameters
        ==========
        filename : str
            The name of the file to save the weights to.
        format : str
            The format to save the data to, this should be one of
            :code:`"hdf5"`, :code:`"npz"`, default=:code:`"hdf5"`.

        Returns
        =======
        weights: dict
            Dictionary containing the ROQ weights.
        """
        if format is None:
            format = filename.split(".")[-1]
        if format not in ["json", "npz", "hdf5"]:
            raise IOError(f"Format {format} not recognized.")
        if format == "json":
            import warnings

            warnings.warn(
                "json format for ROQ weights is deprecated, use hdf5 instead.",
                DeprecationWarning
            )
        logger.info(f"Loading ROQ weights from {filename}")
        if format == "json" or format == "npz":
            # Old file format assumed to contain only a single basis
            if format == "json":
                with open(filename, 'r') as file:
                    weights = json.load(file, object_hook=decode_bilby_json)
            else:
                # Wrap in dict to load data into memory
                weights = dict(np.load(filename))
            for basis_type in ['linear', 'quadratic']:
                for ifo in self.interferometers:
                    key = f'{ifo.name}_{basis_type}'
                    weights[key] = [weights[key]]
        else:
            weights = dict()
            import h5py
            with h5py.File(filename, 'r') as f:
                weights['time_samples'] = f['time_samples'][()]
                for basis_type in ['linear', 'quadratic']:
                    key = f'prior_range_{basis_type}'
                    if key in f:
                        idxs_in_prior_range, selected_prior_ranges = \
                            self._select_prior_ranges(f[key])
                        weights[key] = selected_prior_ranges
                    else:
                        idxs_in_prior_range = [0]
                    key = f'frequency_nodes_{basis_type}'
                    if key in f:
                        weights[key] = [f[key][str(i)][()]
                                        for i in idxs_in_prior_range]
                    for ifo in self.interferometers:
                        key = f"{ifo.name}_{basis_type}"
                        weights[key] = [f[key][str(i)][()]
                                        for i in idxs_in_prior_range]
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
            from scipy.integrate import simpson
            integrand1 = np.power(freq, -7. / 3) / psd
            integral1 = simpson(y=integrand1, x=freq)
            integrand3 = np.power(freq, 2. / 3.) / (psd * integral1)
            f_3_bar = simpson(y=integrand3, x=freq)

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

    def generate_time_sample_from_marginalized_likelihood(self, signal_polarizations=None):
        from ...core.utils.random import rng

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
        return rng.choice(times, p=time_post)


class BilbyROQParamsRangeError(Exception):
    pass
