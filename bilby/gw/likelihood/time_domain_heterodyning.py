import copy

import numpy as np
from scipy.optimize import differential_evolution

from ...core.utils import gravitational_constant, logger, solar_mass, speed_of_light
from ..conversion import component_masses_to_chirp_mass
from ..time_domain_utils import gohberg_semencul_product, times_from_merger
from .relative import RelativeBinningGravitationalWaveTransient
from .time_domain import TimeDomainGravitationalWaveTransient


class TimeDomainHeterodyningGravitationalWaveTransient(TimeDomainGravitationalWaveTransient):
    """ A time-domain gravitational-wave transient likelihood object which uses
    heterodyning (relative binning) to calculate a fast likelihood, for the
    IMRPhenomT, IMRPhenomTHM, IMRPhenomTP and IMRPhenomTPHM waveform models.

    The waveform is evaluated only at the edges of time bins, and the
    likelihood is computed from summary data built from a fiducial waveform.
    See Sharma et al. arXiv:2601.11239, and Cornish arXiv:1007.4820 and
    Zackay et al. arXiv:1806.08792 for the frequency-domain method.

    Parameters
    ==========
    interferometers: list, bilby.gw.detector.InterferometerList
        A list of `bilby.detector.Interferometer` instances - contains the
        detector data, analysis windows ("imr", "inspiral" or "post_inspiral")
        and autocovariance functions
    waveform_generator: `bilby.waveform_generator.WaveformGenerator`
        A WaveformGenerator with
        :code:`bilby.gw.source.lal_binary_black_hole_time_domain_heterodyning`
        as its time-domain source model
    fiducial_parameters: dict, optional
        A starting guess for initial parameters of the event for finding the
        maximum likelihood (fiducial) waveform. These should be specified in
        the same parameter basis as the one that sampling is carried out in.
        If no fiducial parameters are passed, a random point will be drawn from
        the prior.
    maximization_kwargs: dict, optional
        Keyword arguments for :code:`scipy.optimize.differential_evolution`
    update_fiducial_parameters: bool
        Whether to try to update the fiducial parameters by finding the maximum
        likelihood point using scipy differential evolution. Default is
        :code:`False`.
    parameters_to_be_updated: list, optional
        The parameters over which the likelihood is maximized, within their
        prior bounds. Default is :code:`["chirp_mass", "mass_ratio"]`.
    distance_marginalization: bool, optional
        If true, marginalize over distance in the likelihood.
        This uses a look up table calculated at run time.
        The distance prior is set to be a delta function at the minimum
        distance allowed in the prior being marginalised over.
    priors: dict, optional
        If given, used in the distance marginalization and to find the
        fiducial parameters.
    distance_marginalization_lookup_table: (dict, str), optional
        If a dict, dictionary containing the lookup_table, distance_array,
        (distance) prior_array, and reference_distance used to construct
        the table.
        If a string the name of a file containing these quantities.
    reference_frame: (str, bilby.gw.detector.InterferometerList, list), optional
        Definition of the reference frame for the sky location, as in
        :code:`GravitationalWaveTransient`.
    time_reference: str, optional
        Name of the reference for the sampled time parameter, as in
        :code:`GravitationalWaveTransient`.
    epsilon: tuple, optional
        The bin width in the phase proxy before and after `time_split`;
        smaller values give more bins (see :code:`time_domain_bin_edges`)
    time_split: float, optional
        The time relative to the merger (in s) at which `epsilon` changes
    spacing: tuple, optional
        The number of samples per bin after the merger: before
        `spacing_times[0]`, before `spacing_times[1]` and after
    spacing_times: tuple, optional
        The times relative to the merger (in s) at which `spacing` changes
    chi: float, optional
        Tunable parameter of the phase proxy. See https://arxiv.org/abs/1806.08792.
    fiducial_minimum_frequency: float, optional
        The minimum frequency (in Hz) of the fiducial waveform, default: chosen
        so that it starts before the analysis window
    time_marginalization, phase_marginalization, calibration_marginalization: bool, optional
        Not implemented for this likelihood; must be False
    use_float32: bool, optional
        If true, the summary data are stored and multiplied in single precision.
        Default is :code:`False`.

    Returns
    =======
    Likelihood: `bilby.core.likelihood.Likelihood`
        A likelihood object, able to compute the likelihood of the data given
        some model parameters.

    Notes
    =====
    The time-domain heterodyning likelihood does not currently support time,
    phase or calibration marginalization.
    """

    def __init__(self, interferometers, waveform_generator, fiducial_parameters=None,
                 maximization_kwargs=None, update_fiducial_parameters=False,
                 parameters_to_be_updated=("chirp_mass", "mass_ratio"),
                 distance_marginalization=False, priors=None,
                 distance_marginalization_lookup_table=None, reference_frame="sky",
                 time_reference="geocenter", epsilon=(0.5, 0.1), time_split=-0.02,
                 spacing=(2, 10, 100), spacing_times=(0.02, 0.1), chi=1,
                 fiducial_minimum_frequency=None,
                 time_marginalization=False, phase_marginalization=False,
                 calibration_marginalization=False, use_float32=False):
        super(TimeDomainHeterodyningGravitationalWaveTransient, self).__init__(
            interferometers=interferometers,
            waveform_generator=waveform_generator,
            distance_marginalization=distance_marginalization,
            priors=priors,
            distance_marginalization_lookup_table=distance_marginalization_lookup_table,
            reference_frame=reference_frame,
            time_reference=time_reference,
            placement="model_time",
            time_marginalization=time_marginalization,
            phase_marginalization=phase_marginalization,
            calibration_marginalization=calibration_marginalization,
        )
        self.epsilon = epsilon
        self.time_split = time_split
        self.spacing = spacing
        self.spacing_times = spacing_times
        self.chi = chi
        self.use_float32 = use_float32
        # this likelihood sets its own entries in waveform_arguments; copy the dictionary so that
        # they do not reach other waveform generators that share it
        self.waveform_generator.waveform_arguments = dict(self.waveform_generator.waveform_arguments)
        self.requested_fiducial_minimum_frequency = fiducial_minimum_frequency

        if fiducial_parameters is None:
            logger.info("Drawing fiducial parameters from prior.")
            fiducial_parameters = priors.sample()
        self.fiducial_parameters = dict(fiducial_parameters)

        if update_fiducial_parameters:
            logger.info("Using scipy optimization to find maximum likelihood parameters.")
            self.parameters_to_be_updated = list(parameters_to_be_updated)
            logger.info(f"Parameters over which likelihood is maximized: {self.parameters_to_be_updated}")
            self.parameter_bounds = self.get_bounds_from_priors(self.priors)
            self.fiducial_parameters = self.find_maximum_likelihood_parameters(
                self.parameter_bounds, maximization_kwargs=maximization_kwargs)

        if self.distance_marginalization:
            self.fiducial_parameters["luminosity_distance"] = self._ref_dist
        self.set_fiducial_waveforms(self.fiducial_parameters)
        logger.info("Fiducial waveforms set up")
        self.setup_bins()
        self.compute_summary_data()
        logger.info("Summary Data Obtained")
        logger.info(f"Fiducial likelihood: {self.log_likelihood_ratio(self.fiducial_parameters):.2f}")

    def _model_times(self, interferometer, indices, parameters):
        """ Times (in s) of the analysis-window samples `indices` relative to the arrival of the merger """
        # the time of the first sample, plus the samples `indices` (not all N samples)
        first = times_from_merger(
            interferometer.time_array[interferometer.analysis_slice][0], 1,
            interferometer.sampling_frequency, parameters["geocent_time"],
            interferometer.time_delay_from_geocenter(parameters["ra"], parameters["dec"], parameters["geocent_time"]))
        return first[0] + np.asarray(indices) / interferometer.sampling_frequency

    def _modes_and_harmonics_at_combined_bin_edges(self, signal, model_times):
        """ Modes and harmonic factors of `signal` at the model times of each
        interferometer, from one waveform call (None if the waveform failed)
        """
        all_modes_and_harmonics = signal["heterodyning_terms_at"](np.concatenate(model_times))
        if all_modes_and_harmonics is None:
            return None
        all_modes, all_harmonics = all_modes_and_harmonics
        per_interferometer = list()
        first = 0
        for interferometer_model_times in model_times:
            interferometer_part = slice(first, first + len(interferometer_model_times))
            first += len(interferometer_model_times)
            modes = {key: mode[interferometer_part] for key, mode in all_modes.items()}
            # (Z, Zbar): arrays for the precessing models, constants otherwise
            harmonics = {
                key: tuple(np.asarray(factor)[interferometer_part] if np.ndim(factor) else factor
                           for factor in factors)
                for key, factors in all_harmonics.items()}
            per_interferometer.append((modes, harmonics))
        return per_interferometer

    def _r_tilde_coefficients(self, waveform_polarizations, interferometers, parameters):
        """ The heterodyning coefficients of each interferometer in each bin (None if the waveform failed) """
        modes_and_harmonics = self._modes_and_harmonics_at_combined_bin_edges(
            waveform_polarizations,
            [self._model_times(interferometer, self.bin_edges[interferometer.name], parameters)
             for interferometer in interferometers])
        if modes_and_harmonics is None:
            return None
        r_tilde = list()
        for interferometer, (modes, harmonics) in zip(interferometers, modes_and_harmonics):
            # as in Interferometer.get_time_domain_detector_response
            antenna_time = parameters["geocent_time"] if interferometer.reference_time is None \
                else interferometer.reference_time
            antenna_factor = (  # F = F_+ + i F_x
                interferometer.antenna_response(
                    parameters["ra"], parameters["dec"], antenna_time, parameters["psi"], "plus")
                + 1j * interferometer.antenna_response(
                    parameters["ra"], parameters["dec"], antenna_time, parameters["psi"], "cross"))
            # r_tilde_lm = h_lm (F Z_lm + (F Zbar_lm)^*) / h^0_lm at the bin edges
            r_tilde_at_edges = np.array([
                modes[mode] * (antenna_factor * harmonics[mode][0] + np.conj(antenna_factor * harmonics[mode][1]))
                if mode in modes else np.zeros(len(self.bin_edges[interferometer.name]))
                for mode in self.mode_keys]) * self._inverse_h0_at_edges[interferometer.name]
            r_tilde_0 = (r_tilde_at_edges[:, 1:] + r_tilde_at_edges[:, :-1]) / 2
            r_tilde_1 = (r_tilde_at_edges[:, 1:] - r_tilde_at_edges[:, :-1]) / self.bin_widths[interferometer.name]
            r_tilde.append(np.stack([r_tilde_0, r_tilde_1], axis=1))
        return r_tilde

    def set_fiducial_waveforms(self, parameters):
        """ Compute the modes of the fiducial waveform at the samples of the analysis
        window of each interferometer

        Parameters
        ==========
        parameters: dict
            The fiducial parameters
        """
        parameters = parameters.copy()
        parameters.update(self.get_sky_frame_parameters(parameters=parameters))
        self._fiducial_times = {
            interferometer.name: self._model_times(
                interferometer, np.arange(len(self._data_times_inverse_covariance[interferometer.name])), parameters)
            for interferometer in self.interferometers}
        window_start = -min(times[0] for times in self._fiducial_times.values())
        # leading-order (2, 2) frequency 1.5 window_start + 0.1 s before the merger, minus 1 Hz
        converted = self.waveform_generator.parameter_conversion(parameters.copy())[0]
        chirp_mass_time = component_masses_to_chirp_mass(converted["mass_1"], converted["mass_2"]) \
            * solar_mass * gravitational_constant / speed_of_light ** 3
        frequency = (5 / 256 * chirp_mass_time ** (-5 / 3) / (1.5 * max(window_start, 0) + 0.1)) ** (3 / 8) / np.pi
        derived = min(max(frequency - 1, 1.0),
                      self.waveform_generator.waveform_arguments.get("minimum_frequency", 20.0))
        requested = self.requested_fiducial_minimum_frequency
        if requested is not None and requested > derived:
            raise ValueError(
                f"fiducial_minimum_frequency ({requested:g} Hz) must not be above {derived:.2f} Hz "
                "(the lower of the waveforms' minimum_frequency and the frequency derived from the fiducial chirp mass "
                "and the start of the analysis window)")
        self.fiducial_minimum_frequency = derived if requested is None else requested
        while True:
            self._set_fiducial()
            modes_and_harmonics = self._modes_and_harmonics_at_combined_bin_edges(
                self.waveform_generator.time_domain_strain(parameters),
                [self._fiducial_times[interferometer.name] for interferometer in self.interferometers])
            self._unset_fiducial()
            if modes_and_harmonics is None:
                raise ValueError(f"Cannot compute fiducial waveforms for {parameters}")
            if all(np.any(modes[(2, 2)][:1]) for modes, _ in modes_and_harmonics):
                break
            if requested is not None or self.fiducial_minimum_frequency <= 1:
                raise ValueError(
                    f"The fiducial waveform from {self.fiducial_minimum_frequency:.2f} Hz does not start "
                    f"before the analysis window for {parameters}")
            self.fiducial_minimum_frequency = max(self.fiducial_minimum_frequency - 1, 1)
        logger.info(f"Fiducial waveform from {self.fiducial_minimum_frequency:.2f} Hz")
        self.fiducial_modes = dict()
        self._fiducial_end = dict()
        for interferometer, (modes, _) in zip(self.interferometers, modes_and_harmonics):
            self.fiducial_modes[interferometer.name] = dict()
            ends = list()
            for key, mode in modes.items():
                nonzero = np.flatnonzero(mode)
                if len(nonzero) == 0:  # mode not in the fiducial waveform
                    continue
                mode[nonzero[-1] + 1:] = mode[nonzero[-1]]  # hold the last non-zero value after the ringdown
                self.fiducial_modes[interferometer.name][key] = mode
                ends.append(nonzero[-1])
            self._fiducial_end[interferometer.name] = max(ends)
        self.mode_keys = sorted({key for modes in self.fiducial_modes.values() for key in modes})
        logger.debug(f"Modes of the heterodyning basis: {self.mode_keys}")

    @staticmethod
    def time_domain_bin_edges(times, epsilon=(0.5, 0.1), time_split=-0.02, spacing=(2, 10, 100),
                              spacing_times=(0.02, 0.1), chi=1):
        """ Bin edges for the time-domain heterodyning likelihood

        Before the merger, the bins are equally spaced in a phase proxy as in
        Zackay et al. arXiv:1806.08792, with powers of time in place of powers of
        frequency; after the merger, they have a fixed number of samples. See
        Sharma et al. arXiv:2601.11239.

        Parameters
        ==========
        times: array_like
            Increasing times relative to the merger (in s)
        epsilon, time_split, spacing, spacing_times, chi:
            See :code:`TimeDomainHeterodyningGravitationalWaveTransient`

        Returns
        =======
        array_like: The indices of the bin edges in `times`, first and last included
        """
        gamma = np.array([5 / 8, 3 / 8, 1 / 4, -3 / 8])  # powers of |t| in the phase proxy
        edges = [0, len(times) - 1]
        inspiral = np.flatnonzero(times < 0)
        if len(inspiral) > 1:
            t = np.abs(times[inspiral])
            d_alpha = 2 * np.pi * chi / np.where(gamma < 0, t.min(), t.max()) ** gamma
            proxy = np.sum(np.sign(gamma) * d_alpha * t[:, None] ** gamma, axis=1)
            for region, region_epsilon in zip((times[inspiral] <= time_split, times[inspiral] > time_split), epsilon):
                if np.sum(region) < 2:
                    continue
                phase = np.abs(proxy[region] - proxy[region][0])
                number_of_bins = max(int(phase[-1] / region_epsilon), 1)
                targets = np.linspace(0, phase[-1], number_of_bins + 1)
                edges.extend(inspiral[region][np.searchsorted(phase, targets, side="right") - 1])
        merger = inspiral[-1] if len(inspiral) else 0
        boundaries = [merger] + [max(merger, np.searchsorted(times, x)) for x in spacing_times] + [len(times)]
        for start, stop, step in zip(boundaries[:-1], boundaries[1:], spacing):
            edges.extend(range(start, stop, step))
        return np.unique(edges)

    def setup_bins(self):
        """ Set up the bin edges of each interferometer from its fiducial waveform """
        self.bin_edges = dict()
        self.bin_widths = dict()
        self.number_of_bins = dict()
        for interferometer in self.interferometers:
            name = interferometer.name
            times = self._fiducial_times[name]
            end = self._fiducial_end[name]
            edges = self.time_domain_bin_edges(
                times[:end + 1],
                epsilon=self.epsilon,
                time_split=self.time_split,
                spacing=self.spacing,
                spacing_times=self.spacing_times,
                chi=self.chi)
            if end < len(times) - 1:
                edges = np.append(edges, len(times) - 1)
            self.bin_edges[name] = edges
            self.bin_widths[name] = np.diff(times[edges])
            self.number_of_bins[name] = len(edges) - 1
            logger.info(
                f"{name}: {self.number_of_bins[name]} bins between {times[edges[0]]:.3f} s "
                f"and {times[edges[-1]]:.3f} s from the merger")

    def compute_summary_data(self):
        """ Compute the summary data of each interferometer from its fiducial waveform """
        self.summary_data = dict()
        self._inverse_h0_at_edges = dict()
        for interferometer in self.interferometers:
            name = interferometer.name
            edges = self.bin_edges[name]  # sample indices of the bin edges: first and last sample included
            t = self._fiducial_times[name]  # t_j
            number_of_samples, number_of_bins = len(t), len(edges) - 1
            samples_per_bin = np.diff(edges)
            samples_per_bin[-1] += 1
            bin_of_sample = np.repeat(np.arange(number_of_bins), samples_per_bin)
            bin_centres = (t[edges[:-1]] + t[edges[1:]]) / 2  # t_b
            time_from_bin_centre = t - bin_centres[bin_of_sample]  # t_j - t_b

            h0 = np.array([self.fiducial_modes[name][mode] for mode in self.mode_keys])  # h^0_lm(t_j)
            self._inverse_h0_at_edges[name] = 1 / h0[:, edges]

            # basis functions h^0_lm (t_j - t_b)^p as real functions, in the order (Re/-Im, lm, p);
            # basis[i] restricted to bin b is the column (i, b) of V
            h0_times_power_of_time = np.stack([h0, h0 * time_from_bin_centre], axis=1)
            basis = np.concatenate([h0_times_power_of_time.real, -h0_times_power_of_time.imag])
            basis = basis.reshape(-1, number_of_samples)
            number_of_basis = len(basis)

            # a = V^T C^-1 d: basis times C^-1 d, summed over the samples of each bin
            a = np.add.reduceat(basis * self._data_times_inverse_covariance[name], edges[:-1], axis=1)

            # S = V^T C^-1 V without the terms quadratic in the slopes r_tilde_1 (p = p' = 1, odd i):
            # C^-1 times each column (i, b) of V, then times basis[j], summed over the samples of
            # each bin b'; S is symmetric, so only j >= i is computed
            x, y = interferometer.gohberg_semencul_vectors
            S = np.zeros((number_of_basis, number_of_bins, number_of_basis, number_of_bins))
            for i in range(number_of_basis):
                columns = np.zeros((number_of_bins, number_of_samples))
                columns[bin_of_sample, np.arange(number_of_samples)] = basis[i]
                C_inv_columns = np.array([gohberg_semencul_product(x, y, column) for column in columns]).T
                for j in range(i, number_of_basis):
                    if i % 2 == 1 and j % 2 == 1:  # r_tilde_1 r_tilde_1
                        continue
                    block = np.add.reduceat(basis[j][:, None] * C_inv_columns, edges[:-1], axis=0)
                    S[j, :, i, :] = block
                    S[i, :, j, :] = block.T

            # S = sum_n lambda_n e_n e_n^T, keeping the eigenvalues above 1e-12 of the largest
            size = number_of_basis * number_of_bins
            eigenvalues, eigenvectors = np.linalg.eigh(S.reshape(size, size))
            keep = np.abs(eigenvalues) > 1e-12 * np.max(np.abs(eigenvalues))

            self.summary_data[name] = dict(
                a=a.ravel(), S_eigenvalues=eigenvalues[keep],
                S_eigenvectors=np.ascontiguousarray(
                    eigenvectors[:, keep].T, dtype=np.float32 if self.use_float32 else np.float64),
                bin_of_sample=bin_of_sample, time_from_bin_centre=time_from_bin_centre)
            logger.info(f"{name}: {len(self.mode_keys)} modes, {number_of_bins} bins, "
                        f"S of size {size} and rank {np.sum(keep)}")

    def calculate_snrs(self, waveform_polarizations, interferometer, *, return_array=True, parameters,
                       r_tilde=None):
        """ Compute the snrs

        Parameters
        ==========
        waveform_polarizations: dict
            The output of the heterodyning source model
        interferometer: bilby.gw.detector.Interferometer
            The bilby interferometer object
        return_array: bool
            Not used; kept for the :code:`GravitationalWaveTransient` interface
        parameters: dict
            The parameters of the signal
        r_tilde: array_like, optional
            The heterodyning coefficients of this interferometer, if already computed

        Returns
        =======
        calculated_snrs: _CalculatedSNRs
            An object containing the SNR quantities (None if the waveform failed)
        """
        if r_tilde is None:
            r_tilde = self._r_tilde_coefficients(waveform_polarizations, [interferometer], parameters)
            if r_tilde is None:
                return None
            r_tilde = r_tilde[0]
        summary = self.summary_data[interferometer.name]
        v = np.concatenate([r_tilde.real.ravel(), r_tilde.imag.ravel()])
        d_inner_s = float(summary["a"] @ v)
        # e_n . v in the precision of the stored eigenvectors (see use_float32)
        projections = summary["S_eigenvectors"] @ v.astype(summary["S_eigenvectors"].dtype, copy=False)
        s_inner_s = float(summary["S_eigenvalues"] @ projections.astype(np.float64) ** 2)
        return self._CalculatedSNRs(
            d_inner_h=d_inner_s,
            optimal_snr_squared=s_inner_s,
            complex_matched_filter_snr=d_inner_s / s_inner_s ** 0.5 if s_inner_s > 0 else 0.0,
            d_inner_h_array=None,
            optimal_snr_squared_array=None,
        )

    def log_likelihood_ratio(self, parameters):
        parameters = copy.deepcopy(parameters)
        parameters.update(self.get_sky_frame_parameters(parameters))
        signal = self.waveform_generator.time_domain_strain(parameters)
        r_tilde = None if signal is None else self._r_tilde_coefficients(signal, self.interferometers, parameters)
        if r_tilde is None:
            return np.nan_to_num(-np.inf)
        total_snrs = self._CalculatedSNRs()
        for interferometer, interferometer_r_tilde in zip(self.interferometers, r_tilde):
            total_snrs += self.calculate_snrs(
                signal, interferometer, parameters=parameters, r_tilde=interferometer_r_tilde)
        log_l = self.compute_log_likelihood_from_snrs(total_snrs, parameters=parameters)
        return float(np.real(log_l))

    def _compute_full_waveform(self, signal_polarizations, interferometer, parameters):
        """ The heterodyned template at the samples of the analysis window """
        r_tilde = self._r_tilde_coefficients(signal_polarizations, [interferometer], parameters)
        if r_tilde is None:
            return None
        summary = self.summary_data[interferometer.name]
        r_tilde_at_samples = r_tilde[0][:, :, summary["bin_of_sample"]]
        h0 = np.array([self.fiducial_modes[interferometer.name][mode] for mode in self.mode_keys])
        return np.sum(((r_tilde_at_samples[:, 0] + r_tilde_at_samples[:, 1] * summary["time_from_bin_centre"])
                       * h0).real, axis=0)

    def _rescale_signal(self, signal, new_distance):
        factor = self._ref_dist / new_distance
        terms_at = signal["heterodyning_terms_at"]

        def rescaled(times):
            terms = terms_at(times)
            if terms is None:
                return None
            modes, harmonics = terms
            return {key: mode * factor for key, mode in modes.items()}, harmonics

        signal["heterodyning_terms_at"] = rescaled

    def find_maximum_likelihood_parameters(self, parameter_bounds, maximization_kwargs=None):
        """ Find the maximum of the time-domain likelihood over
        :code:`parameters_to_be_updated`, using scipy differential evolution

        Parameters
        ==========
        parameter_bounds: list
            The bounds of the parameters
        maximization_kwargs: dict, optional
            Keyword arguments for :code:`scipy.optimize.differential_evolution`

        Returns
        =======
        dict: The updated fiducial parameters
        """
        if maximization_kwargs is None:
            maximization_kwargs = dict()
        exact_likelihood = self.exact_likelihood()

        def negative_log_likelihood_ratio(parameter_list):
            return -exact_likelihood.log_likelihood_ratio(self.get_parameter_dictionary_from_list(parameter_list))

        start = self.get_parameter_list_from_dictionary(self.fiducial_parameters)
        logger.info(f"Initial ln likelihood ratio: {-negative_log_likelihood_ratio(start):.2f}")
        output = differential_evolution(
            negative_log_likelihood_ratio, bounds=parameter_bounds, x0=start, **maximization_kwargs)
        updated_parameters = self.get_parameter_dictionary_from_list(output["x"])
        logger.info(f"Maximum ln likelihood ratio: {-output['fun']:.2f} at "
                    f"{dict(zip(self.parameters_to_be_updated, output['x']))}")
        return updated_parameters

    def exact_likelihood(self):
        """ The exact time-domain likelihood (model time) with the same data, waveform
        model, priors and settings as this likelihood

        Returns
        =======
        bilby.gw.likelihood.TimeDomainGravitationalWaveTransient
        """
        from ..source import lal_binary_black_hole_model_time

        waveform_arguments = {
            key: value for key, value in self.waveform_generator.waveform_arguments.items()
            if key not in ("fiducial", "fiducial_minimum_frequency")}
        return TimeDomainGravitationalWaveTransient(
            interferometers=self.interferometers,
            waveform_generator=self.waveform_generator.__class__(
                duration=self.waveform_generator.duration,
                sampling_frequency=self.waveform_generator.sampling_frequency,
                start_time=self.waveform_generator.start_time,
                time_domain_source_model=lal_binary_black_hole_model_time,
                parameter_conversion=self.waveform_generator.parameter_conversion,
                waveform_arguments=waveform_arguments),
            distance_marginalization=self.distance_marginalization,
            priors=self.priors,
            distance_marginalization_lookup_table=(
                self.cached_lookup_table_filename if self.distance_marginalization else None),
            reference_frame=self.reference_frame,
            time_reference=self.time_reference,
            placement="model_time")

    def compare_with_exact_likelihood(self, samples, outdir=".", label="heterodyning", n_samples=None):
        """ Compare the heterodyned likelihood with the exact time-domain likelihood
        (model time) at a set of samples, e.g. the posterior of a run

        Saves two figures in `outdir`: `{label}_heterodyning_vs_exact.png` (both log
        likelihood ratios against the chirp mass, heterodyned against exact, and their
        relative difference against the chirp mass) and
        `{label}_heterodyning_differences.png` (heterodyned minus exact against the
        other parameters).

        Parameters
        ==========
        samples: pandas.DataFrame or dict
            The samples, e.g. :code:`result.posterior`
        outdir: str
            The directory for the figures
        label: str
            The label used in the file names
        n_samples: int, optional
            The number of samples drawn at random from `samples` (default: all)

        Returns
        =======
        dict: The samples used and the exact and heterodyned log likelihood ratios
        """
        import matplotlib.pyplot as plt
        import pandas as pd

        from ...core.utils import safe_save_figure

        samples = pd.DataFrame(samples)
        if n_samples is not None and n_samples < len(samples):
            samples = samples.sample(n_samples)
        exact_likelihood = self.exact_likelihood()
        exact, heterodyned = np.array([
            (exact_likelihood.log_likelihood_ratio(dict(sample)), self.log_likelihood_ratio(dict(sample)))
            for _, sample in samples.iterrows()]).T
        difference = heterodyned - exact
        logger.info(f"Heterodyned - exact ln likelihood ratio at {len(samples)} samples: median "
                    f"{np.median(difference):.3g}, maximum absolute value {np.max(np.abs(difference)):.3g}")
        if "chirp_mass" in samples:
            chirp_mass = samples["chirp_mass"].to_numpy()
        else:
            chirp_mass = component_masses_to_chirp_mass(samples["mass_1"], samples["mass_2"]).to_numpy()

        fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
        axes[0].plot(chirp_mass, exact, ".", label="exact (model time)")
        axes[0].plot(chirp_mass, heterodyned, ".", alpha=0.5, label="heterodyned")
        axes[0].set_xlabel("chirp_mass")
        axes[0].set_ylabel("log likelihood ratio")
        axes[0].legend()
        limits = [min(exact.min(), heterodyned.min()), max(exact.max(), heterodyned.max())]
        axes[1].plot(exact, heterodyned, ".")
        axes[1].plot(limits, limits, "k--", label="y = x")
        axes[1].set_xlabel("exact log likelihood ratio")
        axes[1].set_ylabel("heterodyned log likelihood ratio")
        axes[1].legend()
        axes[2].plot(chirp_mass, difference / np.abs(exact), ".")
        axes[2].axhline(0, color="k", ls="--")
        axes[2].set_xlabel("chirp_mass")
        axes[2].set_ylabel("(heterodyned - exact) / |exact|")
        fig.tight_layout()
        safe_save_figure(fig, f"{outdir}/{label}_heterodyning_vs_exact.png")
        plt.close(fig)

        keys = ["mass_ratio", "chi_1", "chi_2", "a_1", "a_2", "tilt_1", "tilt_2", "theta_jn", "geocent_time"]
        if not self.distance_marginalization:
            keys.append("luminosity_distance")
        keys = [key for key in keys if key in samples and samples[key].nunique() > 1]
        if len(keys) > 0:
            columns = min(len(keys), 4)
            rows = int(np.ceil(len(keys) / columns))
            fig, axes = plt.subplots(rows, columns, figsize=(4 * columns, 3.5 * rows), squeeze=False)
            for ax, key in zip(axes.flat, keys):
                if key == "geocent_time":
                    ax.plot(samples[key] - self.fiducial_parameters["geocent_time"], difference, ".")
                    ax.set_xlabel("geocent_time - fiducial geocent_time (s)")
                else:
                    ax.plot(samples[key], difference, ".")
                    ax.set_xlabel(key)
                ax.axhline(0, color="k", ls="--")
                ax.set_ylabel("heterodyned - exact")
            for ax in list(axes.flat)[len(keys):]:
                ax.set_visible(False)
            fig.tight_layout()
            safe_save_figure(fig, f"{outdir}/{label}_heterodyning_differences.png")
            plt.close(fig)
        return dict(samples=samples, exact=exact, heterodyned=heterodyned)

    def _set_fiducial(self):
        """ Make the waveform generator generate the fiducial waveform """
        self.waveform_generator.waveform_arguments["fiducial"] = 1
        self.waveform_generator.waveform_arguments["fiducial_minimum_frequency"] = self.fiducial_minimum_frequency
        self.waveform_generator._cache["parameters"] = None

    def _unset_fiducial(self):
        """ Make the waveform generator generate waveforms at sampled parameters """
        self.waveform_generator.waveform_arguments["fiducial"] = 0
        self.waveform_generator._cache["parameters"] = None

    get_parameter_dictionary_from_list = RelativeBinningGravitationalWaveTransient.get_parameter_dictionary_from_list
    get_parameter_list_from_dictionary = RelativeBinningGravitationalWaveTransient.get_parameter_list_from_dictionary
    get_bounds_from_priors = RelativeBinningGravitationalWaveTransient.get_bounds_from_priors

    @property
    def meta_data(self):
        meta_data = super(TimeDomainHeterodyningGravitationalWaveTransient, self).meta_data
        meta_data.update(
            fiducial_parameters=self.fiducial_parameters,
            parameters_to_be_updated=getattr(self, "parameters_to_be_updated", None),
            fiducial_minimum_frequency=self.fiducial_minimum_frequency,
            mode_keys=self.mode_keys,
            epsilon=self.epsilon,
            time_split=self.time_split,
            spacing=self.spacing,
            spacing_times=self.spacing_times,
            chi=self.chi,
            number_of_bins=self.number_of_bins,
            bin_edges=self.bin_edges,
            bin_edge_times={name: self._fiducial_times[name][edges] for name, edges in self.bin_edges.items()})
        return meta_data
