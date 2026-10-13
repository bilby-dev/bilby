import copy

import numpy as np
from scipy.optimize import differential_evolution

from ...core.utils import gravitational_constant, logger, solar_mass, speed_of_light
from ..conversion import component_masses_to_chirp_mass
from ..time_domain_utils import gohberg_semencul_product, times_from_merger
from .relative import RelativeBinningGravitationalWaveTransient
from .time_domain import TimeDomainGravitationalWaveTransient


class TimeDomainHeterodyningGravitationalWaveTransient(TimeDomainGravitationalWaveTransient):
    """ A time-domain gravitational-wave transient likelihood which uses
    heterodyning (relative binning in time) to calculate a fast likelihood,
    for IMRPhenomT, IMRPhenomTHM, IMRPhenomTP and IMRPhenomTPHM. See Sharma et
    al. arXiv:2601.11239 and, in the frequency domain, Zackay et al.
    arXiv:1806.08792 and :code:`RelativeBinningGravitationalWaveTransient`.

    The waveform is written as (see
    :code:`bilby.gw.model_time_waveforms.heterodyning_terms_at_model_times`)

    .. math::

        h_+ - i h_\\times = \\sum_q \\left[ Z_q(t) h_q(t) + \\bar{Z}_q(t) h_q^*(t) \\right],

    with :math:`h_q` the modes with m >= 0 (co-precessing modes for
    IMRPhenomTP/TPHM). The signal in an interferometer is
    :math:`s = \\Re[(F_+ + i F_\\times)(h_+ - i h_\\times)] = \\Re \\sum_q w_q u_q`,
    where :math:`u_q` are the modes of a fiducial waveform at the data samples
    and

    .. math::

        w_q(t) = \\frac{h_q(t)}{u_q(t)} \\left[ F Z_q(t) + F^* \\bar{Z}_q^*(t) \\right],
        \\quad F = F_+ + i F_\\times.

    :math:`w_q` is evaluated at the edges of time bins and taken to be linear
    in each bin, :math:`w_q = w_{q,0}(b) + w_{q,1}(b) (t - t_b)`. Then
    :math:`d^T C^{-1} s` and :math:`s^T C^{-1} s` are the inner products of
    this piecewise-linear model, without the terms quadratic in the slopes
    :math:`w_{q,1}` (as in Sharma et al.), computed from summary data:
    :math:`s = V v` with :math:`v = (\\Re w, \\Im w)`, so that
    :math:`d^T C^{-1} s = a^T v` and :math:`s^T C^{-1} s = v^T S v`
    (from the eigen-decomposition of S, without its negligible eigenvalues), with
    :math:`a = V^T C^{-1} d` and :math:`S = V^T C^{-1} V`.

    The waveform is evaluated only at the bin edges, with one waveform call for
    all interferometers.

    The waveforms switch on abruptly at their minimum frequency, so their
    ratios are smooth only where both are on. The bins cover the whole
    analysis window, and here:

    - the fiducial waveform starts before the window (its minimum frequency
      is derived from the chirp mass and the start of the window, or
      given), and after its ringdown each of its
      modes is held at its last non-zero value, so it is non-zero everywhere
      in the window;
    - the waveforms at sampled parameters start at
      :code:`sampled_minimum_frequency` (default: 2 Hz below the
      interferometers' minimum_frequency), a few cycles before the waveforms
      reach it.

    With the interferometers' minimum_frequency at or below the frequency of
    the signal at the start of the analysis window, every waveform (injected,
    fiducial and sampled) is then on in the whole window.

    The waveform's :code:`minimum_frequency` argument is not used by this
    likelihood (it is for the injection and the exact likelihood).

    Parameters
    ----------
    interferometers: list, bilby.gw.detector.InterferometerList
        As for :code:`TimeDomainGravitationalWaveTransient`.
    waveform_generator: bilby.gw.waveform_generator.WaveformGenerator
        A WaveformGenerator with
        :code:`bilby.gw.source.lal_binary_black_hole_time_domain_heterodyning`
        as its time-domain source model.
    fiducial_parameters: dict, optional
        A starting guess for initial parameters of the event for finding the
        maximum likelihood (fiducial) waveform. These should be specified in
        the same parameter basis as the one that sampling is carried out in.
        If no fiducial parameters are passed, a random point will be drawn from
        the prior.
    update_fiducial_parameters: bool
        Whether to try to update the fiducial parameters by finding the maximum
        likelihood point using scipy differential evolution. Default is
        :code:`False`.
    parameters_to_be_updated: list, optional
        Parameters over which the likelihood is maximized when
        :code:`update_fiducial_parameters` is True, within their prior bounds.
        Default is :code:`["chirp_mass", "mass_ratio"]`.
    maximization_kwargs: dict, optional
        Keyword arguments for :code:`scipy.optimize.differential_evolution`.
    distance_marginalization: bool, optional
        If true, marginalize over distance in the likelihood.
        This uses a look up table calculated at run time.
        The distance prior is set to be a delta function at the minimum
        distance allowed in the prior being marginalised over.
    priors: dict, optional
        If given, used in the distance marginalization and to find the
        fiducial parameters (the prior bounds are the bounds of the
        optimization).
    distance_marginalization_lookup_table: (dict, str), optional
        As for :code:`TimeDomainGravitationalWaveTransient`.
    reference_frame: (str, bilby.gw.detector.InterferometerList, list), optional
        Definition of the reference frame for the sky location.
    time_reference: str, optional
        Name of the reference for the sampled time parameter.
        - "geocent"/"geocenter": sample in the time at the Earth's center,
          this is the default
        - e.g., "H1": sample in the time of arrival at H1
    epsilon: tuple or dict, optional
        Bin size in the inspiral: phase-proxy spacing before and after
        :code:`time_split` (see :code:`time_domain_bin_edges`). Smaller
        values give more bins. A dict gives a value per interferometer.
    time_split: float or dict, optional
        Time (s, relative to the merger) at which epsilon changes.
    spacing: tuple, optional
        Number of samples per bin after the merger, until
        :code:`spacing_times[0]`, until :code:`spacing_times[1]` and after.
    spacing_times: tuple, optional
        Times (s, relative to the merger) at which the spacing changes.
    chi: float, optional
        Tunable parameter of the phase proxy (as in relative binning).
    sampled_minimum_frequency: float, optional
        Frequency (Hz) at which the waveforms at sampled parameters start; at
        most (and by default) 2 Hz below the lowest minimum_frequency of the
        interferometers.
    fiducial_minimum_frequency: float, optional
        Frequency (Hz) at which the fiducial waveform starts. By default the
        lower of :code:`sampled_minimum_frequency` and the frequency derived
        from the fiducial chirp mass and the start of the analysis window
        (the leading-order frequency 1.5 x (time from the window start to the
        merger) + 0.1 s before the merger, minus 1 Hz); a given value must not
        be above that.
    time_marginalization, phase_marginalization, calibration_marginalization: bool
        Not implemented for this likelihood; must be False.

    Returns
    -------
    Likelihood: `bilby.core.likelihood.Likelihood`
        A likelihood object, able to compute the likelihood of the data given
        some model parameters.
    """

    def __init__(self, interferometers, waveform_generator, fiducial_parameters=None,
                 maximization_kwargs=None, update_fiducial_parameters=False,
                 parameters_to_be_updated=("chirp_mass", "mass_ratio"),
                 distance_marginalization=False, priors=None,
                 distance_marginalization_lookup_table=None, reference_frame="sky",
                 time_reference="geocenter", epsilon=(0.5, 0.1), time_split=-0.02,
                 spacing=(2, 10, 100), spacing_times=(0.02, 0.1), chi=1,
                 sampled_minimum_frequency=None, fiducial_minimum_frequency=None,
                 time_marginalization=False,
                 phase_marginalization=False, calibration_marginalization=False):
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
        # this likelihood sets its own entries in waveform_arguments; copy the dictionary so that
        # they do not reach other waveform generators that share it
        self.waveform_generator.waveform_arguments = dict(self.waveform_generator.waveform_arguments)
        self._set_sampled_minimum_frequency(sampled_minimum_frequency)
        self.requested_fiducial_minimum_frequency = fiducial_minimum_frequency

        if fiducial_parameters is None:
            logger.info("Drawing fiducial parameters from prior.")
            fiducial_parameters = priors.sample()
        self.fiducial_parameters = dict(fiducial_parameters)
        if self.distance_marginalization:
            self.fiducial_parameters["luminosity_distance"] = self._ref_dist
        self.set_fiducial_waveforms(self.fiducial_parameters)
        logger.info("Initial fiducial waveforms set up")
        self.setup_bins()
        self.compute_summary_data()
        logger.info("Summary Data Obtained")

        if update_fiducial_parameters:
            logger.info("Using scipy optimization to find maximum likelihood parameters.")
            self.parameters_to_be_updated = list(parameters_to_be_updated)
            logger.info(f"Parameters over which likelihood is maximized: {self.parameters_to_be_updated}")
            self.parameter_bounds = self.get_bounds_from_priors(priors)
            self.fiducial_parameters = self.find_maximum_likelihood_parameters(
                self.parameter_bounds, maximization_kwargs=maximization_kwargs)
        logger.info(f"Fiducial likelihood: {self.log_likelihood_ratio(self.fiducial_parameters):.2f}")

    def _set_sampled_minimum_frequency(self, sampled_minimum_frequency):
        maximum = min(ifo.minimum_frequency for ifo in self.interferometers) - 2
        if sampled_minimum_frequency is None:
            sampled_minimum_frequency = maximum
        if sampled_minimum_frequency > maximum:
            raise ValueError(
                f"sampled_minimum_frequency ({sampled_minimum_frequency:g} Hz) must be at most {maximum:g} Hz "
                "(2 Hz below the interferometers' minimum_frequency)")
        if sampled_minimum_frequency <= 0:
            raise ValueError("sampled_minimum_frequency must be positive")
        self.sampled_minimum_frequency = sampled_minimum_frequency
        self.waveform_generator.waveform_arguments["sampled_minimum_frequency"] = sampled_minimum_frequency
        self._unset_fiducial()

    # ------------------------------------------------------------------
    # Waveform at given samples
    # ------------------------------------------------------------------
    def _model_times(self, interferometer, indices, parameters):
        """ Times of the analysis-window samples `indices`, relative to the
        arrival of the model's t=0 at the interferometer (as for placement="model_time"). """
        # the time of the first sample, plus the samples `indices` (not all N samples)
        first = times_from_merger(
            interferometer.strain_data.time_array[interferometer._analysis_slice().start], 1,
            interferometer.sampling_frequency, parameters["geocent_time"],
            interferometer.time_delay_from_geocenter(parameters["ra"], parameters["dec"], parameters["geocent_time"]))
        return first[0] + np.asarray(indices) / interferometer.sampling_frequency

    @staticmethod
    def _antenna_factor(interferometer, parameters):
        """ F = F_+ + i F_x. """
        time = parameters["geocent_time"] if interferometer.reference_time is None \
            else interferometer.reference_time
        plus, cross = (interferometer.antenna_response(
            parameters["ra"], parameters["dec"], time, parameters["psi"], mode) for mode in ("plus", "cross"))
        return plus + 1j * cross

    def _terms_at_samples(self, signal, interferometers, indices, parameters):
        """ Modes and harmonic factors at the samples `indices[name]` of each
        interferometer, from one waveform call. None if the waveform failed. """
        times = [self._model_times(interferometer, indices[interferometer.name], parameters)
                 for interferometer in interferometers]
        terms = signal["heterodyning_terms_at"](np.concatenate(times))
        if terms is None:
            return None
        modes, harmonics = terms
        per_interferometer = list()
        start = 0
        for interferometer_times in times:
            samples = slice(start, start + len(interferometer_times))
            start += len(interferometer_times)
            per_interferometer.append((
                {key: mode[samples] for key, mode in modes.items()},
                {key: tuple(np.asarray(z)[samples] if np.ndim(z) else z for z in pair)
                 for key, pair in harmonics.items()}))
        return per_interferometer

    def _heterodyning_coefficients(self, signal, interferometers, parameters):
        """ Coefficients w[q, p, b] of the piecewise-linear model in each
        interferometer (see the class docstring). None if the waveform failed. """
        terms = self._terms_at_samples(signal, interferometers, self.bin_edges, parameters)
        if terms is None:
            return None
        coefficients = list()
        for interferometer, (modes, harmonics) in zip(interferometers, terms):
            name = interferometer.name
            zeros = np.zeros(self._inverse_fiducial_at_edges[name].shape[1], dtype=complex)
            antenna = self._antenna_factor(interferometer, parameters)
            mode = np.array([modes.get(key, zeros) for key in self.mode_keys])
            harmonic = np.array([np.broadcast_to(harmonics[key][0], zeros.shape) if key in modes else zeros
                                 for key in self.mode_keys])
            conjugate_harmonic = np.array([np.broadcast_to(harmonics[key][1], zeros.shape) if key in modes else zeros
                                           for key in self.mode_keys])
            ratio = mode * (antenna * harmonic + np.conj(antenna * conjugate_harmonic)) \
                * self._inverse_fiducial_at_edges[name]
            coefficients.append(np.stack([
                (ratio[:, 1:] + ratio[:, :-1]) / 2,
                (ratio[:, 1:] - ratio[:, :-1]) / self.bin_widths[name]], axis=1))
        return coefficients

    # ------------------------------------------------------------------
    # Fiducial waveform, bins and summary data
    # ------------------------------------------------------------------
    def set_fiducial_waveforms(self, parameters):
        """ The modes of the fiducial waveform at the data samples of each
        interferometer, starting before the analysis window (see
        :code:`fiducial_minimum_frequency`) and held at their last non-zero
        values after the ringdown. """
        parameters = dict(parameters)
        parameters.update(self.get_sky_frame_parameters(parameters=parameters))
        self._fiducial_times = {
            interferometer.name: self._model_times(
                interferometer, np.arange(self._number_of_samples(interferometer)), parameters)
            for interferometer in self.interferometers}
        window_start = -min(times[0] for times in self._fiducial_times.values())
        # leading-order (2, 2) frequency 1.5 window_start + 0.1 s before the merger, minus 1 Hz
        chirp_mass_time = self._chirp_mass(parameters) * solar_mass * gravitational_constant / speed_of_light ** 3
        time_to_merger = 1.5 * max(window_start, 0) + 0.1
        frequency = (5 / 256 * chirp_mass_time ** (-5 / 3) / time_to_merger) ** (3 / 8) / np.pi
        derived = min(max(frequency - 1, 1.0), self.sampled_minimum_frequency)
        requested = self.requested_fiducial_minimum_frequency
        if requested is not None and requested > derived:
            raise ValueError(
                f"fiducial_minimum_frequency ({requested:g} Hz) must not be above {derived:.2f} Hz "
                "(the lower of sampled_minimum_frequency and the frequency derived from the fiducial chirp mass "
                "and the start of the analysis window)")
        self.fiducial_minimum_frequency = derived if requested is None else requested
        while True:
            terms = self._fiducial_terms(parameters)
            if all(np.any(modes[(2, 2)][:1]) for modes, _ in terms):
                break
            if requested is not None or self.fiducial_minimum_frequency <= 1:
                raise ValueError(
                    f"The fiducial waveform from {self.fiducial_minimum_frequency:.2f} Hz does not start "
                    f"before the analysis window for {parameters}")
            self.fiducial_minimum_frequency = max(self.fiducial_minimum_frequency - 1, 1)
        logger.info(f"Fiducial waveform from {self.fiducial_minimum_frequency:.2f} Hz")
        self._fiducial_end = {
            interferometer.name: max(np.flatnonzero(mode)[-1] for mode in modes.values() if np.any(mode))
            for interferometer, (modes, _) in zip(self.interferometers, terms)}
        self.fiducial_modes = dict()
        for interferometer, (modes, _) in zip(self.interferometers, terms):
            self.fiducial_modes[interferometer.name] = dict()
            for key, mode in modes.items():
                mode = np.array(mode)
                nonzero = np.flatnonzero(mode)
                if len(nonzero) > 0:  # hold the last non-zero value after the ringdown
                    mode[nonzero[-1] + 1:] = mode[nonzero[-1]]
                self.fiducial_modes[interferometer.name][key] = mode
        self.mode_keys = sorted({key for modes in self.fiducial_modes.values()
                                 for key, mode in modes.items() if np.any(mode)})
        logger.debug(f"Modes of the heterodyning basis: {self.mode_keys}")

    def _fiducial_terms(self, parameters):
        self._set_fiducial()
        signal = self.waveform_generator.time_domain_strain(parameters)
        terms = None if signal is None else self._terms_at_samples(
            signal, self.interferometers,
            {interferometer.name: np.arange(self._number_of_samples(interferometer))
             for interferometer in self.interferometers},
            parameters)
        self._unset_fiducial()
        if terms is None:
            raise ValueError(f"Cannot compute fiducial waveforms for {parameters}")
        return terms

    def _chirp_mass(self, parameters):
        """ Detector-frame chirp mass (solar masses). """
        converted = self.waveform_generator.parameter_conversion(dict(parameters))[0]
        return component_masses_to_chirp_mass(converted["mass_1"], converted["mass_2"])

    @staticmethod
    def _number_of_samples(interferometer):
        analysis_slice = interferometer._analysis_slice()
        return analysis_slice.stop - analysis_slice.start

    def _per_detector(self, value, name):
        return value[name] if isinstance(value, dict) else value

    def setup_bins(self):
        """ Bin edges (indices of analysis-window samples) for each
        interferometer, from the times relative to the fiducial merger in that
        interferometer (see :code:`time_domain_bin_edges`), up to the end of
        the fiducial ringdown, and one bin from there to the end of the
        analysis window. """
        self.bin_edges = dict()
        self.bin_widths = dict()
        self.number_of_bins = dict()
        for interferometer in self.interferometers:
            name = interferometer.name
            times = self._fiducial_times[name]
            end = self._fiducial_end[name]
            edges = time_domain_bin_edges(
                times[:end + 1],
                epsilon=self._per_detector(self.epsilon, name),
                time_split=self._per_detector(self.time_split, name),
                spacing=self._per_detector(self.spacing, name),
                spacing_times=self._per_detector(self.spacing_times, name),
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
        """ For each interferometer: :math:`a = V^T C^{-1} d` and
        :math:`S = V^T C^{-1} V` (see the class docstring). The columns of V
        are :math:`\\Re(u_q \\tau^p)` and :math:`-\\Im(u_q \\tau^p)` in each bin
        b, where :math:`\\tau = t - t_b`. """
        self.summary_data = dict()
        self._inverse_fiducial_at_edges = dict()
        for interferometer in self.interferometers:
            name = interferometer.name
            edges = self.bin_edges[name]
            first, last = edges[0], edges[-1]
            samples = slice(first, last + 1)
            bin_sizes = np.diff(edges)
            bin_sizes[-1] += 1
            bin_starts = edges[:-1] - first
            bin_of_sample = np.repeat(np.arange(len(bin_sizes)), bin_sizes)
            times = self._fiducial_times[name]
            bin_centers = (times[edges[1:]] + times[edges[:-1]]) / 2
            tau = times[samples] - bin_centers[bin_of_sample]

            fiducial = np.array([self.fiducial_modes[name].get(key, np.zeros(len(times)))
                                 for key in self.mode_keys])
            at_edges = fiducial[:, edges]
            self._inverse_fiducial_at_edges[name] = np.divide(
                1, at_edges, out=np.zeros_like(at_edges), where=at_edges != 0)
            basis = np.stack([fiducial[:, samples], fiducial[:, samples] * tau], axis=1)  # (K, 2, n)
            columns = np.concatenate([basis.real, -basis.imag]).reshape(-1, len(tau))  # (4K, n): (part, q, p)
            data_vector = np.add.reduceat(
                columns * self._data_times_inverse_covariance[name][samples], bin_starts, axis=1)

            summary = dict(
                data_vector=data_vector.ravel(), samples=samples, fiducial=fiducial[:, samples],
                tau=tau, bin_sizes=bin_sizes,
                matrix=self._summary_matrix(interferometer, columns, samples, bin_starts, bin_of_sample))
            logger.info(f"{name}: {len(self.mode_keys)} modes, {len(bin_sizes)} bins")
            self.summary_data[name] = summary

    def _summary_matrix(self, interferometer, columns, samples, bin_starts, bin_of_sample):
        """ S = V^T C^-1 V, one column group of V at a time (symmetric), as its
        eigenvalues and eigenvectors (rows). """
        number_of_groups, number_of_bins = len(columns), len(bin_starts)
        x, y = interferometer.gohberg_semencul_vectors
        rows = np.arange(samples.start, samples.stop)
        matrix = np.empty((number_of_groups, number_of_bins, number_of_groups, number_of_bins))
        for j in range(number_of_groups):
            group = np.zeros((number_of_bins, self._number_of_samples(interferometer)))
            group[bin_of_sample, rows] = columns[j]
            group = np.array([gohberg_semencul_product(x, y, column)[samples] for column in group]).T
            for i in range(j, number_of_groups):
                block = np.add.reduceat(columns[i][:, None] * group, bin_starts, axis=0)
                matrix[i, :, j, :] = block
                matrix[j, :, i, :] = block.T
        # drop the terms quadratic in the slopes (p = 1 for both columns)
        slope = np.arange(number_of_groups) % 2 == 1
        matrix[np.ix_(slope, np.arange(number_of_bins), slope, np.arange(number_of_bins))] = 0
        size = number_of_groups * number_of_bins
        # v^T S v = sum_k lambda_k (e_k . v)^2, keeping the eigenvalues above _EIGENVALUE_TOLERANCE of the largest
        eigenvalues, eigenvectors = np.linalg.eigh(matrix.reshape(size, size))
        keep = np.abs(eigenvalues) > _EIGENVALUE_TOLERANCE * np.max(np.abs(eigenvalues))
        logger.debug(f"{interferometer.name}: summary matrix of size {size} with rank {np.sum(keep)}")
        return eigenvalues[keep], np.ascontiguousarray(eigenvectors[:, keep].T)

    # ------------------------------------------------------------------
    # Likelihood
    # ------------------------------------------------------------------
    def _snrs_from_coefficients(self, name, coefficients):
        summary = self.summary_data[name]
        vector = np.concatenate([coefficients.real.ravel(), coefficients.imag.ravel()])
        d_inner_h = float(summary["data_vector"] @ vector)
        eigenvalues, eigenvectors = summary["matrix"]
        optimal_snr_squared = float(eigenvalues @ (eigenvectors @ vector) ** 2)
        matched_filter_snr = d_inner_h / optimal_snr_squared ** 0.5 if optimal_snr_squared > 0 else 0.0
        return self._CalculatedSNRs(
            d_inner_h=d_inner_h,
            optimal_snr_squared=optimal_snr_squared,
            complex_matched_filter_snr=matched_filter_snr,
            d_inner_h_array=None,
            optimal_snr_squared_array=None,
        )

    def _piecewise_linear_signal(self, name, coefficients):
        """ s = Re sum_q w_q u_q on all samples of the analysis window (zero outside the bins). """
        summary = self.summary_data[name]
        expanded = np.repeat(coefficients, summary["bin_sizes"], axis=2)
        signal = np.zeros(len(self._fiducial_times[name]))
        signal[summary["samples"]] = np.sum(
            ((expanded[:, 0] + expanded[:, 1] * summary["tau"]) * summary["fiducial"]).real, axis=0)
        return signal

    def calculate_snrs(self, waveform_polarizations, interferometer, *, return_array=True, parameters):
        """ :math:`d^T C^{-1} s`, :math:`s^T C^{-1} s` and the matched filter
        SNR of the heterodyned signal in one interferometer.

        Parameters
        ----------
        waveform_polarizations: dict
            Output of the source model (heterodyning_terms_at).
        interferometer: bilby.gw.detector.Interferometer
        return_array: bool
            Unused; kept for the :code:`GravitationalWaveTransient` interface.
        parameters: dict

        Returns
        -------
        calculated_snrs: _CalculatedSNRs, or None if the waveform failed
        """
        coefficients = self._heterodyning_coefficients(waveform_polarizations, [interferometer], parameters)
        if coefficients is None:
            return None
        return self._snrs_from_coefficients(interferometer.name, coefficients[0])

    def log_likelihood_ratio(self, parameters):
        parameters = copy.deepcopy(parameters)
        parameters.update(self.get_sky_frame_parameters(parameters))
        signal = self.waveform_generator.time_domain_strain(parameters)
        coefficients = None if signal is None else self._heterodyning_coefficients(
            signal, self.interferometers, parameters)
        if coefficients is None:
            return np.nan_to_num(-np.inf)
        total_snrs = self._CalculatedSNRs()
        for interferometer, interferometer_coefficients in zip(self.interferometers, coefficients):
            total_snrs += self._snrs_from_coefficients(interferometer.name, interferometer_coefficients)
        log_l = self.compute_log_likelihood_from_snrs(total_snrs, parameters=parameters)
        return float(np.real(log_l))

    def _compute_full_waveform(self, signal_polarizations, interferometer, parameters):
        """ The heterodyned (piecewise-linear) template on the samples of the
        analysis window; zero outside the bins. """
        coefficients = self._heterodyning_coefficients(signal_polarizations, [interferometer], parameters)
        if coefficients is None:
            return None
        return self._piecewise_linear_signal(interferometer.name, coefficients[0])

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

    # ------------------------------------------------------------------
    # Fiducial parameters (as in RelativeBinningGravitationalWaveTransient)
    # ------------------------------------------------------------------
    def find_maximum_likelihood_parameters(self, parameter_bounds, iterations=5, maximization_kwargs=None):
        if maximization_kwargs is None:
            maximization_kwargs = dict()
        parameters = copy.deepcopy(self.fiducial_parameters)
        updated_parameters_list = self.get_parameter_list_from_dictionary(self.fiducial_parameters)
        updated_parameters = copy.deepcopy(self.fiducial_parameters)
        old_fiducial_ln_likelihood = self.log_likelihood_ratio(self.fiducial_parameters)
        logger.info(f"Fiducial ln likelihood ratio: {old_fiducial_ln_likelihood:.2f}")
        for it in range(iterations):
            logger.info(f"Optimizing fiducial parameters. Iteration : {it + 1}")
            output = differential_evolution(
                self.lnlike_scipy_maximize,
                bounds=parameter_bounds,
                args=(parameters,),
                x0=updated_parameters_list,
                **maximization_kwargs,
            )
            updated_parameters_list = output['x']
            updated_parameters = copy.deepcopy(self.fiducial_parameters)
            updated_parameters.update(self.get_parameter_dictionary_from_list(updated_parameters_list))
            self.set_fiducial_waveforms(updated_parameters)
            self.setup_bins()
            self.compute_summary_data()
            new_fiducial_ln_likelihood = self.log_likelihood_ratio(updated_parameters)
            logger.info(f"Fiducial ln likelihood ratio: {new_fiducial_ln_likelihood:.2f}")
            if new_fiducial_ln_likelihood - old_fiducial_ln_likelihood < 0.1:
                break
            old_fiducial_ln_likelihood = new_fiducial_ln_likelihood

        logger.info("Fiducial waveforms updated")
        logger.info("Summary Data updated")
        return updated_parameters

    def _set_fiducial(self):
        """ Make the waveform generator generate the fiducial waveform (from
        fiducial_minimum_frequency), as in
        :code:`RelativeBinningGravitationalWaveTransient`. """
        self.waveform_generator.waveform_arguments["fiducial"] = 1
        self.waveform_generator.waveform_arguments["fiducial_minimum_frequency"] = self.fiducial_minimum_frequency
        self.waveform_generator._cache["parameters"] = None

    def _unset_fiducial(self):
        """ Make the waveform generator generate waveforms at sampled
        parameters (from sampled_minimum_frequency). """
        self.waveform_generator.waveform_arguments["fiducial"] = 0
        self.waveform_generator._cache["parameters"] = None

    lnlike_scipy_maximize = RelativeBinningGravitationalWaveTransient.lnlike_scipy_maximize
    get_parameter_dictionary_from_list = RelativeBinningGravitationalWaveTransient.get_parameter_dictionary_from_list
    get_parameter_list_from_dictionary = RelativeBinningGravitationalWaveTransient.get_parameter_list_from_dictionary
    get_bounds_from_priors = RelativeBinningGravitationalWaveTransient.get_bounds_from_priors

    @property
    def meta_data(self):
        meta_data = super(TimeDomainHeterodyningGravitationalWaveTransient, self).meta_data
        meta_data.update(
            fiducial_parameters=self.fiducial_parameters,
            number_of_bins=self.number_of_bins,
            mode_keys=self.mode_keys)
        return meta_data


_EIGENVALUE_TOLERANCE = 1e-12

_PHASE_PROXY_POWERS = np.array([5 / 8, 3 / 8, 1 / 4, -3 / 8])


def time_domain_bin_edges(times, epsilon=(0.5, 0.1), time_split=-0.02, spacing=(2, 10, 100),
                          spacing_times=(0.02, 0.1), chi=1):
    """ Bin edges for the time-domain heterodyned likelihood (Sharma et al.
    arXiv:2601.11239).

    - Before the merger (t < 0): edges equally spaced in the phase proxy
      :math:`\\psi(t) = \\sum_k {\\rm sign}(\\gamma_k) \\delta\\alpha_k |t|^{\\gamma_k}`,
      :math:`\\gamma = (5/8, 3/8, 1/4, -3/8)`, with
      :math:`\\delta\\alpha_k = 2 \\pi \\chi / |t|_{\\rm max}^{\\gamma_k}`
      (:math:`|t|_{\\rm min}` for :math:`\\gamma_k < 0`), one bin per
      epsilon[0] of :math:`|\\Delta\\psi|` for t <= time_split and per
      epsilon[1] after.
    - From the merger: one edge every spacing[0] samples until
      spacing_times[0], every spacing[1] samples until spacing_times[1] and
      every spacing[2] samples after.

    Parameters
    ==========
    times: array_like
        Increasing times (s) relative to the merger.

    Returns
    =======
    array_like: indices of the edges into times (first and last included).
    """
    edges = [0, len(times) - 1]
    inspiral = np.flatnonzero(times < 0)
    if len(inspiral) > 1:
        t = np.abs(times[inspiral])
        d_alpha = 2 * np.pi * chi / np.where(_PHASE_PROXY_POWERS < 0, t.min(), t.max()) ** _PHASE_PROXY_POWERS
        proxy = np.sum(np.sign(_PHASE_PROXY_POWERS) * d_alpha * t[:, None] ** _PHASE_PROXY_POWERS, axis=1)
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
