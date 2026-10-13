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

        h_+ - i h_\\times = \\sum_{\\ell m} \\left[ Z_{\\ell m}(t) h_{\\ell m}(t)
        + \\bar{Z}_{\\ell m}(t) h_{\\ell m}^*(t) \\right],

    with :math:`h_{\\ell m}` the modes with m >= 0 (co-precessing modes for
    IMRPhenomTP/TPHM), so that the signal in an interferometer is

    .. math::

        s(t_j) = \\Re \\sum_{\\ell m} \\tilde{r}_{\\ell m}(t_j)\\, h^0_{\\ell m}(t_j),\\qquad
        \\tilde{r}_{\\ell m} = \\frac{h_{\\ell m}}{h^0_{\\ell m}}
        \\left[ F Z_{\\ell m} + (F \\bar{Z}_{\\ell m})^* \\right],\\quad F = F_+ + i F_\\times,

    where :math:`h^0_{\\ell m}` are the modes of the fiducial waveform at the
    data samples. :math:`\\tilde{r}_{\\ell m}` is evaluated at the edges of time
    bins and taken to be linear in each bin b,
    :math:`\\tilde{r}_{\\ell m} = \\tilde{r}_{\\ell m,0}(b) + \\tilde{r}_{\\ell m,1}(b) (t - t_b)`.
    With :math:`v = (\\Re \\tilde{r}, \\Im \\tilde{r})`, :math:`s = V v`, where the
    columns of V are :math:`\\Re(h^0_{\\ell m} (t - t_b)^p)` and
    :math:`-\\Im(h^0_{\\ell m} (t - t_b)^p)` in each bin, so that

    .. math::

        \\langle d, s \\rangle = a \\cdot v,\\quad a = V^T C^{-1} d,\\qquad
        \\langle s, s \\rangle = v^T S v,\\quad S = V^T C^{-1} V,

    without the terms quadratic in the slopes :math:`\\tilde{r}_{\\ell m,1}` (as in
    Sharma et al.); a contains :math:`A_0, A_1` and S the B matrices of
    Sharma et al., for all modes.

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
    epsilon: tuple, optional
        Bin size in the inspiral: phase-proxy spacing before and after
        :code:`time_split` (see :code:`time_domain_bin_edges`). Smaller
        values give more bins.
    time_split: float, optional
        Time (s, relative to the merger) at which epsilon changes.
    spacing: tuple, optional
        Number of samples per bin after the merger, until
        :code:`spacing_times[0]`, until :code:`spacing_times[1]` and after.
    spacing_times: tuple, optional
        Times (s, relative to the merger) at which the spacing changes.
    chi: float, optional
        Tunable parameter of the phase proxy (as in relative binning).
    sampled_minimum_frequency: float, optional
        Frequency (Hz) at which the waveforms at sampled parameters start. By
        default 2 Hz below the lowest minimum_frequency of the interferometers;
        a higher value gives a warning.
    fiducial_minimum_frequency: float, optional
        Frequency (Hz) at which the fiducial waveform starts. By default the
        lower of :code:`sampled_minimum_frequency` and the frequency derived
        from the fiducial chirp mass and the start of the analysis window
        (the leading-order frequency 1.5 x (time from the window start to the
        merger) + 0.1 s before the merger, minus 1 Hz); a given value must not
        be above that.
    single_precision: bool, optional
        If True, the eigenvectors of S are stored and multiplied in float32
        instead of float64: :math:`v^T S v` is a few times faster for many modes
        and bins, with relative errors of ~1e-7 in it. Default is False.
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
                 single_precision=False, time_marginalization=False,
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
        self.single_precision = single_precision
        # this likelihood sets its own entries in waveform_arguments; copy the dictionary so that
        # they do not reach other waveform generators that share it
        self.waveform_generator.waveform_arguments = dict(self.waveform_generator.waveform_arguments)
        self._set_sampled_minimum_frequency(sampled_minimum_frequency)
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

    def _set_sampled_minimum_frequency(self, sampled_minimum_frequency):
        maximum = min(ifo.minimum_frequency for ifo in self.interferometers) - 2
        if sampled_minimum_frequency is None:
            sampled_minimum_frequency = maximum
        if sampled_minimum_frequency > maximum:
            logger.warning(
                f"sampled_minimum_frequency ({sampled_minimum_frequency:g} Hz) is above {maximum:g} Hz "
                "(2 Hz below the interferometers' minimum_frequency)")
        self.sampled_minimum_frequency = sampled_minimum_frequency
        self.waveform_generator.waveform_arguments["sampled_minimum_frequency"] = sampled_minimum_frequency

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

    def _modes_and_harmonics_at_combined_bin_edges(self, signal, model_times):
        """ Modes and harmonic factors at `model_times`, a list with the model
        times of each interferometer (of the bin edges; of all samples for the
        fiducial waveform): the model times of all interferometers are joined,
        the waveform is evaluated once, and the result is split back per
        interferometer. None if the waveform failed. """
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
        """ The coefficients r_tilde[lm, p, b] of each interferometer (see the class
        docstring): r_tilde_lm = h_lm (F Z_lm + (F Zbar_lm)^*) / h^0_lm at the bin edges,
        then r_tilde_lm,0 (p = 0) and r_tilde_lm,1 (p = 1) in each bin b. None if the
        waveform failed. """
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

    # ------------------------------------------------------------------
    # Fiducial waveform, bins and summary data
    # ------------------------------------------------------------------
    def set_fiducial_waveforms(self, parameters):
        """ The modes of the fiducial waveform at the data samples of each
        interferometer, starting before the analysis window (see
        :code:`fiducial_minimum_frequency`) and held at their last non-zero
        values after the ringdown. """
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
        derived = min(max(frequency - 1, 1.0), self.sampled_minimum_frequency)
        requested = self.requested_fiducial_minimum_frequency
        if requested is not None and requested > derived:
            raise ValueError(
                f"fiducial_minimum_frequency ({requested:g} Hz) must not be above {derived:.2f} Hz "
                "(the lower of sampled_minimum_frequency and the frequency derived from the fiducial chirp mass "
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
        """ Summary data of each interferometer (see the class docstring):

        - the columns of V: in each bin b, :math:`\\Re(h^0_{\\ell m} (t - t_b)^p)` and
          :math:`-\\Im(h^0_{\\ell m} (t - t_b)^p)` for every mode and p = 0, 1;
        - :math:`a = V^T C^{-1} d` (contains :math:`A_0, A_1`);
        - :math:`S = V^T C^{-1} V` (contains the B matrices), without the terms with
          p = p' = 1, stored as :math:`S = \\sum_n \\lambda_n e_n e_n^T` so that
          :math:`v^T S v = \\sum_n \\lambda_n (e_n \\cdot v)^2`.

        Bin b holds the samples from its initial edge up to its final edge (the
        last bin also holds the last sample). """
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
                    eigenvectors[:, keep].T, dtype=np.float32 if self.single_precision else np.float64),
                bin_of_sample=bin_of_sample, time_from_bin_centre=time_from_bin_centre)
            logger.info(f"{name}: {len(self.mode_keys)} modes, {number_of_bins} bins, "
                        f"S of size {size} and rank {np.sum(keep)}")

    # ------------------------------------------------------------------
    # Likelihood
    # ------------------------------------------------------------------
    def calculate_snrs(self, waveform_polarizations, interferometer, *, return_array=True, parameters,
                       r_tilde=None):
        """ :math:`\\langle d, s \\rangle = a \\cdot v` and
        :math:`\\langle s, s \\rangle = v^T S v = \\sum_n \\lambda_n (e_n \\cdot v)^2` in one
        interferometer, with :math:`v = (\\Re \\tilde{r}, \\Im \\tilde{r})`.

        Parameters
        ----------
        waveform_polarizations: dict
            Output of the source model (heterodyning_terms_at).
        interferometer: bilby.gw.detector.Interferometer
        return_array: bool
            Unused; kept for the :code:`GravitationalWaveTransient` interface.
        parameters: dict
        r_tilde: array_like, optional
            The coefficients r_tilde[lm, p, b] of this interferometer, if already
            computed (as in :code:`log_likelihood_ratio`, for all interferometers
            from one waveform call); otherwise they are computed here.

        Returns
        -------
        calculated_snrs: _CalculatedSNRs, or None if the waveform failed
        """
        if r_tilde is None:
            r_tilde = self._r_tilde_coefficients(waveform_polarizations, [interferometer], parameters)
            if r_tilde is None:
                return None
            r_tilde = r_tilde[0]
        summary = self.summary_data[interferometer.name]
        v = np.concatenate([r_tilde.real.ravel(), r_tilde.imag.ravel()])
        d_inner_s = float(summary["a"] @ v)
        # e_n . v in the precision of the stored eigenvectors (see single_precision)
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
        """ The heterodyned template on all samples of the analysis window (not used in the
        likelihood): s(t_j) = Re sum_lm [r_tilde_lm,0(b) + r_tilde_lm,1(b) (t_j - t_b)] h^0_lm(t_j). """
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

    # ------------------------------------------------------------------
    # Fiducial parameters (as in RelativeBinningGravitationalWaveTransient)
    # ------------------------------------------------------------------
    def find_maximum_likelihood_parameters(self, parameter_bounds, maximization_kwargs=None):
        """ Maximize the exact time-domain likelihood
        (:code:`TimeDomainGravitationalWaveTransient` with placement="model_time",
        with the same distance marginalization) over :code:`parameters_to_be_updated`
        with scipy differential evolution, starting from the fiducial parameters. """
        from ..source import lal_binary_black_hole_model_time
        from .time_domain import TimeDomainGravitationalWaveTransient

        if maximization_kwargs is None:
            maximization_kwargs = dict()
        waveform_arguments = {
            key: value for key, value in self.waveform_generator.waveform_arguments.items()
            if key not in ("fiducial", "fiducial_minimum_frequency", "sampled_minimum_frequency")}
        exact_likelihood = TimeDomainGravitationalWaveTransient(
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
