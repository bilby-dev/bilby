from itertools import combinations_with_replacement

import numpy as np
import scipy.interpolate
import scipy.sparse

from .relative import RelativeBinningGravitationalWaveTransient
from ...core import utils
from ..conversion import (
    bilby_to_lalsimulation_spins,
    chirp_mass_and_mass_ratio_to_component_masses,
    component_masses_to_chirp_mass,
    component_masses_to_symmetric_mass_ratio,
    symmetric_mass_ratio_to_mass_ratio,
)

_SAFE_KEYS = [
    "mass_1", "mass_2", "a_1", "a_2", "tilt_1", "tilt_2", "phi_12", "phi_jl",
    "theta_jn", "phase", "luminosity_distance", "ra", "dec", "psi", "geocent_time",
]


def get_safe_fiducial_parameters(parameters, waveform_generator, eta_max=0.24):
    """
    Return fiducial parameters that avoid zeros in single harmonics of the
    fiducial waveform, following :code:`ReferenceWaveformFinder._get_safe_par_dic`
    of cogwheel (arXiv:2105.04587).

    The in-plane spin components at the reference frequency are set to zero,
    the inclination of the orbital angular momentum is set to 1 or pi - 1,
    whichever is on the same side of pi / 2, and the symmetric mass ratio is
    capped at :code:`eta_max` at constant chirp mass.

    Parameters
    ----------
    parameters: dict
        Fiducial parameters, in any basis that the parameter conversion of
        :code:`waveform_generator` accepts.
    waveform_generator: bilby.gw.waveform_generator.WaveformGenerator
        Provides the parameter conversion and the reference frequency.
    eta_max: float, optional
        Largest symmetric mass ratio of the returned parameters.

    Returns
    -------
    safe_parameters: dict
        The parameters in the basis :code:`mass_1`, :code:`mass_2`, :code:`a_1`,
        :code:`a_2`, :code:`tilt_1`, :code:`tilt_2`, :code:`phi_12`,
        :code:`phi_jl`, :code:`theta_jn`, :code:`phase`,
        :code:`luminosity_distance`, :code:`ra`, :code:`dec`, :code:`psi`,
        :code:`geocent_time`, together with any calibration parameters and
        :code:`time_jitter` of the input.
    """
    converted, _ = waveform_generator.parameter_conversion(parameters.copy())
    reference_frequency = waveform_generator.waveform_arguments["reference_frequency"]

    def lalsimulation_spins(values):
        return bilby_to_lalsimulation_spins(
            theta_jn=values["theta_jn"], phi_jl=values["phi_jl"], tilt_1=values["tilt_1"],
            tilt_2=values["tilt_2"], phi_12=values["phi_12"], a_1=values["a_1"], a_2=values["a_2"],
            mass_1=values["mass_1"] * utils.solar_mass, mass_2=values["mass_2"] * utils.solar_mass,
            reference_frequency=reference_frequency, phase=values["phase"])

    safe = {key: converted[key] for key in _SAFE_KEYS}
    iota, _, _, spin_1z, _, _, spin_2z = lalsimulation_spins(safe)
    target_iota = 1.0 if iota < np.pi / 2 else np.pi - 1.0

    if component_masses_to_symmetric_mass_ratio(safe["mass_1"], safe["mass_2"]) > eta_max:
        chirp_mass = component_masses_to_chirp_mass(safe["mass_1"], safe["mass_2"])
        safe["mass_1"], safe["mass_2"] = chirp_mass_and_mass_ratio_to_component_masses(
            chirp_mass, symmetric_mass_ratio_to_mass_ratio(eta_max))

    safe.update(
        a_1=abs(spin_1z), a_2=abs(spin_2z),
        tilt_1=0.0 if spin_1z >= 0 else np.pi, tilt_2=0.0 if spin_2z >= 0 else np.pi,
        phi_12=0.0, phi_jl=0.0, theta_jn=target_iota)
    # J and L are antiparallel if the aligned spins outweigh L; theta_jn is then pi - iota
    if not np.isclose(lalsimulation_spins(safe)[0], target_iota):
        safe["theta_jn"] = np.pi - target_iota

    for key in parameters:
        if key.startswith("recalib_") or key == "time_jitter":
            safe[key] = parameters[key]
    return safe


class ModeByModeRelativeBinningGravitationalWaveTransient(RelativeBinningGravitationalWaveTransient):
    """A gravitational-wave transient likelihood object which uses relative
    binning separately for each azimuthal index m of the co-precessing modes
    and for each polarization, following Leslie, Dai and Pratten
    arXiv:2109.09872 and the implementation in cogwheel (arXiv:2105.04587).

    For each detector, harmonic m and polarization p, the ratio of the
    waveform :math:`h_{m,p}` to the detector response :math:`h_{0,m}` of
    harmonic m of the fiducial waveform is interpolated between the bin edges
    with a spline of degree :code:`spline_degree`. The detector antenna
    pattern multiplies :math:`h_{m,p}` after the interpolation.

    Two steps keep :math:`h_{0,m}` away from zero at the bin edges. The
    fiducial parameters are first made safe with
    :func:`get_safe_fiducial_parameters`. Above the bin edge where the
    cumulative squared SNR of :math:`h_{0,m}` reaches 99%, :math:`h_{0,m}` is
    faded to a constant over :code:`spline_degree` bins. The bins span the full
    frequency band of the interferometers.

    The waveform generator must use
    :func:`bilby.gw.source.lal_binary_black_hole_relative_binning_by_mode`, or
    another source model that returns arrays of shape (number of harmonics,
    number of frequencies) for each polarization.

    Parameters
    ----------
    interferometers: list, bilby.gw.detector.InterferometerList
        A list of `bilby.detector.Interferometer` instances - contains the
        detector data and power spectral densities
    waveform_generator: `bilby.waveform_generator.WaveformGenerator`
        An object which computes the frequency-domain strain of the signal,
        given some set of parameters
    fiducial_parameters: dict, optional
        A starting guess for initial parameters of the event for finding the
        maximum likelihood (fiducial) waveform.
        If no fiducial parameters are passed, a random point will be drawn from
        the prior.
    parameter_bounds: dict, optional
        Dictionary of bounds (lists) for the initial parameters when finding
        the initial maximum likelihood (fiducial) waveform.
    maximization_kwargs: dict, optional
        Keyword arguments passed to :code:`scipy.optimize.differential_evolution`.
    update_fiducial_parameters: bool
        Whether to try to update the fiducial parameters by finding the maximum
        likelihood point using scipy differential evolution. Default is
        :code:`False`.
    distance_marginalization: bool, optional
        If true, marginalize over distance in the likelihood.
    time_marginalization: bool, optional
        Not supported, must be :code:`False`.
    phase_marginalization: bool, optional
        If true, marginalize over phase in the likelihood.
    priors: dict, optional
        If given, used in the distance and phase marginalization.
    distance_marginalization_lookup_table: (dict, str), optional
        See :code:`RelativeBinningGravitationalWaveTransient`.
    jitter_time: bool, optional
        See :code:`RelativeBinningGravitationalWaveTransient`.
    reference_frame: (str, bilby.gw.detector.InterferometerList, list), optional
        Definition of the reference frame for the sky location.
    time_reference: str, optional
        Name of the reference for the sampled time parameter.
    chi: float, optional
        Tunable parameter which limits the perturbation of alpha when setting
        up the bin range. See https://arxiv.org/abs/1806.08792.
    epsilon: float, optional
        Tunable parameter which limits the differential phase change in each
        bin when setting up the bin range. See https://arxiv.org/abs/1806.08792.
    spline_degree: int, optional
        Degree of the spline that interpolates the waveform ratios between
        bin edges, 1 to 5. Default is 3.
    safe_fiducial_parameters: bool, optional
        Whether to replace the fiducial parameters with
        :func:`get_safe_fiducial_parameters` of them. Default is :code:`True`.

    Returns
    -------
    Likelihood: `bilby.core.likelihood.Likelihood`
        A likelihood object, able to compute the likelihood of the data given
        some model parameters.

    Notes
    -----
    Time marginalization and calibration marginalization are not supported.
    """

    def __init__(self, interferometers,
                 waveform_generator,
                 fiducial_parameters=None,
                 parameter_bounds=None,
                 maximization_kwargs=None,
                 update_fiducial_parameters=False,
                 distance_marginalization=False,
                 time_marginalization=False,
                 phase_marginalization=False,
                 priors=None,
                 distance_marginalization_lookup_table=None,
                 jitter_time=True,
                 reference_frame="sky",
                 time_reference="geocenter",
                 chi=1,
                 epsilon=0.5,
                 spline_degree=3,
                 safe_fiducial_parameters=True):
        if time_marginalization:
            raise NotImplementedError(
                f"{self.__class__.__name__} does not support time marginalization.")
        self.spline_degree = spline_degree
        self.safe_fiducial_parameters = safe_fiducial_parameters
        self.fiducial_waveform_parameters = None
        self.per_detector_fiducial_mode_waveforms = dict()
        super().__init__(
            interferometers=interferometers,
            waveform_generator=waveform_generator,
            fiducial_parameters=fiducial_parameters,
            parameter_bounds=parameter_bounds,
            maximization_kwargs=maximization_kwargs,
            update_fiducial_parameters=update_fiducial_parameters,
            distance_marginalization=distance_marginalization,
            time_marginalization=time_marginalization,
            phase_marginalization=phase_marginalization,
            priors=priors,
            distance_marginalization_lookup_table=distance_marginalization_lookup_table,
            jitter_time=jitter_time,
            reference_frame=reference_frame,
            time_reference=time_reference,
            chi=chi,
            epsilon=epsilon)

    def set_fiducial_waveforms(self, parameters):
        """
        Compute the detector response of each harmonic of the fiducial
        waveform on the full frequency grid.

        If :code:`safe_fiducial_parameters` is set, the waveform is computed
        at :func:`get_safe_fiducial_parameters` of :code:`parameters`. The
        bins are not limited by where the fiducial waveform ends.

        Parameters
        ----------
        parameters: dict
            The fiducial parameters.
        """
        parameters = parameters.copy()
        parameters.update(self.get_sky_frame_parameters(parameters=parameters))
        if self.safe_fiducial_parameters:
            parameters = get_safe_fiducial_parameters(parameters, self.waveform_generator)
        self._set_fiducial()
        self.fiducial_polarizations = self.waveform_generator.frequency_domain_strain(parameters)
        self._unset_fiducial()
        if self.fiducial_polarizations is None:
            raise ValueError(f"Cannot compute fiducial waveforms for {parameters}")
        self.fiducial_waveform_parameters = parameters

        for interferometer in self.interferometers:
            waveforms = interferometer.get_detector_response(self.fiducial_polarizations, parameters)
            self.per_detector_fiducial_mode_waveforms[interferometer.name] = waveforms
            self.per_detector_fiducial_waveforms[interferometer.name] = waveforms.sum(axis=0)

    def compute_summary_data(self):
        r"""
        Compute the summary data of each interferometer: for each harmonic m,
        the weights :math:`D_{m,j}` of :math:`\langle h|d \rangle`, and for
        each pair of harmonics m <= m', the weights :math:`H_{mm',j}` of
        :math:`\langle h|h \rangle`, at bin edge j.

        With :math:`S_j(f)` the spline that is one at bin edge j and zero at
        the others,

        .. math::

            D_{m,j} = \frac{4}{T} \sum_f S_j(f) \frac{h_{0,m}^*(f) d(f)}{P(f)}
            \Big/ h_{0,m}^*(f_j)

            H_{mm',j} = \frac{4}{T} \sum_f S_j(f)
            \frac{h_{0,m}(f) h_{0,m'}^*(f)}{P(f)}
            \Big/ \left(h_{0,m}(f_j) h_{0,m'}^*(f_j)\right)

        with :math:`H_{mm',j}` doubled for :math:`m \neq m'`. Before this,
        :math:`h_{0,m}` is faded to a constant above the bin where its
        cumulative squared SNR reaches 99%.
        """
        names = [interferometer.name for interferometer in self.interferometers]
        number_of_harmonics = len(self.per_detector_fiducial_mode_waveforms[names[0]])
        self.harmonic_pairs = tuple(
            np.array(indices) for indices in zip(*combinations_with_replacement(range(number_of_harmonics), 2)))
        m_inds, mprime_inds = self.harmonic_pairs
        inverse_psds = dict()
        for interferometer in self.interferometers:
            inverse_psds[interferometer.name] = np.zeros(len(interferometer.frequency_array))
            inverse_psds[interferometer.name][interferometer.frequency_mask] = (
                1 / interferometer.power_spectral_density_array[interferometer.frequency_mask])
        network_inverse_psd = sum(inverse_psds.values())

        frequency_array = self.interferometers[0].frequency_array
        self._set_splines(frequency_array)
        summary_data = dict()
        for interferometer in self.interferometers:
            name = interferometer.name

            # Remove the overall time shift so that the fiducial waveform is smooth in the ringdown
            time_shift = np.exp(-2j * np.pi * frequency_array * (
                self.fiducial_waveform_parameters["geocent_time"] - interferometer.strain_data.start_time))
            h0 = self.per_detector_fiducial_mode_waveforms[name] / time_shift
            h0_bins = h0[:, self.bin_inds]
            for harmonic in range(number_of_harmonics):
                self._fade_ringdown(h0[harmonic], h0_bins[harmonic], frequency_array, network_inverse_psd)
            h0 *= time_shift
            h0_bins *= time_shift[self.bin_inds]
            if np.any(h0_bins == 0):
                raise ValueError(
                    "A harmonic of the fiducial waveform is zero at a bin edge; "
                    "use safe_fiducial_parameters=True or other fiducial parameters.")

            inverse_psd = inverse_psds[name]
            strain = interferometer.frequency_domain_strain
            d_inner_h_weights = (
                self._get_summary_weights(h0.conj() * strain * inverse_psd, interferometer.duration)
                / h0_bins.conj())
            h_inner_h_weights = (
                self._get_summary_weights(h0[m_inds] * h0[mprime_inds].conj() * inverse_psd, interferometer.duration)
                / (h0_bins[m_inds] * h0_bins[mprime_inds].conj()))
            h_inner_h_weights[m_inds != mprime_inds] *= 2
            summary_data[name] = (d_inner_h_weights, h_inner_h_weights)

        self.summary_data = summary_data

    def _set_splines(self, frequency_array):
        """
        Set :code:`_spline_coefficients`, whose row j holds the B-spline
        coefficients of the spline of degree :code:`spline_degree` that is one
        at bin edge j and zero at the other edges, and :code:`_basis_splines`,
        a sparse array whose rows are the B-spline basis elements on
        :code:`frequency_array`.

        Parameters
        ----------
        frequency_array: array_like
            The frequencies at which the basis elements are evaluated.
        """
        number_of_edges = len(self.bin_freqs)
        coefficients = np.empty((number_of_edges, number_of_edges))
        for index, values in enumerate(np.eye(number_of_edges)):
            knots, spline_coefficients, _ = scipy.interpolate.splrep(
                self.bin_freqs, values, s=0, k=self.spline_degree)
            coefficients[index] = spline_coefficients[:number_of_edges]
        self._spline_coefficients = coefficients

        rows, columns, values = list(), list(), list()
        for index in range(number_of_edges):
            element_knots = knots[index:index + self.spline_degree + 2]
            element = scipy.interpolate.BSpline.basis_element(element_knots)
            start = np.searchsorted(frequency_array, element_knots[0], side="left")
            stop = np.searchsorted(frequency_array, element_knots[-1], side="right")
            rows.append(np.full(stop - start, index))
            columns.append(np.arange(start, stop))
            # A basis element vanishes at its last knot; evaluate just below it so the last element is one there
            values.append(element(np.minimum(frequency_array[start:stop], np.nextafter(element_knots[-1], 0))))
        self._basis_splines = scipy.sparse.csr_matrix(
            (np.concatenate(values), (np.concatenate(rows), np.concatenate(columns))),
            shape=(number_of_edges, len(frequency_array)))

    def _get_summary_weights(self, integrand, duration):
        r"""
        Return the weights :math:`W_j = (4 / T) \sum_f g(f) S_j(f)`, so that
        :math:`(4 / T) \sum_f g(f) r(f) \approx \sum_j W_j r(f_j)` for a
        smooth function r, exactly if r is a spline of degree
        :code:`spline_degree` with knots at the bin edges.

        Parameters
        ----------
        integrand: array_like
            :math:`g(f)`, of shape (..., number of frequencies).
        duration: float
            The duration T of the data.

        Returns
        -------
        weights: array_like
            Of shape (..., number of bin edges).
        """
        projected = (self._basis_splines @ np.atleast_2d(integrand).T).T
        weights = 4 / duration * projected @ self._spline_coefficients.T
        return weights.reshape(integrand.shape[:-1] + (len(self.bin_freqs),))

    def _fade_ringdown(self, h0, h0_bins, frequency_array, inverse_psd):
        r"""
        Replace one harmonic of the fiducial waveform by a constant at high
        frequency, in place, as cogwheel does.

        The cutoff is the first bin edge at or above the frequency where the
        cumulative :math:`|h_0|^2 / P` reaches 99% of its total, with
        :math:`1 / P` summed over the interferometers. Between the edge
        :code:`spline_degree` bins below the cutoff and the cutoff, the
        waveform is multiplied by a :math:`\cos^2` taper and the remainder is
        filled with its value at the cutoff. If that value is zero, the cutoff
        moves down to the last bin edge where the waveform is nonzero.

        Parameters
        ----------
        h0: array_like
            The harmonic on the full frequency grid, without the overall time
            shift.
        h0_bins: array_like
            The harmonic at the bin edges, without the overall time shift.
        frequency_array: array_like
            The full frequency grid.
        inverse_psd: array_like
            :math:`1 / P` summed over the interferometers, zero outside the
            frequency masks.
        """
        cumulative_snr_squared = np.cumsum(np.abs(h0) ** 2 * inverse_psd)
        if cumulative_snr_squared[-1] == 0:
            return
        cumulative_snr_squared /= cumulative_snr_squared[-1]
        frequency_99 = frequency_array[np.searchsorted(cumulative_snr_squared, 0.99)]
        end_index = min(np.searchsorted(self.bin_freqs, frequency_99), len(self.bin_freqs) - 1)
        while end_index > 0 and h0_bins[end_index] == 0:
            end_index -= 1
        start_frequency = self.bin_freqs[max(end_index - self.spline_degree, 0)]
        end_frequency = self.bin_freqs[end_index]
        constant = h0_bins[end_index]

        def fade(frequencies):
            if end_frequency == start_frequency:
                return (frequencies <= end_frequency).astype(float)
            return np.cos(np.clip((frequencies - start_frequency) / (end_frequency - start_frequency), 0, 1)
                          * np.pi / 2) ** 2

        fade_full = fade(frequency_array)
        fade_bins = fade(self.bin_freqs)
        h0[:] = h0 * fade_full + constant * (1 - fade_full)
        h0_bins[:] = h0_bins * fade_bins + constant * (1 - fade_bins)

    def _waveform_at_bins_per_polarization(self, waveform_polarizations, interferometer, parameters):
        """
        Apply the time shift to the interferometer and its calibration to
        each harmonic and polarization at the bin edges, without the antenna
        pattern.

        Parameters
        ----------
        waveform_polarizations: dict
            Plus and cross polarizations, each of shape (number of harmonics,
            number of bin edges).
        interferometer: bilby.gw.detector.Interferometer
            The interferometer.
        parameters: dict
            Parameters of the signal.

        Returns
        -------
        waveform_per_polarization: array_like
            Of shape (2, number of harmonics, number of bin edges), plus first.
        """
        time_shift = interferometer.time_delay_from_geocenter(
            parameters["ra"], parameters["dec"], parameters["geocent_time"])
        dt = parameters["geocent_time"] - interferometer.strain_data.start_time + time_shift
        response = np.exp(-2j * np.pi * dt * self.bin_freqs)
        response = response * interferometer.calibration_model.get_calibration_factor(
            self.bin_freqs, prefix=f"recalib_{interferometer.name}_", **parameters)
        return np.array([waveform_polarizations["plus"], waveform_polarizations["cross"]]) * response

    def calculate_snrs(self, waveform_polarizations, interferometer, *, return_array=True, parameters):
        r"""
        Compute the inner products of the waveform with the data and with
        itself from the summary data,

        .. math::

            \langle h|d \rangle = \sum_{m,p} F_p \sum_j D_{m,j} h_{m,p}^*(f_j)

            \langle h|h \rangle = \mathrm{Re} \sum_{m \leq m'} \sum_{p,p'}
            F_p F_{p'} \sum_j H_{mm',j} h_{m,p}(f_j) h_{m',p'}^*(f_j)

        with :math:`F_p` the antenna pattern of polarization p and
        :math:`h_{m,p}` harmonic m of polarization p, time shifted to the
        interferometer.

        Parameters
        ----------
        waveform_polarizations: dict
            Plus and cross polarizations at the bin edges, each of shape
            (number of harmonics, number of bin edges).
        interferometer: bilby.gw.detector.Interferometer
            The interferometer.
        return_array: bool
            Unused, as time marginalization is not supported.
        parameters: dict
            Parameters of the signal.

        Returns
        -------
        calculated_snrs: _CalculatedSNRs
            The inner products, with :code:`d_inner_h_array` set to None.
        """
        waveform_per_polarization = self._waveform_at_bins_per_polarization(
            waveform_polarizations, interferometer, parameters)
        if interferometer.reference_time is None:
            antenna_time = parameters["geocent_time"]
        else:
            antenna_time = interferometer.reference_time
        antenna_pattern = np.array([
            interferometer.antenna_response(
                parameters["ra"], parameters["dec"], antenna_time, parameters["psi"], mode)
            for mode in ["plus", "cross"]])
        d_inner_h_weights, h_inner_h_weights = self.summary_data[interferometer.name]
        m_inds, mprime_inds = self.harmonic_pairs

        d_inner_h = np.einsum("p,mj,pmj->", antenna_pattern, d_inner_h_weights, waveform_per_polarization.conj())
        strain = np.einsum("p,pmj->pmj", antenna_pattern, waveform_per_polarization)
        h_inner_h = np.einsum(
            "kj,pkj,qkj->", h_inner_h_weights, strain[:, m_inds], strain[:, mprime_inds].conj()).real
        optimal_snr_squared = h_inner_h
        complex_matched_filter_snr = d_inner_h / (optimal_snr_squared ** 0.5)

        return self._CalculatedSNRs(
            d_inner_h=d_inner_h,
            optimal_snr_squared=optimal_snr_squared,
            complex_matched_filter_snr=complex_matched_filter_snr,
            d_inner_h_array=None
        )

    def _compute_full_waveform(self, signal_polarizations, interferometer, parameters):
        raise NotImplementedError(
            f"{self.__class__.__name__} does not compute the full waveform.")
