from copy import deepcopy

import numpy as np
import scipy.interpolate
import scipy.sparse
from scipy.optimize import differential_evolution

from .base import GravitationalWaveTransient
from ...compat.utils import array_module
from ...core.utils import logger
from ...core.prior.base import Constraint
from ...core.prior import DeltaFunction


class RelativeBinningGravitationalWaveTransient(GravitationalWaveTransient):
    """A gravitational-wave transient likelihood object which uses the relative
    binning procedure to calculate a fast likelihood. See Zackay et al.
    arXiv1806.08792

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
        maximum likelihood (fiducial) waveform. These should be specified in
        the same parameter basis as the one that sampling is carried out in.
        For example, if sampling in `mass_1` and `mass_2`, the fiducial
        parameters should also be provided in `mass_1` and `mass_2.`
        If no fiducial parameters are passed, a random point will be drawn from
        the prior.
    update_fiducial_parameters: bool
        Whether to try to update the fiducial parameters by finding the maximum
        likelihood point using scipy differential evolution. Default is
        :code:`False`.
    parameter_bounds: dict, optional
        Dictionary of bounds (lists) for the initial parameters when finding
        the initial maximum likelihood (fiducial) waveform.
    distance_marginalization: bool, optional
        If true, marginalize over distance in the likelihood.
        This uses a look up table calculated at run time.
        The distance prior is set to be a delta function at the minimum
        distance allowed in the prior being marginalised over.
    time_marginalization: bool, optional
        If true, marginalize over time in the likelihood.
        This uses a FFT to calculate the likelihood over a regularly spaced
        grid.
        In order to cover the whole space the prior is set to be uniform over
        the spacing of the array of times.
        If using time marginalisation and jitter_time is True a "jitter"
        parameter is added to the prior which modifies the position of the
        grid of times.
    phase_marginalization: bool, optional
        If true, marginalize over phase in the likelihood.
        This is done analytically using a Bessel function.
        The phase prior is set to be a delta function at phase=0.
    priors: dict, optional
        If given, used in the distance and phase marginalization.
    distance_marginalization_lookup_table: (dict, str), optional
        If a dict, dictionary containing the lookup_table, distance_array,
        (distance) prior_array, and reference_distance used to construct
        the table.
        If a string the name of a file containing these quantities.
        The lookup table is stored after construction in either the
        provided string or a default location:
        '.distance_marginalization_lookup_dmin{}_dmax{}_n{}.npz'
    jitter_time: bool, optional
        Whether to introduce a `time_jitter` parameter. This avoids either
        missing the likelihood peak, or introducing biases in the
        reconstructed time posterior due to an insufficient sampling frequency.
        Default is False, however using this parameter is strongly encouraged.
    reference_frame: (str, bilby.gw.detector.InterferometerList, list), optional
        Definition of the reference frame for the sky location.

        - "sky": sample in RA/dec, this is the default
        - e.g., "H1L1", ["H1", "L1"], InterferometerList(["H1", "L1"]):
          sample in azimuth and zenith, `azimuth` and `zenith` defined in the
          frame where the z-axis is aligned with the vector connecting H1
          and L1.

    time_reference: str, optional
        Name of the reference for the sampled time parameter.

        - "geocent"/"geocenter": sample in the time at the Earth's center,
          this is the default
        - e.g., "H1": sample in the time of arrival at H1

    chi: float, optional
        Tunable parameter which limits the perturbation of alpha when setting
        up the bin range. See https://arxiv.org/abs/1806.08792.
    epsilon: float, optional
        Tunable parameter which limits the differential phase change in each
        bin when setting up the bin range. See https://arxiv.org/abs/1806.08792.
    spline_degree: int, optional
        Degree of the B-spline that interpolates the ratio of the waveform to
        the fiducial waveform between the bin edges, from 1 to 5. Degree 1 is
        linear interpolation within each bin. Default is 3.
    fade_fiducial_waveform: bool, optional
        Whether to replace the fiducial waveform used for the summary data by
        a constant above the frequency that contains 99.99% of its squared
        signal-to-noise ratio, see :meth:`_get_summary_fiducial_waveforms`.
        This avoids dividing by a fiducial waveform that is close to zero at
        the end of its support, e.g., in the taper of the NRTidal models.
        Default is :code:`True`.

    Returns
    -------
    Likelihood: `bilby.core.likelihood.Likelihood`
        A likelihood object, able to compute the likelihood of the data given
        some model parameters.

    Notes
    -----
    The relative binning likelihood does not currently support calibration marginalization.
    """

    def __init__(
        self,
        interferometers,
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
        fade_fiducial_waveform=True,
    ):
        super(RelativeBinningGravitationalWaveTransient, self).__init__(
            interferometers=interferometers,
            waveform_generator=waveform_generator,
            distance_marginalization=distance_marginalization,
            phase_marginalization=phase_marginalization,
            time_marginalization=time_marginalization,
            priors=priors,
            distance_marginalization_lookup_table=distance_marginalization_lookup_table,
            jitter_time=jitter_time,
            reference_frame=reference_frame,
            time_reference=time_reference,
        )

        if fiducial_parameters is None:
            logger.info("Drawing fiducial parameters from prior.")
            fiducial_parameters = priors.sample()
        self.fiducial_parameters = fiducial_parameters.copy()
        self._unset_fiducial()
        if self.time_marginalization:
            self.fiducial_parameters["geocent_time"] = interferometers.start_time
        if self.distance_marginalization:
            self.fiducial_parameters["luminosity_distance"] = self._ref_dist
        if self.phase_marginalization:
            self.fiducial_parameters["phase"] = 0.0
        self.chi = chi
        self.epsilon = epsilon
        if spline_degree not in (1, 2, 3, 4, 5):
            raise ValueError(
                f"spline_degree must be 1, 2, 3, 4 or 5, not {spline_degree}."
            )
        self.spline_degree = spline_degree
        self.fade_fiducial_waveform = fade_fiducial_waveform
        self._fiducial_waveform_parameters = None
        self._summary_fiducial_waveforms = dict()
        self.gamma = np.array([-5 / 3, -2 / 3, 1, 5 / 3, 7 / 3])
        self.maximum_frequency = waveform_generator.frequency_array[-1]
        self.fiducial_waveform_obtained = False
        self.check_if_bins_are_setup = False
        self.fiducial_polarizations = None
        self.per_detector_fiducial_waveforms = dict()
        self.per_detector_fiducial_waveform_points = dict()
        self.set_fiducial_waveforms(self.fiducial_parameters)
        logger.info("Initial fiducial waveforms set up")
        self.setup_bins()
        self.compute_summary_data()
        logger.info("Summary Data Obtained")

        if update_fiducial_parameters:
            # write a check to make sure prior is not None
            logger.info(
                "Using scipy optimization to find maximum likelihood parameters."
            )
            self.parameters_to_be_updated = [
                key
                for key in priors
                if not isinstance(priors[key], (DeltaFunction, Constraint, float, int))
            ]
            logger.info(
                f"Parameters over which likelihood is maximized: {self.parameters_to_be_updated}"
            )
            if parameter_bounds is None:
                logger.info("No parameter bounds were given. Using priors instead.")
                self.parameter_bounds = self.get_bounds_from_priors(priors)
            else:
                self.parameter_bounds = self.get_parameter_list_from_dictionary(
                    parameter_bounds
                )
            self.fiducial_parameters = self.find_maximum_likelihood_parameters(
                self.parameter_bounds, maximization_kwargs=maximization_kwargs
            )
        logger.info(
            f"Fiducial likelihood: {self.log_likelihood_ratio(self.fiducial_parameters):.2f}"
        )

    def __repr__(self):
        return (
            self.__class__.__name__
            + "(interferometers={},\n\twaveform_generator={},\n\fiducial_parameters={},".format(
                self.interferometers, self.waveform_generator, self.fiducial_parameters
            )
        )

    def setup_bins(self):
        """
        Setup the frequency bins following the method in
        https://arxiv.org/abs/1806.08792.

        If :code:`epsilon` is too small, the naive bins can be smaller than
        the frequency spacing of the data. We require that bins are at least
        as wide as this spacing.
        """
        frequency_array = self.waveform_generator.frequency_array
        gamma = self.gamma[:, np.newaxis]
        maximum_frequency = frequency_array[0]
        minimum_frequency = frequency_array[-1]
        for interferometer in self.interferometers:
            maximum_frequency = max(maximum_frequency, interferometer.maximum_frequency)
            minimum_frequency = min(minimum_frequency, interferometer.minimum_frequency)
        maximum_frequency = min(maximum_frequency, self.maximum_frequency)
        frequency_array_useful = frequency_array[
            (frequency_array >= minimum_frequency)
            & (frequency_array <= maximum_frequency)
        ]

        d_alpha = (
            self.chi
            * 2
            * np.pi
            / np.abs(
                (minimum_frequency**gamma) * np.heaviside(-gamma, 1)
                - (maximum_frequency**gamma) * np.heaviside(gamma, 1)
            )
        )
        d_phi = np.sum(np.sign(gamma) * d_alpha * frequency_array_useful**gamma, axis=0)
        d_phi_from_start = d_phi - d_phi[0]
        number_of_bins = int(d_phi_from_start[-1] // self.epsilon)
        bin_inds = list()
        bin_freqs = list()

        last_index = -1
        for i in range(number_of_bins + 1):
            bin_index = np.where(
                d_phi_from_start >= ((i / number_of_bins) * d_phi_from_start[-1])
            )[0][0]
            if bin_index == last_index:
                continue
            bin_freq = frequency_array_useful[bin_index]
            last_index = bin_index
            bin_index = np.where(frequency_array >= bin_freq)[0][0]
            bin_inds.append(bin_index)
            bin_freqs.append(bin_freq)
        self.bin_inds = np.array(bin_inds, dtype=int)
        self.bin_sizes = np.diff(bin_inds)
        self.bin_sizes[-1] += 1
        self.bin_freqs = np.array(bin_freqs)
        self.number_of_bins = len(self.bin_inds) - 1
        logger.debug(
            f"Set up {self.number_of_bins} bins "
            f"between {minimum_frequency} Hz and {maximum_frequency} Hz"
        )
        self.waveform_generator.waveform_arguments["frequency_bin_edges"] = (
            self.bin_freqs
        )
        self.bin_widths = self.bin_freqs[1:] - self.bin_freqs[:-1]
        self.bin_centers = (self.bin_freqs[1:] + self.bin_freqs[:-1]) / 2
        for interferometer in self.interferometers:
            name = interferometer.name
            self.per_detector_fiducial_waveform_points[name] = (
                self.per_detector_fiducial_waveforms[name][self.bin_inds]
            )
        if len(self.bin_freqs) <= self.spline_degree:
            raise ValueError(
                f"{len(self.bin_freqs)} bin edges are too few for splines of degree "
                f"{self.spline_degree}; decrease epsilon."
            )

    def _get_summary_fiducial_waveforms(self, snr_fraction=0.9999):
        r"""
        Return the fiducial waveform of each interferometer used for the
        summary data.

        Without :code:`fade_fiducial_waveform`, this is
        :code:`per_detector_fiducial_waveforms`. Otherwise the fiducial waveform
        is replaced by a constant at high frequency, following cogwheel
        (Roulet et al., (2022) Phys. Rev. D 106, 123015), which uses
        :code:`snr_fraction=0.99`; a larger fraction leaves the likelihood of
        waveforms without a taper essentially unchanged.

        Near the end of its support, e.g., in the taper of the NRTidal models,
        the fiducial waveform can be close to zero, so that the ratio of a
        waveform to it, which is interpolated between the bin edges, is large
        and not smooth. The cutoff is the first bin edge at or above the
        frequency where the cumulative :math:`\sum |h_0|^2 / P` over the
        interferometers reaches :code:`snr_fraction` of its total. Above the
        cutoff, the fiducial polarizations are replaced by their values at
        the cutoff, and over the :code:`spline_degree` bins below the cutoff
        they are joined with a :math:`\cos^2` taper. The detector responses
        are then computed from the faded polarizations, so that the time shift
        is applied after the fade. :code:`per_detector_fiducial_waveforms` is
        not changed.

        Parameters
        ==========
        snr_fraction: float
            Fraction of the squared signal-to-noise ratio of the fiducial
            waveform below the cutoff.

        Returns
        =======
        fiducial_waveforms: dict
            The fiducial waveform of each interferometer on the full frequency
            grid.
        """
        if not self.fade_fiducial_waveform:
            return dict(self.per_detector_fiducial_waveforms)
        frequency_array = self.waveform_generator.frequency_array
        polarizations = self.fiducial_polarizations
        snr_density = np.zeros(len(frequency_array))
        for interferometer in self.interferometers:
            mask = interferometer.frequency_mask
            waveform = self.per_detector_fiducial_waveforms[interferometer.name]
            snr_density[mask] += (
                np.abs(waveform[mask]) ** 2
                / interferometer.power_spectral_density_array[mask]
            )
        cumulative = np.cumsum(snr_density)
        if cumulative[-1] == 0:
            return dict(self.per_detector_fiducial_waveforms)
        cutoff_frequency = frequency_array[
            np.searchsorted(cumulative / cumulative[-1], snr_fraction)
        ]
        end = min(
            int(np.searchsorted(self.bin_freqs, cutoff_frequency)),
            len(self.bin_freqs) - 1,
        )
        while end > 0 and all(
            polarizations[mode][self.bin_inds[end]] == 0 for mode in polarizations
        ):
            end -= 1
        start_frequency = self.bin_freqs[max(end - self.spline_degree, 0)]
        end_frequency = self.bin_freqs[end]
        if end_frequency > start_frequency:
            fraction = (frequency_array - start_frequency) / (
                end_frequency - start_frequency
            )
            weight = np.cos(np.clip(fraction, 0, 1) * np.pi / 2) ** 2
        else:
            weight = (frequency_array <= end_frequency).astype(float)
        logger.debug(
            f"Fading the fiducial waveform between {start_frequency} Hz "
            f"and {end_frequency} Hz"
        )
        index = self.bin_inds[end]
        faded = {
            mode: values * weight + values[index] * (1 - weight)
            for mode, values in polarizations.items()
        }
        fiducial_waveforms = dict()
        for interferometer in self.interferometers:
            waveform = interferometer.get_detector_response(
                faded, self._fiducial_waveform_parameters
            )
            waveform *= interferometer.frequency_array <= self.maximum_frequency
            fiducial_waveforms[interferometer.name] = waveform
        return fiducial_waveforms

    def set_fiducial_waveforms(self, parameters):
        parameters = parameters.copy()
        self._set_fiducial()
        parameters.update(self.get_sky_frame_parameters(parameters=parameters))
        self._fiducial_waveform_parameters = parameters
        self.fiducial_polarizations = self.waveform_generator.frequency_domain_strain(
            parameters
        )
        self._unset_fiducial()

        maximum_nonzero_index = np.where(self.fiducial_polarizations["plus"] != 0j)[0][
            -1
        ]
        logger.debug(f"Maximum Nonzero Index is {maximum_nonzero_index}")
        maximum_nonzero_frequency = self.waveform_generator.frequency_array[
            maximum_nonzero_index
        ]
        logger.debug(f"Maximum Nonzero Frequency is {maximum_nonzero_frequency}")
        self.maximum_frequency = maximum_nonzero_frequency

        if self.fiducial_polarizations is None:
            raise ValueError(f"Cannot compute fiducial waveforms for {parameters}")

        for interferometer in self.interferometers:
            logger.debug(f"Maximum Frequency is {interferometer.maximum_frequency}")
            wf = interferometer.get_detector_response(
                self.fiducial_polarizations, parameters
            )
            wf *= interferometer.frequency_array <= self.maximum_frequency
            self.per_detector_fiducial_waveforms[interferometer.name] = wf

    def find_maximum_likelihood_parameters(
        self, parameter_bounds, iterations=5, maximization_kwargs=None
    ):
        if maximization_kwargs is None:
            maximization_kwargs = dict()
        parameters = deepcopy(self.fiducial_parameters)
        self._unset_fiducial()
        updated_parameters_list = self.get_parameter_list_from_dictionary(
            self.fiducial_parameters
        )
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
            updated_parameters_list = output["x"]
            updated_parameters = deepcopy(self.fiducial_parameters)
            updated_parameters.update(
                self.get_parameter_dictionary_from_list(updated_parameters_list)
            )
            self.set_fiducial_waveforms(updated_parameters)
            self.setup_bins()
            self.compute_summary_data()
            new_fiducial_ln_likelihood = self.log_likelihood_ratio(updated_parameters)
            logger.info(
                f"Fiducial ln likelihood ratio: {new_fiducial_ln_likelihood:.2f}"
            )
            if new_fiducial_ln_likelihood - old_fiducial_ln_likelihood < 0.1:
                break
            old_fiducial_ln_likelihood = new_fiducial_ln_likelihood

        logger.info("Fiducial waveforms updated")
        logger.info("Summary Data updated")
        return updated_parameters

    def lnlike_scipy_maximize(self, parameter_list, parameters):
        parameters.update(self.get_parameter_dictionary_from_list(parameter_list))
        return -self.log_likelihood_ratio(parameters)

    def get_parameter_dictionary_from_list(self, parameter_list):
        parameter_dictionary = dict(zip(self.parameters_to_be_updated, parameter_list))
        excluded_parameter_keys = set(self.fiducial_parameters) - set(
            self.parameters_to_be_updated
        )
        for key in excluded_parameter_keys:
            parameter_dictionary[key] = self.fiducial_parameters[key]
        return parameter_dictionary

    def get_parameter_list_from_dictionary(self, parameter_dict):
        return [parameter_dict[k] for k in self.parameters_to_be_updated]

    def get_bounds_from_priors(self, priors):
        bounds = []
        for key in self.parameters_to_be_updated:
            bounds.append([priors[key].minimum, priors[key].maximum])
        return bounds

    def compute_summary_data(self):
        r"""
        Compute the summary data of each interferometer: the weights
        :math:`D_j` of :math:`\langle h|d \rangle` and :math:`H_j` of
        :math:`\langle h|h \rangle` at bin edge :math:`j`.

        With :math:`S_j(f)` the spline of degree :code:`spline_degree` that is
        one at bin edge :math:`j` and zero at the other edges, and
        :math:`h_0` the fiducial waveform from
        :meth:`_get_summary_fiducial_waveforms`,

        .. math::

            D_j = \frac{4}{T} \sum_f S_j(f) \frac{h_0^*(f) d(f)}{P(f)}
            \Big/ h_0^*(f_j)

            H_j = \frac{4}{T} \sum_f S_j(f) \frac{|h_0(f)|^2}{P(f)}
            \Big/ |h_0(f_j)|^2

        so that :math:`\langle h|d \rangle = \sum_j D_j h^*(f_j)` and
        :math:`\langle h|h \rangle = \sum_j H_j |h(f_j)|^2` when the ratio of
        :math:`h` to :math:`h_0`, and its squared modulus, are splines with
        knots at the bin edges.
        """
        frequency_array = self.waveform_generator.frequency_array
        self._set_splines(frequency_array)
        self._summary_fiducial_waveforms = self._get_summary_fiducial_waveforms()
        summary_data = dict()
        for interferometer in self.interferometers:
            name = interferometer.name
            h0 = np.asarray(self._summary_fiducial_waveforms[name])
            h0_bins = h0[self.bin_inds]
            if np.any(h0_bins == 0):
                raise ValueError(
                    f"The fiducial waveform in {name} is zero at a bin edge; "
                    "use other fiducial parameters."
                )
            inverse_psd = np.zeros(len(interferometer.frequency_array))
            mask = interferometer.frequency_mask
            inverse_psd[mask] = 1 / interferometer.power_spectral_density_array[mask]
            strain = interferometer.frequency_domain_strain
            d_inner_h_weights = (
                self._get_summary_weights(
                    h0.conj() * strain * inverse_psd, interferometer.duration
                )
                / h0_bins.conj()
            )
            h_inner_h_weights = (
                self._get_summary_weights(
                    np.abs(h0) ** 2 * inverse_psd, interferometer.duration
                )
                / np.abs(h0_bins) ** 2
            ).real
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
        ==========
        frequency_array: array_like
            The frequencies at which the basis elements are evaluated.
        """
        number_of_edges = len(self.bin_freqs)
        coefficients = np.empty((number_of_edges, number_of_edges))
        for index, values in enumerate(np.eye(number_of_edges)):
            knots, spline_coefficients, _ = scipy.interpolate.splrep(
                self.bin_freqs, values, s=0, k=self.spline_degree
            )
            coefficients[index] = spline_coefficients[:number_of_edges]
        self._spline_coefficients = coefficients

        rows, columns, values = list(), list(), list()
        for index in range(number_of_edges):
            element_knots = knots[index : index + self.spline_degree + 2]
            element = scipy.interpolate.BSpline.basis_element(element_knots)
            start = np.searchsorted(frequency_array, element_knots[0], side="left")
            stop = np.searchsorted(frequency_array, element_knots[-1], side="right")
            rows.append(np.full(stop - start, index))
            columns.append(np.arange(start, stop))
            # A basis element vanishes at its last knot; evaluate just below it
            # so that the last element is one at the last bin edge
            values.append(
                element(
                    np.minimum(
                        frequency_array[start:stop], np.nextafter(element_knots[-1], 0)
                    )
                )
            )
        self._basis_splines = scipy.sparse.csr_matrix(
            (np.concatenate(values), (np.concatenate(rows), np.concatenate(columns))),
            shape=(number_of_edges, len(frequency_array)),
        )

    def _get_summary_weights(self, integrand, duration):
        r"""
        Return the weights :math:`W_j = (4 / T) \sum_f g(f) S_j(f)`, so that
        :math:`(4 / T) \sum_f g(f) r(f) = \sum_j W_j r(f_j)` for a spline
        :math:`r` of degree :code:`spline_degree` with knots at the bin edges.

        Parameters
        ==========
        integrand: array_like
            :math:`g(f)`, of shape (..., number of frequencies).
        duration: float
            The duration :math:`T` of the data.

        Returns
        =======
        weights: array_like
            Of shape (..., number of bin edges).
        """
        projected = (self._basis_splines @ np.atleast_2d(integrand).T).T
        weights = 4 / duration * projected @ self._spline_coefficients.T
        return weights.reshape(integrand.shape[:-1] + (len(self.bin_freqs),))

    def _compute_full_waveform(self, signal_polarizations, interferometer, parameters):
        """
        Return the waveform on the full frequency grid, as the fiducial
        waveform times the spline through the ratio of the waveform to it at
        the bin edges.
        """
        name = interferometer.name
        fiducial_waveform = self._summary_fiducial_waveforms[name]
        strain = interferometer.get_detector_response(
            waveform_polarizations=signal_polarizations,
            parameters=parameters,
            frequencies=self.bin_freqs,
        )
        ratio = np.asarray(strain) / fiducial_waveform[self.bin_inds]
        full_ratio = self._basis_splines.T @ (self._spline_coefficients.T @ ratio)
        return fiducial_waveform * full_ratio

    def calculate_snrs(
        self, waveform_polarizations, interferometer, *, return_array=True, parameters
    ):
        strain = interferometer.get_detector_response(
            waveform_polarizations=waveform_polarizations,
            parameters=parameters,
            frequencies=self.bin_freqs,
        )
        xp = array_module(strain)
        d_inner_h_weights, h_inner_h_weights = self.summary_data[interferometer.name]
        d_inner_h = (d_inner_h_weights * strain.conj()).sum()
        h_inner_h = (h_inner_h_weights * xp.abs(strain) ** 2).sum()
        optimal_snr_squared = h_inner_h
        complex_matched_filter_snr = d_inner_h / (optimal_snr_squared**0.5)

        if return_array and self.time_marginalization:
            full_waveform = self._compute_full_waveform(
                signal_polarizations=waveform_polarizations,
                interferometer=interferometer,
                parameters=parameters,
            )
            d_inner_h_array = (
                4
                / self.waveform_generator.duration
                * xp.fft.fft(
                    full_waveform[0:-1]
                    * interferometer.frequency_domain_strain.conj()[0:-1]
                    / interferometer.power_spectral_density_array[0:-1]
                )
            )

        else:
            d_inner_h_array = None

        return self._CalculatedSNRs(
            d_inner_h=d_inner_h,
            optimal_snr_squared=optimal_snr_squared,
            complex_matched_filter_snr=complex_matched_filter_snr,
            d_inner_h_array=d_inner_h_array,
        )

    def _set_fiducial(self):
        """
        Set the internal waveform generator to generate fiducial waveforms,
        e.g., for setting up the reference point.

        This is handled automatically within the likelihood for most applications
        and users shouldn't need to use this method.
        """
        self.waveform_generator.waveform_arguments["fiducial"] = 1
        self.waveform_generator._cache["parameters"] = None

    def _unset_fiducial(self):
        """
        Set the internal waveform generator to generate waveforms at the binning
        frequencies.

        This is handled automatically within the likelihood for most applications
        and users shouldn't need to use this method.
        """
        self.waveform_generator.waveform_arguments["fiducial"] = 0
        self.waveform_generator._cache["parameters"] = None
