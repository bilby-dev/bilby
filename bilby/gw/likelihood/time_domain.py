import copy

import numpy as np

from ...core.utils import logger
from ..time_domain_utils import gohberg_semencul_product
from .base import GravitationalWaveTransient


class TimeDomainGravitationalWaveTransient(GravitationalWaveTransient):
    """ A gravitational-wave transient likelihood evaluated in the time domain.

    The noise in each interferometer is a stationary Gaussian process with
    covariance matrix :math:`C_{ij} = \\rho(|i - j|)`, where :math:`\\rho` is
    the autocovariance function (ACF) of the interferometer. For the data in
    the analysis window, :math:`d` (:code:`interferometer.analysis_data`), and
    a template :math:`h` on the same samples

    .. math::

        \\ln \\mathcal{L} = \\sum_{\\rm ifo} \\left[ d^T C^{-1} h
        - \\frac{1}{2} h^T C^{-1} h - \\frac{1}{2} d^T C^{-1} d \\right].

    :code:`log_likelihood_ratio` returns the first two terms and
    :code:`noise_log_likelihood` the last. Products with :math:`C^{-1}` are
    computed from the Gohberg-Semencul vectors stored on each interferometer
    (:code:`interferometer.gohberg_semencul_vectors`), without forming
    :math:`C^{-1}`.

    The template is projected onto each interferometer and placed so that the
    model's t=0 arrives at geocent_time plus the time delay from the
    geocenter. Template samples outside the analysis window are dropped.

    Parameters
    ==========
    interferometers: list, bilby.gw.detector.InterferometerList
        A list of :code:`bilby.gw.detector.Interferometer` instances, each with
        an analysis window, time-domain strain data covering it and an
        autocovariance function.
    waveform_generator: bilby.gw.waveform_generator.WaveformGenerator
        A WaveformGenerator with a time-domain source model, e.g.
        :code:`bilby.gw.source.lal_binary_black_hole_time_domain`.
    distance_marginalization: bool, optional
        If true, marginalize over distance in the likelihood. This uses a
        look up table calculated at run time. The distance prior is set to be a
        delta function at the minimum distance allowed in the prior being
        marginalised over.
    priors: dict, optional
        If given, used in the distance marginalization.
    distance_marginalization_lookup_table: (dict, str), optional
        If a dict, dictionary containing the lookup_table, distance_array,
        (distance) prior_array, and reference_distance used to construct
        the table. If a string the name of a file containing these quantities.
        The lookup table is stored after construction in either the provided
        string or a default location: '.distance_marginalization_lookup_dmin{}_dmax{}_n{}.npz'
    reference_frame: (str, bilby.gw.detector.InterferometerList, list), optional
        Definition of the reference frame for the sky location, as in
        :code:`GravitationalWaveTransient`.
    time_reference: str, optional
        Name of the reference for the sampled time parameter, as in
        :code:`GravitationalWaveTransient`.
    placement: str, optional
        How the template is placed on the data samples: "nearest" (default)
        or "fd_shift" (see :code:`bilby.gw.time_domain_utils.place_time_domain_signal`),
        or "model_time": the waveform is evaluated at each detector's data
        samples, without a taper (needs a model-time source model, e.g.
        :code:`bilby.gw.source.lal_binary_black_hole_model_time`).
        Use the same placement for injection and analysis.
    time_marginalization, phase_marginalization, calibration_marginalization: bool
        Not implemented for this likelihood; must be False.

    Returns
    =======
    Likelihood: `bilby.core.likelihood.Likelihood`
        A likelihood object, able to compute the likelihood of the data given
        some model parameters.
    """

    likelihood_domain = "time"

    def __init__(
            self, interferometers, waveform_generator, distance_marginalization=False,
            priors=None, distance_marginalization_lookup_table=None, reference_frame="sky",
            time_reference="geocenter", placement="nearest", time_marginalization=False,
            phase_marginalization=False, calibration_marginalization=False,
    ):
        for name, value in dict(
                time_marginalization=time_marginalization,
                phase_marginalization=phase_marginalization,
                calibration_marginalization=calibration_marginalization).items():
            if value:
                raise NotImplementedError(
                    f"{name} is not implemented for {self.__class__.__name__}")
        if getattr(waveform_generator, "time_domain_source_model", None) is None:
            raise ValueError(
                f"{self.__class__.__name__} needs a waveform generator with a "
                "time_domain_source_model, e.g. bilby.gw.source.lal_binary_black_hole_time_domain")
        if placement not in ("nearest", "fd_shift", "model_time"):
            raise ValueError(
                f"placement must be 'nearest', 'fd_shift' or 'model_time', not {placement!r}")

        super(TimeDomainGravitationalWaveTransient, self).__init__(
            interferometers=interferometers,
            waveform_generator=waveform_generator,
            time_marginalization=False,
            distance_marginalization=distance_marginalization,
            phase_marginalization=False,
            calibration_marginalization=False,
            priors=priors,
            distance_marginalization_lookup_table=distance_marginalization_lookup_table,
            jitter_time=False,
            reference_frame=reference_frame,
            time_reference=time_reference,
        )
        self.placement = placement
        self._check_minimum_frequency()
        self._setup_noise_model()

    def __repr__(self):
        return self.__class__.__name__ + (
            '(interferometers={},\n\twaveform_generator={},\n\tdistance_marginalization={}, '
            'priors={}, placement={})'.format(
                self.interferometers, self.waveform_generator, self.distance_marginalization,
                self.priors, self.placement))

    def _check_minimum_frequency(self):
        """ Warn if the waveform's minimum_frequency differs from an
        interferometer's minimum_frequency. """
        waveform_minimum_frequency = getattr(
            self.waveform_generator, "waveform_arguments", dict()).get("minimum_frequency")
        if waveform_minimum_frequency is None:
            return
        for interferometer in self.interferometers:
            if not np.isclose(waveform_minimum_frequency, interferometer.minimum_frequency):
                logger.warning(
                    f"{interferometer.name}: minimum_frequency {interferometer.minimum_frequency:g} Hz "
                    f"differs from the waveform's minimum_frequency {waveform_minimum_frequency:g} Hz")

    def _setup_noise_model(self):
        """ d^T C^-1 and d^T C^-1 d for each interferometer. """
        self._data_times_inverse_covariance = dict()
        self._data_inverse_covariance_data = dict()
        for interferometer in self.interferometers:
            acf = interferometer.autocovariance_function
            if acf is None or acf.acf_array is None:
                raise ValueError(f"{interferometer.name}: no autocovariance_function set")
            if not np.isclose(acf.sampling_frequency, interferometer.sampling_frequency,
                              rtol=1e-9, atol=0):
                raise ValueError(
                    f"{interferometer.name}: autocovariance function sampling frequency "
                    f"{acf.sampling_frequency:g} Hz differs from the data's "
                    f"{interferometer.sampling_frequency:g} Hz")
            data = interferometer.analysis_data
            data_times_inverse_covariance = gohberg_semencul_product(
                *interferometer.gohberg_semencul_vectors, data)
            self._data_times_inverse_covariance[interferometer.name] = data_times_inverse_covariance
            self._data_inverse_covariance_data[interferometer.name] = float(
                np.dot(data_times_inverse_covariance, data))

    def _compute_full_waveform(self, signal_polarizations, interferometer, parameters):
        """ Project the time-domain polarizations onto the interferometer and
        place them on the samples of its analysis window.

        Parameters
        ==========
        signal_polarizations: dict
            Output of the time-domain source model (polarizations and 'epoch').
        interferometer: bilby.gw.detector.Interferometer
        parameters: dict

        Returns
        =======
        array_like: The template on the samples of the analysis window.
        """
        return interferometer.get_time_domain_detector_response(
            signal_polarizations, parameters, placement=self.placement)

    def calculate_snrs(self, waveform_polarizations, interferometer, *, return_array=True, parameters):
        """ Compute :math:`d^T C^{-1} h`, :math:`h^T C^{-1} h` and the matched
        filter SNR in the analysis window of one interferometer.

        Parameters
        ----------
        waveform_polarizations: dict
            Output of the time-domain source model.
        interferometer: bilby.gw.detector.Interferometer
        return_array: bool
            Unused (no time or calibration marginalization); kept for the
            :code:`GravitationalWaveTransient` interface.
        parameters: dict

        Returns
        -------
        calculated_snrs: _CalculatedSNRs, or None if a model-time waveform failed
        """
        signal = self._compute_full_waveform(
            signal_polarizations=waveform_polarizations,
            interferometer=interferometer,
            parameters=parameters,
        )
        if signal is None:
            return None
        d_inner_h = float(np.dot(self._data_times_inverse_covariance[interferometer.name], signal))
        optimal_snr_squared = float(np.dot(
            signal, gohberg_semencul_product(*interferometer.gohberg_semencul_vectors, signal)))
        if optimal_snr_squared > 0:
            matched_filter_snr = d_inner_h / optimal_snr_squared ** 0.5
        else:
            matched_filter_snr = 0.0
        return self._CalculatedSNRs(
            d_inner_h=d_inner_h,
            optimal_snr_squared=optimal_snr_squared,
            complex_matched_filter_snr=matched_filter_snr,
            d_inner_h_array=None,
            optimal_snr_squared_array=None,
        )

    def _calculate_noise_log_likelihood(self):
        return -0.5 * sum(self._data_inverse_covariance_data.values())

    def log_likelihood_ratio(self, parameters):
        parameters = copy.deepcopy(parameters)
        parameters.update(self.get_sky_frame_parameters(parameters))
        waveform_polarizations = self.waveform_generator.time_domain_strain(parameters)
        if waveform_polarizations is None:
            return np.nan_to_num(-np.inf)

        total_snrs = self._CalculatedSNRs()
        for interferometer in self.interferometers:
            snrs = self.calculate_snrs(
                waveform_polarizations=waveform_polarizations,
                interferometer=interferometer,
                parameters=parameters,
            )
            if snrs is None:
                return np.nan_to_num(-np.inf)
            total_snrs += snrs
        log_l = self.compute_log_likelihood_from_snrs(total_snrs, parameters=parameters)
        return float(np.real(log_l))

    def compute_per_detector_log_likelihood(self, parameters):
        parameters.update(self.get_sky_frame_parameters(parameters))
        waveform_polarizations = self.waveform_generator.time_domain_strain(parameters)
        for interferometer in self.interferometers:
            per_detector_snr = self.calculate_snrs(
                waveform_polarizations=waveform_polarizations,
                interferometer=interferometer,
                parameters=parameters,
            )
            parameters['{}_log_likelihood'.format(interferometer.name)] = (
                np.nan_to_num(-np.inf) if per_detector_snr is None else
                self.compute_log_likelihood_from_snrs(per_detector_snr, parameters=parameters))
        return parameters.copy()

    # ------------------------------------------------------------------
    # Reconstruction of marginalized parameters
    # ------------------------------------------------------------------
    def generate_posterior_sample_from_marginalized_likelihood(self, parameters):
        """ Reconstruct the distance posterior from a run which used the
        distance-marginalized likelihood (Eq. (C29-C32) of
        https://arxiv.org/abs/1809.02293).

        Returns
        =======
        sample: dict
            Returns the parameters with new samples.
        """
        if not self.distance_marginalization:
            return parameters
        parameters['luminosity_distance'] = self.generate_distance_sample_from_marginalized_likelihood(
            parameters=parameters)
        return parameters.copy()

    def generate_distance_sample_from_marginalized_likelihood(
            self, signal_polarizations=None, *, parameters):
        """ Generate a single sample from the posterior distribution for
        luminosity distance when using the distance-marginalized likelihood.

        Parameters
        ==========
        signal_polarizations: dict, optional
            Output of the time-domain source model. These are rescaled in place
            after the distance sample is generated.

        Returns
        =======
        new_distance: float
            Sample from the distance posterior.
        """
        if signal_polarizations is None:
            parameters.update(self.get_sky_frame_parameters(parameters))
            signal_polarizations = copy.deepcopy(
                self.waveform_generator.time_domain_strain(parameters))
        return super(TimeDomainGravitationalWaveTransient, self).\
            generate_distance_sample_from_marginalized_likelihood(
                signal_polarizations=signal_polarizations, parameters=parameters)

    def _rescale_signal(self, signal, new_distance):
        factor = self._ref_dist / new_distance
        if "polarizations_at" in signal:
            polarizations_at = signal["polarizations_at"]

            def rescaled(times):
                polarizations = polarizations_at(times)
                if polarizations is None:
                    return None
                return {mode: value * factor for mode, value in polarizations.items()}

            signal["polarizations_at"] = rescaled
            return
        for mode in signal:
            if mode != "epoch":
                signal[mode] *= factor

    @property
    def meta_data(self):
        meta_data = super(TimeDomainGravitationalWaveTransient, self).meta_data
        meta_data.update(likelihood_domain=self.likelihood_domain, placement=self.placement)
        return meta_data
