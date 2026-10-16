import copy

import numpy as np

from ...core.utils import logger
from ..time_domain_utils import gohberg_semencul_product, time_domain_start_frequency
from .base import GravitationalWaveTransient


class TimeDomainGravitationalWaveTransient(GravitationalWaveTransient):
    """ A gravitational-wave transient likelihood object evaluated in the time domain

    It computes the log-likelihood of the data in the analysis window of each
    interferometer, assuming stationary colored Gaussian noise described by an
    autocovariance function, as in time-domain ringdown analyses (Carullo et al.
    arXiv:1902.07527, Isi & Farr arXiv:2107.05609). Products with the inverse
    noise covariance matrix use the Gohberg-Semencul formula. See Sharma et al.
    arXiv:2601.11239.

    Parameters
    ==========
    interferometers: list, bilby.gw.detector.InterferometerList
        A list of `bilby.detector.Interferometer` instances - contains the
        detector data, analysis windows and autocovariance functions
    waveform_generator: `bilby.waveform_generator.WaveformGenerator`
        An object which computes the time-domain strain of the signal,
        given some set of parameters
    distance_marginalization: bool, optional
        If true, marginalize over distance in the likelihood.
        This uses a look up table calculated at run time.
        The distance prior is set to be a delta function at the minimum
        distance allowed in the prior being marginalised over.
    priors: dict, optional
        If given, used in the distance marginalization.
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

        - :code:`sky`: sample in RA/dec, this is the default
        - e.g., :code:`"H1L1", ["H1", "L1"], InterferometerList(["H1", "L1"])`:
          sample in azimuth and zenith, `azimuth` and `zenith` defined in the
          frame where the z-axis is aligned the the vector connecting H1
          and L1.

    time_reference: str, optional
        Name of the reference for the sampled time parameter.

        - :code:`geocent`/:code:`geocenter`: sample in the time at the
          Earth's center, this is the default
        - e.g., :code:`H1`: sample in the time of arrival at H1

    placement: str, optional
        How the waveform is placed on the data samples: "nearest" (default),
        "fd_shift" or "model_time"; see
        :code:`bilby.gw.detector.Interferometer.get_time_domain_detector_response`
    time_marginalization, phase_marginalization, calibration_marginalization: bool, optional
        Not implemented for this likelihood; must be False

    Returns
    =======
    Likelihood: `bilby.core.likelihood.Likelihood`
        A likelihood object, able to compute the likelihood of the data given
        some model parameters
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
        """ Warn if the minimum frequency of the waveform and of an interferometer differ,
        and log the start frequency of the waveforms """
        waveform_minimum_frequency = getattr(
            self.waveform_generator, "waveform_arguments", dict()).get("minimum_frequency")
        if waveform_minimum_frequency is None:
            return
        for interferometer in self.interferometers:
            if not np.isclose(waveform_minimum_frequency, interferometer.minimum_frequency):
                logger.warning(
                    f"{interferometer.name}: minimum_frequency {interferometer.minimum_frequency:g} Hz "
                    f"differs from the waveform's minimum_frequency {waveform_minimum_frequency:g} Hz")
        arguments = self.waveform_generator.waveform_arguments
        if arguments.get("start_at_higher_mode_frequency", True) and "waveform_approximant" in arguments:
            start_frequency = time_domain_start_frequency(
                waveform_minimum_frequency, arguments["waveform_approximant"], arguments.get("mode_array"))
            logger.info(f"Waveforms start at {start_frequency:g} Hz, so that their modes with l <= 4 "
                        f"are present from {waveform_minimum_frequency:g} Hz")

    def _setup_noise_model(self):
        """ Compute and store C^-1 d and d^T C^-1 d for each interferometer """
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
            nyquist_frequency = interferometer.sampling_frequency / 2
            if interferometer.maximum_frequency < nyquist_frequency and not np.isclose(
                    interferometer.maximum_frequency, nyquist_frequency, rtol=1e-9, atol=0):
                logger.warning(
                    f"{interferometer.name}: {interferometer.maximum_frequency:g} - {nyquist_frequency:g} Hz "
                    "is outside the band and filled in the autocovariance function (consider downsampling "
                    f"the data to {2 * interferometer.maximum_frequency:g} Hz with condition_strain_data)")
            data = interferometer.analysis_data
            data_times_inverse_covariance = gohberg_semencul_product(
                *interferometer.gohberg_semencul_vectors, data)
            self._data_times_inverse_covariance[interferometer.name] = data_times_inverse_covariance
            self._data_inverse_covariance_data[interferometer.name] = float(
                np.dot(data_times_inverse_covariance, data))

    def _compute_full_waveform(self, signal_polarizations, interferometer, parameters):
        """ Project the waveform polarizations onto the interferometer and place them
        on the samples of its analysis window.

        Parameters
        ==========
        signal_polarizations: dict
            Output of the time-domain source model
        interferometer: bilby.gw.detector.Interferometer
            Interferometer to compute the response with respect to.
        parameters: dict
            The parameters of the signal
        """
        return interferometer.get_time_domain_detector_response(
            signal_polarizations, parameters, placement=self.placement)

    def calculate_snrs(self, waveform_polarizations, interferometer, *, return_array=True, parameters):
        """ Compute the snrs

        Parameters
        ==========
        waveform_polarizations: dict
            A dictionary of waveform polarizations and the corresponding array
        interferometer: bilby.gw.detector.Interferometer
            The bilby interferometer object
        return_array: bool
            Not used; kept for the :code:`GravitationalWaveTransient` interface
        parameters: dict
            The parameters of the signal

        Returns
        =======
        calculated_snrs: _CalculatedSNRs
            An object containing the SNR quantities (None if a model-time waveform failed)
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

    def generate_posterior_sample_from_marginalized_likelihood(self, parameters):
        """ Reconstruct the distance posterior from a run which used a likelihood
        which explicitly marginalised over distance.

        See Eq. (C29-C32) of https://arxiv.org/abs/1809.02293

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
        """ Generate a single sample from the posterior distribution for luminosity
        distance when using a likelihood which explicitly marginalises over
        distance.

        See Eq. (C29-C32) of https://arxiv.org/abs/1809.02293

        Parameters
        ==========
        signal_polarizations: dict, optional
            Polarizations modes of the template.
            Note: These are rescaled in place after the distance sample is
            generated to allow further parameter reconstruction to occur.

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
