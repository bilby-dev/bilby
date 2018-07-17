import numpy as np
from scipy.interpolate import interp2d, interp1d

import tupak.gw.prior

try:
    from scipy.special import logsumexp
except ImportError:
    from scipy.misc import logsumexp
from scipy.special._ufuncs import i0e

import tupak
from tupak.core import likelihood as likelihood
from tupak.core.utils import logger


class GravitationalWaveTransient(likelihood.Likelihood):
    """ A gravitational-wave transient likelihood object

    This is the usual likelihood object to use for transient gravitational
    wave parameter estimation. It computes the log-likelihood in the frequency
    domain assuming a colored Gaussian noise model described by a power
    spectral density


    Parameters
    ----------
    interferometers: list
        A list of `tupak.detector.Interferometer` instances - contains the
        detector data and power spectral densities
    waveform_generator: `tupak.waveform_generator.WaveformGenerator`
        An object which computes the frequency-domain strain of the signal,
        given some set of parameters
    distance_marginalization: bool, optional
        If true, marginalize over distance in the likelihood.
        This uses a look up table calculated at run time.
    time_marginalization: bool, optional
        If true, marginalize over time in the likelihood.
        This uses a FFT.
    phase_marginalization: bool, optional
        If true, marginalize over phase in the likelihood.
        This is done analytically using a Bessel function.
    prior: dict, optional
        If given, used in the distance and phase marginalization.

    Returns
    -------
    Likelihood: `tupak.core.likelihood.Likelihood`
        A likelihood object, able to compute the likelihood of the data given
        some model parameters

    """

    def __init__(self, interferometers, waveform_generator, time_marginalization=False, distance_marginalization=False,
                 phase_marginalization=False, prior=None):

        self.waveform_generator = waveform_generator
        likelihood.Likelihood.__init__(self, waveform_generator.parameters)
        self.interferometers = tupak.gw.detector.InterferometerSet(interferometers)
        self.time_marginalization = time_marginalization
        self.distance_marginalization = distance_marginalization
        self.phase_marginalization = phase_marginalization
        self.prior = prior
        self._check_set_duration_and_sampling_frequency_of_waveform_generator()

        if self.distance_marginalization:
            self.check_prior_is_set()
            self.distance_array = np.array([])
            self.delta_distance = 0
            self.distance_prior_array = np.array([])
            self.setup_distance_marginalization()
            prior['luminosity_distance'] = 1  # this means the prior is a delta function fixed at the RHS value

        if self.phase_marginalization:
            self.check_prior_is_set()
            self.bessel_function_interped = None
            self.setup_phase_marginalization()
            prior['phase'] = 0

        if self.time_marginalization:
            self.check_prior_is_set()

    def _check_set_duration_and_sampling_frequency_of_waveform_generator(self):
        """ Check the waveform_generator has the same duration and
        sampling_frequency as the interferometers. If they are unset, then
        set them, if they differ, raise an error
        """

        attributes = ['duration', 'sampling_frequency', 'start_time']
        for attr in attributes:
            wfg_attr = getattr(self.waveform_generator, attr)
            ifo_attr = getattr(self.interferometers, attr)
            if wfg_attr is None:
                logger.debug(
                    "The waveform_generator {} is None. Setting from the "
                    "provided interferometers.".format(attr))
                setattr(self.waveform_generator, attr, ifo_attr)
            elif wfg_attr != ifo_attr:
                logger.warning(
                    "The waveform_generator {} is not equal to that of the "
                    "provided interferometers. Overwriting the "
                    "waveform_generator.".format(attr))

    def check_prior_is_set(self):
        if self.prior is None:
            raise ValueError("You can't use a marginalized likelihood without specifying a prior")

    @property
    def prior(self):
        return self.__prior

    @prior.setter
    def prior(self, prior):
        if prior is not None:
            self.__prior = prior
        else:
            self.__prior = dict()

    @property
    def non_standard_sampling_parameter_keys(self):
        return self.waveform_generator.non_standard_sampling_parameter_keys

    @property
    def parameters(self):
        return self.waveform_generator.parameters

    @parameters.setter
    def parameters(self, parameters):
        self.waveform_generator.parameters = parameters

    def noise_log_likelihood(self):
        log_l = 0
        for interferometer in self.interferometers:
            log_l -= tupak.gw.utils.noise_weighted_inner_product(
                interferometer.frequency_domain_strain,
                interferometer.frequency_domain_strain,
                interferometer.power_spectral_density_array,
                self.waveform_generator.duration) / 2
        return log_l.real

    def log_likelihood_ratio(self):
        waveform_polarizations = self.waveform_generator.frequency_domain_strain()

        if waveform_polarizations is None:
            return np.nan_to_num(-np.inf)

        matched_filter_snr_squared = 0
        optimal_snr_squared = 0
        matched_filter_snr_squared_tc_array = np.zeros(
                self.interferometers.frequency_array[0:-1].shape, dtype=np.complex128)
        for interferometer in self.interferometers:
            signal_ifo = interferometer.get_detector_response(waveform_polarizations,
                                                              self.waveform_generator.parameters)
            matched_filter_snr_squared += tupak.gw.utils.matched_filter_snr_squared(
                signal_ifo, interferometer, self.waveform_generator.duration)

            optimal_snr_squared += tupak.gw.utils.optimal_snr_squared(
                signal_ifo, interferometer, self.waveform_generator.duration)
            if self.time_marginalization:
                interferometer.time_marginalization = self.time_marginalization
                matched_filter_snr_squared_tc_array += 4. * (1. / self.waveform_generator.duration) * np.fft.ifft(
                    signal_ifo.conjugate()[0:-1] * interferometer.frequency_domain_strain[0:-1]
                    / interferometer.power_spectral_density_array[0:-1])\
                    * len(interferometer.frequency_domain_strain[0:-1])

        if self.time_marginalization:

            delta_tc = 1. / self.waveform_generator.sampling_frequency
            tc_log_norm = np.log(self.waveform_generator.duration * delta_tc)

            if self.distance_marginalization:

                rho_mf_ref_tc_array, rho_opt_ref = self.__setup_rho(matched_filter_snr_squared_tc_array,
                                                                    optimal_snr_squared)

                if self.phase_marginalization:

                    phase_marged_rho_mf_tc_array = self.bessel_function_interped(abs(rho_mf_ref_tc_array))

                    dist_marged_log_l_tc_array = self.interp_dist_margd_loglikelihood(phase_marged_rho_mf_tc_array,
                                                                                      rho_opt_ref)
                    log_l = logsumexp(dist_marged_log_l_tc_array, axis=0, b=delta_tc) - tc_log_norm

                else:

                    dist_marged_log_l_tc_array = self.interp_dist_margd_loglikelihood(rho_mf_ref_tc_array.real,
                                                                                      rho_opt_ref)

                    log_l = logsumexp(dist_marged_log_l_tc_array, axis=0, b=delta_tc)

            elif self.phase_marginalization:
                log_l = logsumexp(self.bessel_function_interped(abs(matched_filter_snr_squared_tc_array)), b=delta_tc) \
                        - optimal_snr_squared / 2. - tc_log_norm

            else:

                log_l = logsumexp(matched_filter_snr_squared_tc_array.real, axis=0,
                                  b=delta_tc) - optimal_snr_squared / 2. - tc_log_norm

        elif self.distance_marginalization:

            rho_mf_ref, rho_opt_ref = self.__setup_rho(matched_filter_snr_squared.real,
                                                       optimal_snr_squared)

            log_l = self.interp_dist_margd_loglikelihood(rho_mf_ref, rho_opt_ref)[0]

        elif self.phase_marginalization:
            matched_filter_snr_squared = self.bessel_function_interped(abs(matched_filter_snr_squared))
            log_l = matched_filter_snr_squared - optimal_snr_squared / 2

        else:
            log_l = matched_filter_snr_squared.real - optimal_snr_squared / 2

        return log_l.real

    def __setup_rho(self, matched_filter_snr_squared, optimal_snr_squared):
        rho_opt_ref = optimal_snr_squared.real * \
                      self.waveform_generator.parameters['luminosity_distance'] ** 2 \
                      / self.ref_dist ** 2.
        rho_mf_ref = matched_filter_snr_squared * \
            self.waveform_generator.parameters['luminosity_distance'] / self.ref_dist
        return rho_mf_ref, rho_opt_ref

    def log_likelihood(self):
        return self.log_likelihood_ratio() + self.noise_log_likelihood()

    def setup_distance_marginalization(self):
        if 'luminosity_distance' not in self.prior.keys():
            logger.info('No prior provided for distance, using default prior.')
            self.prior['luminosity_distance'] = tupak.core.prior.create_default_prior('luminosity_distance')
        self.distance_array = np.linspace(self.prior['luminosity_distance'].minimum,
                                          self.prior['luminosity_distance'].maximum, int(1e4))
        self.delta_distance = self.distance_array[1] - self.distance_array[0]
        self.distance_prior_array = np.array([self.prior['luminosity_distance'].prob(distance)
                                              for distance in self.distance_array])

        # Make the lookup table ###
        logger.info('Building lookup table for distance marginalisation.')
        self.ref_dist = 1000  # 1000 Mpc
        self.dist_margd_loglikelihood_array = np.zeros((400, 800))

        # optimal filter snr at fiducial distance of ref_dist Mpc
        self.rho_opt_ref_array = np.logspace(-3, 4, self.dist_margd_loglikelihood_array.shape[0])
        # matched filter snr at fiducial distance of ref_dist Mpc
        self.rho_mf_ref_array = np.hstack(
                (-np.logspace(2, -3, self.dist_margd_loglikelihood_array.shape[1] / 2),
                 np.logspace(-3, 4, self.dist_margd_loglikelihood_array.shape[1] / 2)))

        for ii, rho_opt_ref in enumerate(self.rho_opt_ref_array):

            for jj, rho_mf_ref in enumerate(self.rho_mf_ref_array):
                optimal_snr_squared_array = rho_opt_ref * self.ref_dist ** 2. \
                                            / self.distance_array ** 2

                matched_filter_snr_squared_array = rho_mf_ref * self.ref_dist / self.distance_array

                self.dist_margd_loglikelihood_array[ii][jj] = \
                    logsumexp(matched_filter_snr_squared_array -
                              optimal_snr_squared_array / 2,
                              b=self.distance_prior_array * self.delta_distance)

        log_norm = logsumexp(0. / self.distance_array - 0. / self.distance_array ** 2.,
                             b=self.distance_prior_array * self.delta_distance)
        self.dist_margd_loglikelihood_array -= log_norm

        self.interp_dist_margd_loglikelihood = interp2d(self.rho_mf_ref_array, self.rho_opt_ref_array,
                                                        self.dist_margd_loglikelihood_array)

    def setup_phase_marginalization(self):
        if 'phase' not in self.prior.keys() or not isinstance(self.prior['phase'], tupak.core.prior.Prior):
            logger.info('No prior provided for phase at coalescence, using default prior.')
            self.prior['phase'] = tupak.core.prior.create_default_prior('phase')
        self.bessel_function_interped = interp1d(np.linspace(0, 1e6, int(1e5)),
                                                 np.log([i0e(snr) for snr in np.linspace(0, 1e6, int(1e5))])
                                                 + np.linspace(0, 1e6, int(1e5)),
                                                 bounds_error=False, fill_value=-np.inf)


class BasicGravitationalWaveTransient(likelihood.Likelihood):

    def __init__(self, interferometers, waveform_generator):
        """

        A likelihood object, able to compute the likelihood of the data given
        some model parameters

        The simplest frequency-domain gravitational wave transient likelihood. Does
        not include distance/phase marginalization.


        Parameters
        ----------
        interferometers: list
            A list of `tupak.gw.detector.Interferometer` instances - contains the
            detector data and power spectral densities
        waveform_generator: tupak.gw.waveform_generator.WaveformGenerator
            An object which computes the frequency-domain strain of the signal,
            given some set of parameters

        """
        likelihood.Likelihood.__init__(self, waveform_generator.parameters)
        self.interferometers = interferometers
        self.waveform_generator = waveform_generator

    def noise_log_likelihood(self):
        """ Calculates the real part of noise log-likelihood

        Returns
        -------
        float: The real part of the noise log likelihood

        """
        log_l = 0
        for interferometer in self.interferometers:
            log_l -= 2. / self.waveform_generator.duration * np.sum(
                abs(interferometer.frequency_domain_strain) ** 2 /
                interferometer.power_spectral_density_array)
        return log_l.real

    def log_likelihood(self):
        """ Calculates the real part of log-likelihood value

        Returns
        -------
        float: The real part of the log likelihood

        """
        log_l = 0
        waveform_polarizations = self.waveform_generator.frequency_domain_strain()
        if waveform_polarizations is None:
            return np.nan_to_num(-np.inf)
        for interferometer in self.interferometers:
            log_l += self.log_likelihood_interferometer(
                waveform_polarizations, interferometer)
        return log_l.real

    def log_likelihood_interferometer(self, waveform_polarizations,
                                      interferometer):
        """

        Parameters
        ----------
        waveform_polarizations: dict
            Dictionary containing the desired waveform polarization modes and the related strain
        interferometer: tupak.gw.detector.Interferometer
            The Interferometer object we want to have the log-likelihood for

        Returns
        -------
        float: The real part of the log-likelihood for this interferometer

        """
        signal_ifo = interferometer.get_detector_response(
            waveform_polarizations, self.waveform_generator.parameters)

        log_l = - 2. / self.waveform_generator.duration * np.vdot(
            interferometer.frequency_domain_strain - signal_ifo,
            (interferometer.frequency_domain_strain - signal_ifo)
            / interferometer.power_spectral_density_array)
        return log_l.real


def get_binary_black_hole_likelihood(interferometers):
    """ A rapper to quickly set up a likelihood for BBH parameter estimation

    Parameters
    ----------
    interferometers: list
        A list of `tupak.detector.Interferometer` instances, typically the
        output of either `tupak.detector.get_interferometer_with_open_data`
        or `tupak.detector.get_interferometer_with_fake_noise_and_injection`

    Returns
    -------
    tupak.GravitationalWaveTransient: The likelihood to pass to `run_sampler`

    """
    waveform_generator = tupak.gw.waveform_generator.WaveformGenerator(
        duration=interferometers.duration, sampling_frequency=interferometers.sampling_frequency,
        frequency_domain_source_model=tupak.gw.source.lal_binary_black_hole,
        parameters={'waveform_approximant': 'IMRPhenomPv2', 'reference_frequency': 50})
    return tupak.gw.likelihood.GravitationalWaveTransient(interferometers, waveform_generator)

