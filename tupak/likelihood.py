from __future__ import division, print_function

import numpy as np

try:
    from scipy.special import logsumexp
except ImportError:
    from scipy.misc import logsumexp
from scipy.special import i0e
from scipy.interpolate import interp1d
from scipy.interpolate import interp2d
import tupak
import logging


class Likelihood(object):
    """ Empty likelihood class to be subclassed by other likelihoods """

    def __init__(self, parameters=None):
        self.parameters = parameters

    def log_likelihood(self):
        return np.nan

    def noise_log_likelihood(self):
        return np.nan

    def log_likelihood_ratio(self):
        return self.log_likelihood() - self.noise_log_likelihood()


class GravitationalWaveTransient(Likelihood):
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
    distance_marginalization: bool
        If true, analytic distance marginalization
    phase_marginalization: bool
        If true, analytic phase marginalization
    prior: dict
        If given, used in the distance and phase marginalization.

    Returns
    -------
    Likelihood: `tupak.likelihood.Likelihood`
        A likelihood object, able to compute the likelihood of the data given
        some model parameters

    """

    def __init__(self, interferometers, waveform_generator, time_marginalization=False, distance_marginalization=False,
                 phase_marginalization=False, prior=None):

        Likelihood.__init__(self, waveform_generator.parameters)
        self.interferometers = interferometers
        self.waveform_generator = waveform_generator
        self.non_standard_sampling_parameter_keys = self.waveform_generator.non_standard_sampling_parameter_keys
        self.time_marginalization = time_marginalization
        self.distance_marginalization = distance_marginalization
        self.phase_marginalization = phase_marginalization
        self.prior = prior

        if self.distance_marginalization:
            self.distance_array = np.array([])
            self.delta_distance = 0
            self.distance_prior_array = np.array([])
            self.setup_distance_marginalization()
            prior['luminosity_distance'] = 1  # this means the prior is a delta function fixed at the RHS value

        if self.phase_marginalization:
            self.bessel_function_interped = None
            self.setup_phase_marginalization()
            prior['phase'] = 0

    @property
    def prior(self):
        return self.__prior

    @prior.setter
    def prior(self, prior):
        if prior is not None:
            self.__prior = prior
        else:
            self.__prior = dict()

    def noise_log_likelihood(self):
        log_l = 0
        for interferometer in self.interferometers:
            log_l -= tupak.utils.noise_weighted_inner_product(interferometer.data, interferometer.data,
                                                              interferometer.power_spectral_density_array,
                                                              self.waveform_generator.time_duration) / 2
        return log_l.real

    def log_likelihood_ratio(self):
        waveform_polarizations = self.waveform_generator.frequency_domain_strain()

        if waveform_polarizations is None:
            return np.nan_to_num(-np.inf)

        matched_filter_snr_squared = 0
        optimal_snr_squared = 0
        matched_filter_snr_squared_tc_array = np.zeros(self.interferometers[0].data[0:-1].shape, dtype=np.complex128)
        for interferometer in self.interferometers:
            signal_ifo = interferometer.get_detector_response(waveform_polarizations,
                                                              self.waveform_generator.parameters)
            matched_filter_snr_squared += tupak.utils.matched_filter_snr_squared(
                signal_ifo, interferometer, self.waveform_generator.time_duration)

            optimal_snr_squared += tupak.utils.optimal_snr_squared(
                signal_ifo, interferometer, self.waveform_generator.time_duration)
            if self.time_marginalization:
                interferometer.time_marginalization = self.time_marginalization
                matched_filter_snr_squared_tc_array += 4. * (1. / interferometer.duration) * np.fft.ifft(
                    signal_ifo.conjugate()[0:-1] * interferometer.data[0:-1]
                    / interferometer.power_spectral_density_array[0:-1]) * len(interferometer.data[0:-1])

        if self.time_marginalization:

            delta_tc = 1. / self.waveform_generator.sampling_frequency
            tc_log_norm = np.log(self.interferometers[0].duration * delta_tc)

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
                log_l = logsumexp(self.bessel_function_interped(abs(matched_filter_snr_squared_tc_array)), b=delta_tc)\
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
                     self.waveform_generator.parameters['luminosity_distance'] \
                     / self.ref_dist
        return rho_mf_ref, rho_opt_ref

    def log_likelihood(self):
        return self.log_likelihood_ratio() + self.noise_log_likelihood()

    def setup_distance_marginalization(self):
        if 'luminosity_distance' not in self.prior.keys():
            logging.info('No prior provided for distance, using default prior.')
            self.prior['luminosity_distance'] = tupak.prior.create_default_prior('luminosity_distance')
        self.distance_array = np.linspace(self.prior['luminosity_distance'].minimum,
                                          self.prior['luminosity_distance'].maximum, int(1e4))
        self.delta_distance = self.distance_array[1] - self.distance_array[0]
        self.distance_prior_array = np.array([self.prior['luminosity_distance'].prob(distance)
                                              for distance in self.distance_array])

        ### Make the lookup table ###
        self.ref_dist = 1000  # 1000 Mpc
        self.dist_margd_loglikelihood_array = np.zeros((400, 800))

        self.rho_opt_ref_array = np.logspace(-3, 4, self.dist_margd_loglikelihood_array.shape[
            0])  # optimal filter snr at fiducial distance of ref_dist Mpc
        self.rho_mf_ref_array = np.hstack((-np.logspace(2, -3, self.dist_margd_loglikelihood_array.shape[1] / 2), \
                                           np.logspace(-3, 4, self.dist_margd_loglikelihood_array.shape[
                                               1] / 2)))  # matched filter snr at fiducial distance of ref_dist Mpc

        for ii, rho_opt_ref in enumerate(self.rho_opt_ref_array):

            for jj, rho_mf_ref in enumerate(self.rho_mf_ref_array):
                optimal_snr_squared_array = rho_opt_ref * self.ref_dist ** 2. \
                                            / self.distance_array ** 2

                matched_filter_snr_squared_array = rho_mf_ref * self.ref_dist \
                                                   / self.distance_array

                self.dist_margd_loglikelihood_array[ii][jj] = \
                    logsumexp(matched_filter_snr_squared_array - \
                              optimal_snr_squared_array / 2, \
                              b=self.distance_prior_array * self.delta_distance)

        log_norm = logsumexp(0. / self.distance_array - 0. / self.distance_array ** 2., \
                             b=self.distance_prior_array * self.delta_distance)
        self.dist_margd_loglikelihood_array -= log_norm

        self.interp_dist_margd_loglikelihood = interp2d(self.rho_mf_ref_array, self.rho_opt_ref_array,
                                                        self.dist_margd_loglikelihood_array)

    def setup_phase_marginalization(self):
        if 'phase' not in self.prior.keys() or not isinstance(self.prior['phase'], tupak.prior.Prior):
            logging.info('No prior provided for phase at coalescence, using default prior.')
            self.prior['phase'] = tupak.prior.create_default_prior('phase')
        self.bessel_function_interped = interp1d(np.linspace(0, 1e6, int(1e5)),
                                                 np.log([i0e(snr) for snr in np.linspace(0, 1e6, int(1e5))])
                                                 + np.linspace(0, 1e6, int(1e5)),
                                                 bounds_error=False, fill_value=-np.inf)


class BasicGravitationalWaveTransient(Likelihood):
    """ A basic gravitational wave transient likelihood

    The simplest frequency-domain gravitational wave transient likelihood. Does
    not include distance/phase marginalization.

    Parameters
    ----------
    interferometers: list
        A list of `tupak.detector.Interferometer` instances - contains the
        detector data and power spectral densities
    waveform_generator: `tupak.waveform_generator.WaveformGenerator`
        An object which computes the frequency-domain strain of the signal,
        given some set of parameters

    Returns
    -------
    Likelihood: `tupak.likelihood.Likelihood`
        A likelihood object, able to compute the likelihood of the data given
        some model parameters

    """

    def __init__(self, interferometers, waveform_generator):
        Likelihood.__init__(self, waveform_generator.parameters)
        self.interferometers = interferometers
        self.waveform_generator = waveform_generator

    def noise_log_likelihood(self):
        log_l = 0
        for interferometer in self.interferometers:
            log_l -= 2. / self.waveform_generator.time_duration * np.sum(
                abs(interferometer.data) ** 2 /
                interferometer.power_spectral_density_array)
        return log_l.real

    def log_likelihood(self):
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
        signal_ifo = interferometer.get_detector_response(
            waveform_polarizations, self.waveform_generator.parameters)

        log_l = - 2. / self.waveform_generator.time_duration * np.vdot(
            interferometer.data - signal_ifo,
            (interferometer.data - signal_ifo)
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
    likelihood: tupak.likelihood.GravitationalWaveTransient
        The likelihood to pass to `run_sampler`
    """
    waveform_generator = tupak.waveform_generator.WaveformGenerator(
        time_duration=interferometers[0].duration, sampling_frequency=interferometers[0].sampling_frequency,
        frequency_domain_source_model=tupak.source.lal_binary_black_hole,
        parameters={'waveform_approximant': 'IMRPhenomPv2', 'reference_frequency': 50})
    likelihood = tupak.likelihood.GravitationalWaveTransient(interferometers, waveform_generator)
    return likelihood


class HyperparameterLikelihood(Likelihood):
    """ A likelihood for infering hyperparameter posterior distributions

    See Eq. (1) of https://arxiv.org/abs/1801.02699 for a definition.

    Parameters
    ----------
    samples: list
        An N-dimensional list of individual sets of samples. Each set may have
        a different size.
    hyper_prior: `tupak.prior.Prior`
        A prior distribution with a `parameters` argument pointing to the
        hyperparameters to infer from the samples. These may need to be
        initialized to any arbitrary value, but this will not effect the
        result.
    run_prior: `tupak.prior.Prior`
        The prior distribution used in the inidivudal inferences which resulted
        in the set of samples.

    """

    def __init__(self, samples, hyper_prior, run_prior):
        Likelihood.__init__(self, parameters=hyper_prior.__dict__)
        self.samples = samples
        self.hyper_prior = hyper_prior
        self.run_prior = run_prior
        if hasattr(hyper_prior, 'lnprob') and hasattr(run_prior, 'lnprob'):
            logging.info("Using log-probabilities in likelihood")
            self.log_likelihood = self.log_likelihood_using_lnprob
        else:
            logging.info("Using probabilities in likelihood")
            self.log_likelihood = self.log_likelihood_using_prob

    def log_likelihood_using_lnprob(self):
        L = []
        self.hyper_prior.__dict__.update(self.parameters)
        for samp in self.samples:
            f = self.hyper_prior.lnprob(samp) - self.run_prior.lnprob(samp)
            L.append(logsumexp(f))
        return np.sum(L)

    def log_likelihood_using_prob(self):
        L = []
        self.hyper_prior.__dict__.update(self.parameters)
        for samp in self.samples:
            L.append(
                np.sum(self.hyper_prior.prob(samp) /
                       self.run_prior.prob(samp)))
        return np.sum(np.log(L))
