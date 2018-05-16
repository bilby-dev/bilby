from __future__ import division, print_function
import numpy as np

try:
    from scipy.special import logsumexp
except ImportError:
    from scipy.misc import logsumexp
from scipy.special import i0e
from scipy.interpolate import interp1d
import tupak
import logging


class Likelihood(object):
    def __init__(self, interferometers, waveform_generator, distance_marginalization=False, phase_marginalization=False,
                 prior=None):
        # Likelihood.__init__(self, interferometers, waveform_generator)
        self.interferometers = interferometers
        self.waveform_generator = waveform_generator
        self.parameters = self.waveform_generator.parameters
        self.distance_marginalization = distance_marginalization
        self.phase_marginalization = phase_marginalization
        self.prior = prior

        if self.distance_marginalization:
            self.distance_array = np.array([])
            self.delta_distance = 0
            self.distance_prior_array = np.array([])
            self.setup_distance_marginalization()
            prior['luminosity_distance'] = 1

        if self.phase_marginalization:
            self.bessel_function_interped = None
            self.setup_phase_marginalization()
            prior['psi'] = 0

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

        for interferometer in self.interferometers:
            signal_ifo = interferometer.get_detector_response(waveform_polarizations,
                                                              self.waveform_generator.parameters)
            matched_filter_snr_squared += tupak.utils.matched_filter_snr_squared(
                signal_ifo, interferometer, self.waveform_generator.time_duration)

            optimal_snr_squared += tupak.utils.optimal_snr_squared(
                signal_ifo, interferometer, self.waveform_generator.time_duration)

        if self.phase_marginalization:
            matched_filter_snr_squared = self.bessel_function_interped(abs(matched_filter_snr_squared))

        else:
            matched_filter_snr_squared = matched_filter_snr_squared.real

        if self.distance_marginalization:

            optimal_snr_squared_array = optimal_snr_squared \
                                        * self.waveform_generator.parameters['luminosity_distance'] ** 2 \
                                        / self.distance_array ** 2

            matched_filter_snr_squared_array = matched_filter_snr_squared * \
                self.waveform_generator.parameters['luminosity_distance'] / self.distance_array

            log_l = logsumexp(matched_filter_snr_squared_array - optimal_snr_squared_array / 2,
                              b=self.distance_prior_array * self.delta_distance)
        else:
            log_l = matched_filter_snr_squared - optimal_snr_squared / 2

        return log_l.real

    def log_likelihood(self):
        return self.log_likelihood() + self.noise_log_likelihood()

    def setup_distance_marginalization(self):
        if 'luminosity_distance' not in self.prior.keys():
            logging.info('No prior provided for distance, using default prior.')
            self.prior['luminosity_distance'] = tupak.prior.create_default_prior('luminosity_distance')
        self.distance_array = np.linspace(self.prior['luminosity_distance'].minimum,
                                          self.prior['luminosity_distance'].maximum, int(1e4))
        self.delta_distance = self.distance_array[1] - self.distance_array[0]
        self.distance_prior_array = np.array([self.prior['luminosity_distance'].prob(distance)
                                              for distance in self.distance_array])

    def setup_phase_marginalization(self):
        if 'psi' not in self.prior.keys() or not isinstance(self.prior['psi'], tupak.prior.Prior):
            logging.info('No prior provided for polarization, using default prior.')
            self.prior['psi'] = tupak.prior.create_default_prior('psi')
        self.bessel_function_interped = interp1d(np.linspace(0, 1e6, int(1e5)),
                                                 np.log([i0e(snr) for snr in np.linspace(0, 1e6, int(1e5))])
                                                 + np.linspace(0, 1e6, int(1e5)),
                                                 bounds_error=False, fill_value=-np.inf)


class BasicLikelihood(object):
    def __init__(self, interferometers, waveform_generator):
        self.interferometers = interferometers
        self.waveform_generator = waveform_generator

    def noise_log_likelihood(self):
        log_l = 0
        for interferometer in self.interferometers:
            log_l -= 2. / self.waveform_generator.time_duration * np.sum(
                abs(interferometer.data) ** 2 / interferometer.power_spectral_density_array)
        return log_l.real

    def log_likelihood(self):
        log_l = 0
        waveform_polarizations = self.waveform_generator.frequency_domain_strain()
        if waveform_polarizations is None:
            return np.nan_to_num(-np.inf)
        for interferometer in self.interferometers:
            log_l += self.log_likelihood_interferometer(waveform_polarizations, interferometer)
        return log_l.real

    def log_likelihood_interferometer(self, waveform_polarizations, interferometer):
        signal_ifo = interferometer.get_detector_response(waveform_polarizations, self.waveform_generator.parameters)

        log_l = - 2. / self.waveform_generator.time_duration * np.vdot(interferometer.data - signal_ifo,
                                                                       (interferometer.data - signal_ifo)
                                                                       / interferometer.power_spectral_density_array)
        return log_l.real

    def log_likelihood_ratio(self):
        return self.log_likelihood() - self.noise_log_likelihood()


def get_binary_black_hole_likelihood(interferometers):
    """ A rapper to quickly set up a likelihood for BBH parameter estimation

    Parameters
    ----------
    interferometers: list
        A list of `tupak.detector.Interferometer` instances, typically the
        output of either `tupak.detector.get_interferometer_with_open_data`
        or `tupak.detector.get_interferometer_with_fake_noise_and_injection`
    Returns
    likelihood: tupak.likelihood.Likelihood
        The likelihood to pass to `run_sampler`
    """
    waveform_generator = tupak.waveform_generator.WaveformGenerator(time_duration=interferometers[0].duration,
                                                                    sampling_frequency=interferometers[
                                                                        0].sampling_frequency,
                                                                    frequency_domain_source_model=tupak.source.lal_binary_black_hole,
                                                                    parameters={'waveform_approximant': 'IMRPhenomPv2',
                                                                                'reference_frequency': 50})
    likelihood = tupak.likelihood.Likelihood(
        interferometers, waveform_generator)
    return likelihood


class HyperparameterLikelihood():
    def __init__(self, samples, hyper_prior, run_prior, **hyperparameters):
        self.samples = samples
        self.hyper_prior = hyper_prior
        self.run_prior = run_prior
        self.parameters = hyperparameters

    def log_likelihood(self):
        logl = []
        for samp in self.samples:
            logl.append(np.log(
                np.sum(self.hyper_prior(samp, **self.parameters) /
                       self.run_prior(samp))))
        return np.sum(logl)

    def noise_log_likelihood(self):
        return np.nan



