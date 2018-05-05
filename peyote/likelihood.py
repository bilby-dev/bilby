from __future__ import division, print_function
import numpy as np
from scipy.special import logsumexp, jn, i0, iv
import peyote


class Likelihood(object):
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


class MarginalizedLikelihood(Likelihood):
    def __init__(self, interferometers, waveform_generator, distance_marginalization=False, phase_marginalization=False,
                 prior=None):
        Likelihood.__init__(self, interferometers, waveform_generator)
        self.distance_marginalization = distance_marginalization
        self.phase_marginalization = phase_marginalization
        self.prior = prior
        if self.distance_marginalization:
            if 'luminosity_distance' not in self.prior.keys():
                print('No prior provided for distance, using default prior.')
                self.prior['luminosity_distance'] = peyote.prior.create_default_prior('luminosity_distance')
            self.distance_array = np.linspace(self.prior['luminosity_distance'].minimum,
                                              self.prior['luminosity_distance'].maximum, 1000)
            self.delta_distance = self.distance_array[1] - self.distance_array[0]
            self.distance_prior_array = np.array([self.prior['luminosity_distance'].prob(distance)
                                                  for distance in self.distance_array])
            prior['luminosity_distance'] = 1
        if self.phase_marginalization:
            if 'psi' not in self.prior.keys() or not isinstance(prior['psi'], peyote.prior.Prior):
                print('No prior provided for polarization, using default prior.')
                self.prior['psi'] = peyote.prior.create_default_prior('psi')
            # self.phase_array = np.exp(1j * np.linspace(0, 2 * np.pi, 100))
            self.phase_array = np.exp(1j * np.linspace(self.prior['psi'].minimum, self.prior['psi'].maximum, 500))
            self.delta_phase = self.phase_array[1] - self.phase_array[0]
            self.phase_prior_array = np.array([self.prior['psi'].prob(phase) for phase in self.phase_array])
            prior['psi'] = 0

    def noise_log_likelihood(self):
        log_l = 0
        for interferometer in self.interferometers:
            log_l -= peyote.utils.noise_weighted_inner_product(interferometer.data, interferometer.data,
                                                               interferometer.power_spectral_density_array,
                                                               self.waveform_generator.time_duration) / 2
        return log_l.real

    def log_likelihood_ratio(self):
        waveform_polarizations = self.waveform_generator.frequency_domain_strain()

        matched_filter_snr_squared = 0
        optimal_snr_squared = 0

        for interferometer in self.interferometers:
            signal_ifo = interferometer.get_detector_response(waveform_polarizations,
                                                              self.waveform_generator.parameters)
            matched_filter_snr_squared += peyote.utils.matched_filter_snr_squared(signal_ifo, interferometer,
                                                                                  self.waveform_generator.time_duration)
            optimal_snr_squared += peyote.utils.optimal_snr_squared(signal_ifo, interferometer,
                                                                    self.waveform_generator.time_duration)

        if self.phase_marginalization:
            matched_filter_snr_squared = logsumexp([np.real(matched_filter_snr_squared * phase)
                                                    for phase in self.phase_array],
                                                   b=self.phase_prior_array * self.delta_phase)

        if self.distance_marginalization:
            log_l = logsumexp(matched_filter_snr_squared * self.waveform_generator.parameters['luminosity_distance']
                              / self.distance_array
                              - optimal_snr_squared * self.waveform_generator.parameters['luminosity_distance']**2
                              / self.distance_array**2 / 2, b=self.distance_prior_array * self.delta_distance)
        else:
            log_l = matched_filter_snr_squared - optimal_snr_squared / 2

        return log_l.real

    def log_likelihood(self):
        return self.log_likelihood() + self.noise_log_likelihood()


