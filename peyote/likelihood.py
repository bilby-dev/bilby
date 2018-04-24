import numpy as np


class Likelihood(object):
    def __init__(self, interferometers, waveform_generator, noise_log_likelihood=None):
        self.interferometers = interferometers
        self.waveform_generator = waveform_generator
        self.noise_log_likelihood = noise_log_likelihood

    @property
    def noise_log_likelihood(self):
        return self.__noise_log_likelihood

    @noise_log_likelihood.setter
    def noise_log_likelihood(self, noise_log_likelihood):
        if noise_log_likelihood is None:
            log_l = 0
            for interferometer in self.interferometers:
                log_l -= 2. / self.waveform_generator.time_duration * np.sum(abs(interferometer.data) ** 2
                          / interferometer.power_spectral_density_array)
            self.__noise_log_likelihood = log_l.real
        else:
            self.__noise_log_likelihood = noise_log_likelihood

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
        return self.log_likelihood() - self.noise_log_likelihood


