import numpy as np


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
