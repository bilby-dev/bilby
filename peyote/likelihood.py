import numpy as np


class Likelihood:
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

    def get_interferometer_signal(self, waveform_polarizations, interferometer):
        h = []
        for mode in waveform_polarizations:
            det_response = interferometer.antenna_response(
                self.waveform_generator.parameters['ra'],
                self.waveform_generator.parameters['dec'],
                self.waveform_generator.parameters['geocent_time'],
                self.waveform_generator.parameters['psi'], mode)
            h.append(waveform_polarizations[mode] * det_response)
        signal = np.sum(h, axis=0)

        time_shift = interferometer.time_delay_from_geocenter(
            self.waveform_generator.parameters['ra'],
            self.waveform_generator.parameters['dec'],
            self.waveform_generator.parameters['geocent_time'])
        signal = signal * np.exp(-1j * 2 * np.pi * time_shift * self.waveform_generator.frequency_array)

        return signal

    def log_likelihood(self):
        log_l = 0
        waveform_polarizations = self.waveform_generator.frequency_domain_strain()
        for interferometer in self.interferometers:
            log_l += self.log_likelihood_interferometer(waveform_polarizations, interferometer)
        return log_l.real

    def log_likelihood_interferometer(self, waveform_polarizations, interferometer):
        signal_ifo = self.get_interferometer_signal(waveform_polarizations, interferometer)

        log_l = - 2. / self.waveform_generator.time_duration * np.vdot(interferometer.data - signal_ifo,
                                                                       (interferometer.data - signal_ifo)
                                                                       / interferometer.power_spectral_density_array)
        return log_l.real

    def log_likelihood_ratio(self):
        return self.log_likelihood() - self.noise_log_likelihood



class LikelihoodB(Likelihood):

    def __init__(self, interferometers, waveform_generator):
        Likelihood.__init__(self, interferometers, waveform_generator)

        for interferometer in self.interferometers:
            interferometer.whiten_data()

    def log_likelihood(self):
        log_l = 0
        waveform_polarizations = self.waveform_generator.frequency_domain_strain()
        for interferometer in self.interferometers:
            for mode in waveform_polarizations.keys():
                det_response = interferometer.antenna_response(
                    self.waveform_generator.ra, self.waveform_generator.dec,
                    self.waveform_generator.geocent_time, self.waveform_generator.psi, mode)

                waveform_polarizations[mode] *= det_response

            signal_ifo = np.sum(waveform_polarizations.values(), axis=0)

            time_shift = interferometer.time_delay_from_geocenter(
                self.waveform_generator.ra, self.waveform_generator.dec,
                self.waveform_generator.geocent_time)
            signal_ifo *= np.exp(-1j * 2 * np.pi * time_shift)
            signal_ifo_whitened = signal_ifo / (
                interferometer.amplitude_spectral_density_array)

            log_l -= 2. * self.waveform_generator.sampling_frequency * (
                np.real(sum(
                    (interferometer.whitened_data - signal_ifo_whitened) ** 2)))

        return log_l
