import numpy as np


class Likelihood:
    def __init__(self, interferometers, waveformgenerator):
        self.interferometers = interferometers
        self.waveformgenerator = waveformgenerator
        self.noise_log_likelihood = 0
        self.set_noise_log_likelihood()

    def get_interferometer_signal(self, waveform_polarizations, interferometer):
        h = []
        for mode in waveform_polarizations:
            det_response = interferometer.antenna_response(
                self.waveformgenerator.ra, self.waveformgenerator.dec,
                self.waveformgenerator.geocent_time, self.waveformgenerator.psi, mode)
            h.append(waveform_polarizations[mode] * det_response)
        signal = np.sum(h, axis=0)

        time_shift = interferometer.time_delay_from_geocenter(
            self.waveformgenerator.ra, self.waveformgenerator.dec,
            self.waveformgenerator.geocent_time)
        signal *= np.exp(-1j * 2 * np.pi * time_shift * self.waveformgenerator.frequency_array)

        return signal

    def log_likelihood(self):
        log_l = 0
        waveform_polarizations = self.waveformgenerator.frequency_domain_strain()
        for interferometer in self.interferometers:
            log_l += self.log_likelihood_interferometer(waveform_polarizations, interferometer)
        return log_l.real

    def log_likelihood_interferometer(self, waveform_polarizations, interferometer):
        signal_ifo = self.get_interferometer_signal(waveform_polarizations, interferometer)

        log_l = - 4. / self.waveformgenerator.time_duration * np.vdot(interferometer.data - signal_ifo,
                                                                      (interferometer.data - signal_ifo)
                                                                      / interferometer.power_spectral_density_array)
        return log_l.real

    def log_likelihood_ratio(self):
        return self.log_likelihood() - self.noise_log_likelihood

    def set_noise_log_likelihood(self):
        log_l = 0
        for interferometer in self.interferometers:
            log_l -= 4. / self.waveformgenerator.time_duration * np.sum(abs(interferometer.data)**2
                                                                        / interferometer.power_spectral_density_array)
        self.noise_log_likelihood = log_l.real


class LikelihoodB(Likelihood):


    def __init__(self, interferometers, waveformgenerator):
        Likelihood.__init__(self, interferometers, waveformgenerator)

        for interferometer in self.interferometers:
            interferometer.whiten_data()


    def log_likelihood(self):
        log_l = 0
        waveform_polarizations = self.waveformgenerator.frequency_domain_strain()
        for interferometer in self.interferometers:
            for mode in waveform_polarizations.keys():

                det_response = interferometer.antenna_response(
                    self.waveformgenerator.ra, self.waveformgenerator.dec,
                    self.waveformgenerator.geocent_time, self.waveformgenerator.psi, mode)

                waveform_polarizations[mode] *= det_response

            signal_ifo = np.sum(waveform_polarizations.values(), axis=0)

            time_shift = interferometer.time_delay_from_geocenter(
                self.waveformgenerator.ra, self.waveformgenerator.dec,
                self.waveformgenerator.geocent_time)
            signal_ifo *= np.exp(-1j * 2 * np.pi * time_shift)
            signal_ifo_whitened = signal_ifo / (
                interferometer.amplitude_spectral_density_array)

            log_l -= 4. * self.waveformgenerator.sampling_frequency * (
                np.real(sum(
                    (interferometer.whitened_data - signal_ifo_whitened) ** 2)))

        return log_l
