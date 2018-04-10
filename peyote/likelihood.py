import numpy as np


class Likelihood:
    def __init__(self, interferometers, source):
        self.interferometers = interferometers
        self.source = source

    #        self.parameter_keys = set(self.source.parameter_keys +
    #                                  ['ra', 'dec', 'geocent_time', 'psi'])

    def log_likelihood(self):
        log_l = 0
        waveform_polarizations = self.source.frequency_domain_strain()
        for interferometer in self.interferometers:
            h = []
            for mode in waveform_polarizations:
                det_response = interferometer.antenna_response(self.source.ra, self.source.dec,
                                                               self.source.geocent_time,
                                                               self.source.psi, mode)

                h.append(waveform_polarizations[mode] * det_response)

            signal_ifo = np.sum(h, axis=0)

            time_shift = interferometer.time_delay_from_geocenter(self.source.ra, self.source.dec,
                                                                  self.source.geocent_time)

            signal_ifo *= np.exp(1j*2*np.pi*time_shift*self.source.frequency_array)

            log_l -= 4. / self.source.time_duration * np.vdot(
                interferometer.data - signal_ifo,
                (interferometer.data - signal_ifo) / (
                    interferometer.power_spectral_density_array))

        return log_l.real


class LikelihoodB(Likelihood):

    def __init__(self, interferometers, source):
        Likelihood.__init__(self, interferometers, source)

        for interferometer in self.interferometers:
            interferometer.whiten_data()

    def log_likelihood(self):
        log_l = 0
        waveform_polarizations = self.source.frequency_domain_strain()
        for interferometer in self.interferometers:
            for mode in waveform_polarizations.keys():
                det_response = interferometer.antenna_response(self.source.ra, self.source.dec,
                                                               self.source.geocent_time, self.source.psi, mode)
                waveform_polarizations[mode] *= det_response

            signal_ifo = np.sum(waveform_polarizations.values(), axis=0)

            time_shift = interferometer.time_delay_from_geocenter(self.source.ra, self.source.dec,
                                                                  self.source.geocent_time)
            signal_ifo *= np.exp(-1j * 2 * np.pi * time_shift)
            signal_ifo_whitened = signal_ifo / (
                interferometer.amplitude_spectral_density_array)

            log_l -= 4. * self.source.sampling_frequency * (
                np.real(sum(
                    (interferometer.whitened_data - signal_ifo_whitened) ** 2)))

        return log_l
