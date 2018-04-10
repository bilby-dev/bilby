import numpy as np


class likelihood:
    def __init__(self, interferometers, source):
        self.interferometers = interferometers
        self.source = source
        self.parameter_keys = set(self.source.parameter_keys +
                                  ['ra', 'dec', 'geocent_time', 'psi'])
        self.noise_likelihood = 0
        self.set_null_likelihood()

    def loglikelihood(self, parameters):
        log_l = -self.set_null_likelihood()
        waveform_polarizations = self.source.frequency_domain_strain(parameters)
        for interferometer in self.interferometers:
            h = []
            for mode in waveform_polarizations:
                det_response = interferometer.antenna_response(
                    parameters['ra'], parameters['dec'],
                    parameters['geocent_time'], parameters['psi'], mode)

                h.append(waveform_polarizations[mode] * det_response)

            signal_ifo = np.sum(h, axis=0)

            time_shift = interferometer.time_delay_from_geocenter(
                parameters['ra'], parameters['dec'],
                parameters['geocent_time'])
            signal_ifo *= np.exp(-1j*2*np.pi*time_shift*self.source.frequency_array)

            log_l -= 4. / self.source.time_duration * np.vdot(
                interferometer.data - signal_ifo,
                (interferometer.data - signal_ifo) / (
                    interferometer.power_spectral_density_array))

        return log_l.real

    def set_null_likelihood(self):
        log_l = 0
        for interferometer in self.interferometers:
            log_l -= 4. / self.source.time_duration * np.sum(abs(interferometer.data)**2
                                                             / interferometer.power_spectral_density_array)
        self.noise_likelihood = log_l.real



class likelihoodB(likelihood):


    def __init__(self, interferometers, source):
        likelihood.__init__(self, interferometers, source)

        for interferometer in self.interferometers:
            interferometer.whiten_data()


    def loglikelihood(self, parameters):
        log_l = 0
        waveform_polarizations = self.source.frequency_domain_strain(parameters)
        for interferometer in self.interferometers:
            for mode in waveform_polarizations.keys():

                det_response = interferometer.antenna_response(
                    parameters['ra'], parameters['dec'],
                    parameters['geocent_time'], parameters['psi'], mode)

                waveform_polarizations[mode] *= det_response

            signal_ifo = np.sum(waveform_polarizations.values(), axis=0)

            time_shift = interferometer.time_delay_from_geocenter(
                parameters['ra'], parameters['dec'],
                parameters['geocent_time'])
            signal_ifo *= np.exp(-1j*2*np.pi*time_shift)
            signal_ifo_whitened = signal_ifo / (
                    interferometer.amplitude_spectral_density_array)

            log_l -= 4. * self.source.sampling_frequency * (
                np.real(sum(
                    (interferometer.whitened_data - signal_ifo_whitened)**2)))

        return log_l
