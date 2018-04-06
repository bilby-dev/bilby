import numpy as np


class likelihood:
    def __init__(self, interferometers, source):
        self.interferometers = interferometers
        self.source = source
        self.parameter_keys = ['ra', 'dec', 'geocent_time', 'psi']
        self.parameter_keys += self.source.parameter_keys

    def loglikelihood(self, parameters):
        log_l = 0
        waveform_polarizations = self.source.frequency_domain_strain(parameters)
        for interferometer in self.interferometers:
            for mode in waveform_polarizations:
                det_response = interferometer.antenna_response(
                    parameters['ra'], parameters['dec'],
                    parameters['geocent_time'], parameters['psi'], mode)

                waveform_polarizations[mode] *= det_response

            signal_ifo = np.sum(waveform_polarizations.values(), axis=0)

            time_shift = interferometer.time_delay_from_geocenter(
                parameters['ra'], parameters['dec'],
                parameters['geocent_time'])
            signal_ifo *= np.exp(-1j*2*np.pi*time_shift)

            log_l -= 4. * self.source.sampling_frequency * np.vdot(
                interferometer.data - signal_ifo,
                (interferometer.data - signal_ifo) / (
                    interferometer.power_spectral_density_array))

        return log_l.real


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
