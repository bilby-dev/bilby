import numpy as np


class likelihood:
    def __init__(self, interferometers, source):
        self.interferometers = interferometers
        self.source = source

    def logl(self, parameters):
        logL = 0
        waveform_polarizations = self.source.frequency_domain_strain(parameters)
        for interferometer in self.interferometers:
            for mode in waveform_polarizations:
                det_response = interferometer.antenna_response(
                                parameters['ra'], parameters['dec'],
                                parameters['geocent_time'],
                                parameters['psi'], mode)

                waveform_polarizations[mode] *= det_response

            signal_IFO = np.sum(waveform_polarizations.values(), axis=0)

            #time_shift = interferometer.time_shift(source.params['geocent_time'])
            #signal *= np.exp(-1j*2*np.pi*time_shift)
            # This is just here as a reminder that a tc shift needs to be performed
            # on frequency-domain GWs

            logL -= 4. * self.source.sampling_frequency * np.vdot(
                interferometer.data - signal_IFO,
                (interferometer.data - signal_IFO) / interferometer.power_spectral_density_array)

        return logL.real


class gravitational_wave_likelihood:
    def __init__(self, source, interferometers):
        self.source = source
        self.interferometers = interferometers

    def logl_gravitational_wave(self, params):
        """
        Calculate the log likelihood of a given gravitational-wave source given some strain data.

        :param sampling_frequency: data sampling frequency
        :param params: dictionary of parameter values describing the source
        :param source: Source object which contains a model
        :param interferometers: Interferometer classes with PSDs and data
        :return: log likelihood of the strain given the model and parameters
        """
        log_l = 0
        waveform_polarizations = self.source.frequency_domain_strain(params)
        for interferometer in self.interferometers:
            for mode in waveform_polarizations.keys():

                det_response = interferometer.antenna_response(params['ra'], params['dec'], params['geocent_time'],
                                                               params['psi'], mode)

                waveform_polarizations[mode] *= det_response

            signal_ifo = np.sum(waveform_polarizations.values(), axis=0)

            time_shift = interferometer.time_delay_from_geocenter(params['ra'], params['dec'], params['geocent_time'])
            signal_ifo *= np.exp(-1j*2*np.pi*time_shift)
            signal_ifo_whitened = signal_ifo / interferometer.amplitude_spectral_density_array

            log_l -= 4. * self.source.sampling_frequency * np.real(sum((interferometer.whitened_data - signal_ifo_whitened)**2))

        return log_l
