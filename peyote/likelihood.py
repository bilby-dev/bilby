import numpy as np
import pdb


class likelihood:
    def __init__(self, Interferometers, source):
        self.Interferometers = Interferometers
        self.source = source

    def logL(self, times, params):
        logL = 0
        waveform_polarizations = self.source.model(times, params)
        for Interferometer in self.Interferometers:
            for mode in params['modes']:

                det_response = Interferometer.antenna_response(
                                params['ra'], params['dec'],
                                params['geocent_time'], params['psi'],
                                mode)

                waveform_polarizations[mode] *= det_response

            signal_IFO = np.sum(waveform_polarizations.values())

            #time_shift = Interferometer.time_shift(source.params['geocent_time'])
            #signal *= np.exp(-1j*2*np.pi*time_shift)
            # This is just here as a reminder that a tc shift needs to be performed
            # on frequency-domain GWs

            logL += 4. * params['deltaF'] * np.vdot(
                Interferometer.data - signal_IFO,
                (Interferometer.data - signal_IFO) / Interferometer.psd)

        return logL


def logl_gravitational_wave(params, source, interferometers):
    """
    Calculate the log likelihood of a given gravitational-wave source given some strain data.

    :param sampling_frequency: data sampling frequency
    :param params: dictionary of parameter values describing the source
    :param source: Source object which contains a model
    :param interferometers: Interferometer classes with PSDs and data
    :return: log likelihood of the strain given the model and parameters
    """
    log_l = 0
    waveform_polarizations = source.frequency_domain_strain(params)
    for interferometer in interferometers:
        for mode in waveform_polarizations.keys():

            det_response = interferometer.antenna_response(params['ra'], params['dec'], params['geocent_time'],
                                                           params['psi'], mode)

            waveform_polarizations[mode] *= det_response

        signal_ifo = np.sum(waveform_polarizations.values(), axis=0)

        time_shift = interferometer.time_delay_from_geocenter(params['ra'], params['dec'], params['geocent_time'])
        signal_ifo *= np.exp(-1j*2*np.pi*time_shift)
        signal_ifo_whitened = signal_ifo / interferometer.amplitude_spectral_density_array

        log_l -= 4. * source.sampling_frequency * np.real(sum((interferometer.whitened_data - signal_ifo_whitened)**2))

    return log_l
