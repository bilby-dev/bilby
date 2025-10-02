import numpy as np

from ...core.likelihood import Likelihood, _fallback_to_parameters


class BasicGravitationalWaveTransient(Likelihood):

    def __init__(self, interferometers, waveform_generator):
        """

        A likelihood object, able to compute the likelihood of the data given
        some model parameters

        The simplest frequency-domain gravitational wave transient likelihood. Does
        not include distance/phase marginalization.


        Parameters
        ==========
        interferometers: list
            A list of `bilby.gw.detector.Interferometer` instances - contains the
            detector data and power spectral densities
        waveform_generator: bilby.gw.waveform_generator.WaveformGenerator
            An object which computes the frequency-domain strain of the signal,
            given some set of parameters

        """
        super(BasicGravitationalWaveTransient, self).__init__(dict())
        self.interferometers = interferometers
        self.waveform_generator = waveform_generator

    def __repr__(self):
        return self.__class__.__name__ + '(interferometers={},\n\twaveform_generator={})' \
            .format(self.interferometers, self.waveform_generator)

    def noise_log_likelihood(self):
        """ Calculates the real part of noise log-likelihood

        Returns
        =======
        float: The real part of the noise log likelihood

        """
        log_l = 0
        for interferometer in self.interferometers:
            log_l -= 2. / self.waveform_generator.duration * (
                abs(interferometer.frequency_domain_strain) ** 2
                / interferometer.power_spectral_density_array
            ).sum()
        return log_l

    def log_likelihood(self, parameters=None):
        """ Calculates the real part of log-likelihood value

        Returns
        =======
        float: The real part of the log likelihood

        """
        parameters = _fallback_to_parameters(self, parameters)
        log_l = 0
        waveform_polarizations = \
            self.waveform_generator.frequency_domain_strain(parameters)
        if waveform_polarizations is None:
            return np.nan_to_num(-np.inf)
        for interferometer in self.interferometers:
            log_l += self.log_likelihood_interferometer(
                waveform_polarizations, interferometer)
        return log_l.real

    def log_likelihood_interferometer(self, waveform_polarizations,
                                      interferometer, parameters=None):
        """

        Parameters
        ==========
        waveform_polarizations: dict
            Dictionary containing the desired waveform polarization modes and the related strain
        interferometer: bilby.gw.detector.Interferometer
            The Interferometer object we want to have the log-likelihood for

        Returns
        =======
        float: The real part of the log-likelihood for this interferometer

        """
        parameters = _fallback_to_parameters(self, parameters)
        signal_ifo = interferometer.get_detector_response(
            waveform_polarizations, parameters)

        residual = interferometer.frequency_domain_strain - signal_ifo

        log_l = - 2. / self.waveform_generator.duration * (
            abs(residual)**2 / interferometer.power_spectral_density_array
        ).sum()
        return log_l.real
