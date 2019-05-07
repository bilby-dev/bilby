""" Functions for adding calibration factors to waveform templates.
"""

import numpy as np
from scipy.interpolate import interp1d


class Recalibrate(object):

    name = 'none'

    def __init__(self, prefix='recalib_'):
        """
        Base calibration object. This applies no transformation

        Parameters
        ----------
        prefix: str
            Prefix on parameters relating to the calibration.
        """
        self.params = dict()
        self.prefix = prefix

    def __repr__(self):
        return self.__class__.__name__ + '(prefix=\'{}\')'.format(self.prefix)

    def get_calibration_factor(self, frequency_array, **params):
        """Apply calibration model

        This method should be overwritten by subclasses

        Parameters
        ----------
        frequency_array: array-like
            The frequency values to calculate the calibration factor for.
        params : dict
            Dictionary of sampling parameters which includes
            calibration parameters.

        Returns
        -------
        calibration_factor : array-like
            The factor to multiply the strain by.
        """
        self.set_calibration_parameters(**params)
        return np.ones_like(frequency_array)

    def set_calibration_parameters(self, **params):
        self.params.update({key[len(self.prefix):]: params[key] for key in params
                            if self.prefix in key})

    def __eq__(self, other):
        return self.__dict__ == other.__dict__


class CubicSpline(Recalibrate):

    name = 'cubic_spline'

    def __init__(self, prefix, minimum_frequency, maximum_frequency, n_points):
        """
        Cubic spline recalibration

        see https://dcc.ligo.org/DocDB/0116/T1400682/001/calnote.pdf

        This assumes the spline points follow
        np.logspace(np.log(minimum_frequency), np.log(maximum_frequency), n_points)

        Parameters
        ----------
        prefix: str
            Prefix on parameters relating to the calibration.
        minimum_frequency: float
            minimum frequency of spline points
        maximum_frequency: float
            maximum frequency of spline points
        n_points: int
            number of spline points
        """
        Recalibrate.__init__(self, prefix=prefix)
        if n_points < 4:
            raise ValueError('Cubic spline calibration requires at least 4 spline nodes.')
        self.n_points = n_points
        self.minimum_frequency = minimum_frequency
        self.maximum_frequency = maximum_frequency
        self._log_spline_points = np.linspace(
            np.log10(minimum_frequency), np.log10(maximum_frequency), n_points)

    @property
    def log_spline_points(self):
        return self._log_spline_points

    def __repr__(self):
        return self.__class__.__name__ + '(prefix=\'{}\', minimum_frequency={}, maximum_frequency={}, n_points={})'\
            .format(self.prefix, self.minimum_frequency, self.maximum_frequency, self.n_points)

    def get_calibration_factor(self, frequency_array, **params):
        """Apply calibration model

        Parameters
        ----------
        frequency_array: array-like
            The frequency values to calculate the calibration factor for.
        prefix: str
            Prefix for calibration parameter names
        params : dict
            Dictionary of sampling parameters which includes
            calibration parameters.

        Returns
        -------
        calibration_factor : array-like
            The factor to multiply the strain by.
        """
        self.set_calibration_parameters(**params)
        amplitude_parameters = [self.params['amplitude_{}'.format(ii)]
                                for ii in range(self.n_points)]
        delta_amplitude = interp1d(
            self.log_spline_points, amplitude_parameters, kind='cubic',
            bounds_error=False, fill_value=0)(np.log10(frequency_array))

        phase_parameters = [
            self.params['phase_{}'.format(ii)] for ii in range(self.n_points)]
        delta_phase = interp1d(
            self.log_spline_points, phase_parameters, kind='cubic',
            bounds_error=False, fill_value=0)(np.log10(frequency_array))

        calibration_factor = (1 + delta_amplitude) * (2 + 1j * delta_phase) / (2 - 1j * delta_phase)

        return calibration_factor
