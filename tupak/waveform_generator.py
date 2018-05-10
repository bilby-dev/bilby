import inspect

from . import utils
import numpy as np

class WaveformGenerator(object):
    """ A waveform generator

    Parameters
    ----------
    sampling_frequency: float
        The sampling frequency to sample at
    time_duration: float
        Time duration of data
    source_model: func
        A python function taking some arguments and returning the frequency
        domain strain. Note the first argument must be the frequencies at
        which to compute the strain

    Note: the arguments of source_model (except the first, which is the
    frequencies at which to compute the strain) will be added to the
    WaveformGenerator object and initialised to `None`.

    """

    def __init__(self, frequency_domain_source_model=None, time_domain_source_model=None, sampling_frequency=4096, time_duration=1,
                 parameters=None):
        self.time_duration = time_duration
        self.sampling_frequency = sampling_frequency
        self.frequency_domain_source_model = frequency_domain_source_model
        self.time_domain_source_model = time_domain_source_model
        self.parameters = parameters
        self.__frequency_array_updated = False
        self.__time_array_updated = False

    def frequency_domain_strain(self):
        """ Wrapper to source_model """
        if self.frequency_domain_source_model is not None:
            return self.frequency_domain_source_model(self.frequency_array, **self.parameters)
        elif self.time_domain_source_model is not None:
            fft_data = dict()
            time_domain_strain = self.time_domain_source_model(self.time_array, **self.parameters)
            if isinstance(time_domain_strain, np.ndarray):
                return time_domain_strain
            for key in time_domain_strain:
                fft_data[key], self.frequency_array = utils.nfft(time_domain_strain[key], self.sampling_frequency)
            return fft_data
        else:
            raise RuntimeError("No source model given")

    def time_domain_strain(self):
        if self.time_domain_source_model is not None:
            return self.time_domain_source_model(self.time_array, **self.parameters)
        elif self.frequency_domain_source_model is not None:
            ifft_data = dict()
            frequency_domain_strain = self.frequency_domain_source_model(self.frequency_array, **self.parameters)
            if isinstance(frequency_domain_strain, np.ndarray):
                return frequency_domain_strain
            for key in frequency_domain_strain:
                ifft_data = utils.infft(frequency_domain_strain[key], self.sampling_frequency)
            return ifft_data
        else:
            raise RuntimeError("No source model given")

    @property
    def frequency_array(self):
        if self.__frequency_array_updated is False:
            self.__frequency_array = utils.create_fequency_series(
                                        self.sampling_frequency,
                                        self.time_duration)
            self.__frequency_array_updated = True
        return self.__frequency_array

    @frequency_array.setter
    def frequency_array(self, frequency_array):
        self.__frequency_array = frequency_array

    @property
    def time_array(self):
        if self.__time_array_updated is False:
            self.__time_array = utils.create_time_series(
                                        self.sampling_frequency,
                                        self.time_duration)

            self.__time_array_updated = True
        return self.__time_array

    @property
    def parameters(self):
        return self.__parameters

    @parameters.setter
    def parameters(self, parameters):
        if parameters is None:
            if self.frequency_domain_source_model is not None:
                parameters = inspect.getargspec(self.frequency_domain_source_model).args
                parameters.pop(0)
                self.__parameters = dict.fromkeys(parameters)
            elif self.time_domain_source_model is not None:
                parameters = inspect.getargspec(self.time_domain_source_model).args
                parameters.pop(0)
                self.__parameters = dict.fromkeys(parameters)
        elif isinstance(parameters, list):
            parameters.pop(0)
            self.__parameters = dict.fromkeys(parameters)
        elif isinstance(parameters, dict):
            if not hasattr(self, '_WaveformGenerator__parameters'):
                self.__parameters = parameters
            for key in self.__parameters.keys():

                if key in parameters.keys():
                    self.__parameters[key] = parameters[key]
                else:
                    raise KeyError('The provided dictionary did not '
                                   'contain key {}'.format(key))
        else:
            raise TypeError('Parameters must either be set as a list of keys or'
                            ' a dictionary of key-value pairs.')

    @property
    def time_duration(self):
        return self.__time_duration

    @time_duration.setter
    def time_duration(self, time_duration):
        self.__time_duration = time_duration
        self.__frequency_array_updated = False
        self.__time_array_updated = False

    @property
    def sampling_frequency(self):
        return self.__sampling_frequency

    @sampling_frequency.setter
    def sampling_frequency(self, sampling_frequency):
        self.__sampling_frequency = sampling_frequency
        self.__frequency_array_updated = False
        self.__time_array_updated = False
