import inspect

from . import utils
import numpy as np


class WaveformGenerator(object):
    """ A waveform generator

    Parameters
    ----------
    sampling_frequency: float
        The sampling frequency
    time_duration: float
        Time duration of data
    frequency_domain_source_model: func
        A python function taking some arguments and returning the frequency
        domain strain. Note the first argument must be the frequencies at
        which to compute the strain
    time_domain_source_model: func
        A python function taking some arguments and returning the time
        domain strain. Note the first argument must be the times at
        which to compute the strain
    parameters: dict
        Initial values for the parameters
    parameter_conversion: func
        Function to convert from sampled parameters to parameters of the
        waveform generator
    non_standard_sampling_parameter_keys: list
        List of parameter name for *non-standard* sampling parameters.

    Note: the arguments of frequency_domain_source_model (except the first,
    which is the frequencies at which to compute the strain) will be added to
    the WaveformGenerator object and initialised to `None`.

    """

    def __init__(self, time_duration, sampling_frequency, frequency_domain_source_model=None,
                 time_domain_source_model=None, parameters=None, parameter_conversion=None,
                 non_standard_sampling_parameter_keys=None):
        self.time_duration = time_duration
        self.sampling_frequency = sampling_frequency
        self.frequency_domain_source_model = frequency_domain_source_model
        self.time_domain_source_model = time_domain_source_model
        self.time_duration = time_duration
        self.sampling_frequency = sampling_frequency
        self.parameter_conversion = parameter_conversion
        self.non_standard_sampling_parameter_keys = non_standard_sampling_parameter_keys
        self.parameters = parameters
        self.__frequency_array_updated = False
        self.__time_array_updated = False

    def frequency_domain_strain(self):
        """ Wrapper to source_model """
        if self.parameter_conversion is not None:
            added_keys = self.parameter_conversion(self.parameters, self.non_standard_sampling_parameter_keys)

        if self.frequency_domain_source_model is not None:
            model_frequency_strain = self.frequency_domain_source_model(self.frequency_array, **self.parameters)
        elif self.time_domain_source_model is not None:
            model_frequency_strain = dict()
            time_domain_strain = self.time_domain_source_model(self.time_array, **self.parameters)
            if isinstance(time_domain_strain, np.ndarray):
                return utils.nfft(time_domain_strain, self.sampling_frequency)
            for key in time_domain_strain:
                model_frequency_strain[key], self.frequency_array = utils.nfft(time_domain_strain[key],
                                                                               self.sampling_frequency)
        else:
            raise RuntimeError("No source model given")
        if self.parameter_conversion is not None:
            for key in added_keys:
                self.parameters.pop(key)
        return model_frequency_strain

    def time_domain_strain(self):
        if self.parameter_conversion is not None:
            added_keys = self.parameter_conversion(self.parameters, self.non_standard_sampling_parameter_keys)
        if self.time_domain_source_model is not None:
            model_time_series = self.time_domain_source_model(self.time_array, **self.parameters)
        elif self.frequency_domain_source_model is not None:
            model_time_series = dict()
            frequency_domain_strain = self.frequency_domain_source_model(self.frequency_array, **self.parameters)
            if isinstance(frequency_domain_strain, np.ndarray):
                return utils.infft(frequency_domain_strain, self.sampling_frequency)
            for key in frequency_domain_strain:
                model_time_series[key] = utils.infft(frequency_domain_strain[key], self.sampling_frequency)
        else:
            raise RuntimeError("No source model given")
        if self.parameter_conversion is not None:
            for key in added_keys:
                self.parameters.pop(key)
        return model_time_series

    @property
    def frequency_array(self):
        if self.__frequency_array_updated is False:
            self.__frequency_array = utils.create_frequency_series(
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
        self.__parameters_from_source_model()
        if isinstance(parameters, dict):
            for key in parameters.keys():
                self.__parameters[key] = parameters[key]

    def __parameters_from_source_model(self):
        if self.frequency_domain_source_model is not None:
            parameters = inspect.getargspec(self.frequency_domain_source_model).args
            parameters.pop(0)
            self.__parameters = dict.fromkeys(parameters)
        elif self.time_domain_source_model is not None:
            parameters = inspect.getargspec(self.time_domain_source_model).args
            parameters.pop(0)
            self.__parameters = dict.fromkeys(parameters)

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
