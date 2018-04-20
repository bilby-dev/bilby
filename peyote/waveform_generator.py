import inspect

from . import utils
from . import parameter

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

    def __init__(self, source_model, sampling_frequency=4096, time_duration=1,
                 parameters=None):
        self.time_duration = time_duration
        self.sampling_frequency = sampling_frequency
        self.source_model = source_model
        self.parameters = parameters
        self.__frequency_array_updated = False
        self.__time_array_updated = False

    def frequency_domain_strain(self):
        """ Wrapper to source_model """
        parameters = dict()
        for key in self.parameters.keys():
            if isinstance(self.parameters[key], parameter.Parameter):
                parameters[key] = self.parameters[key].value
            else:
                parameters[key] = self.parameters[key]
        return self.source_model(self.frequency_array, **parameters)

    @property
    def frequency_array(self):
        if self.__frequency_array_updated:
            return self.__frequency_array
        else:
            self.__frequency_array = utils.create_fequency_series(
                                        self.sampling_frequency,
                                        self.time_duration)
            self.__frequency_array_updated = True
        return self.__frequency_array

    @property
    def time_array(self):
        if self.__time_array_updated:
            return self.__time_array
        else:
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
            parameters = inspect.getargspec(self.source_model).args
            parameters.pop(0)
            self.__parameters = dict.fromkeys(parameters)
        elif isinstance(parameters, list):
            parameters.pop(0)
            self.__parameters = dict.fromkeys(parameters)
        elif isinstance(parameters, dict):
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
    def source_model(self):
        return self.__source_model

    @source_model.setter
    def source_model(self, source_model):
        self.__source_model = source_model
        self.parameters = inspect.getargspec(source_model).args

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
