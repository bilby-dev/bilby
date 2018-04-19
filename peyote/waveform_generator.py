import inspect

from . import utils

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

    def __init__(self, source_model, sampling_frequency=4096, time_duration=1):
        self.time_duration = time_duration
        self.sampling_frequency = sampling_frequency
        self.source_model = source_model
        self.parameters = inspect.getargspec(source_model).args

    @property
    def frequency_array(self):
        return utils.create_fequency_series(self.sampling_frequency, self.time_duration)

    @property
    def time_array(self):
        return utils.create_time_series(self.sampling_frequency, self.time_duration)

    @property
    def parameters(self):
        return self.__parameters

    @property
    def source_model(self):
        return self.__source_model

    @parameters.setter
    def parameters(self, parameters):
        if isinstance(parameters, list):
            parameters.pop(0)
            self.__parameters = dict.fromkeys(parameters)
        elif isinstance(parameters, dict):
            for key in self.__parameters.keys():
                if key in parameters.keys():
                    self.__parameters[key] = parameters[key]
                else:
                    raise KeyError('The provided dictionary did not contain key {}'.format(key))
        else:
            raise TypeError('Parameters must either be set as a list of keys or a dictionary of key-value pairs.')

    @source_model.setter
    def source_model(self, source_model):
        self.__source_model = source_model
        self.parameters = inspect.getargspec(source_model).args

    def frequency_domain_strain(self):
        """ Wrapper to source_model """
        return self.source_model(self.frequency_array, **self.parameters)

