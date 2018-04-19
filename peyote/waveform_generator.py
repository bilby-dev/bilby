import inspect
import copy
import peyote


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
        keys = inspect.getargspec(source_model).args
        keys.pop(0)
        self.parameters = dict.fromkeys(keys)

    @property
    def frequency_array(self):
        return peyote.utils.create_fequency_series(self.sampling_frequency, self.time_duration)

    @property
    def time_array(self):
        return peyote.utils.create_time_series(self.sampling_frequency, self.time_duration)

    def frequency_domain_strain(self):
        """ Wrapper to source_model """
        return self.source_model(self.frequency_array, **self.parameters)

    def set_values(self, dictionary):
        """ Given a dictionary of values, set the class attributes """

        for key in self.parameters.keys():
            if key in dictionary.keys():
                self.parameters[key] = dictionary[key]
            else:
                raise KeyError('The provided dictionary did not contain key {}'.format(key))
