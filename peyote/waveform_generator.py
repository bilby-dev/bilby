import inspect

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

    def __init__(self, name, sampling_frequency=4096, time_duration=1, source_model=None):
        self.name = name
        self.time_duration = time_duration
        self.sampling_frequency = sampling_frequency
        self.source_model = source_model

    @property
    def frequency_array(self):
        return peyote.utils.create_fequency_series(self.sampling_frequency, self.time_duration)

    @property
    def time_array(self):
        return peyote.utils.create_time_series(self.sampling_frequency, self.time_duration)

    @property
    def parameter_keys(self):
        self.__parameter_keys = inspect.getargspec(self.source_model).args
        self.__parameter_keys.pop(0)
        for a in self.__parameter_keys:
            if hasattr(self, a):
                continue
            setattr(self, a, None)
        return self.__parameter_keys

    def frequency_domain_strain(self):
        """ Wrapper to source_model """
        kwargs = {k: self.__dict__[k] for k in self.parameter_keys}
        return self.source_model(self.frequency_array, **kwargs)

    def set_values(self, dictionary):
        """ Given a dictionary of values, set the class attributes """
        for k in self.parameter_keys:
            try:
                setattr(self, k, dictionary[k])
            except KeyError:
                raise KeyError(
                    'The provided dictionary did not contain key {}'.format(k))
