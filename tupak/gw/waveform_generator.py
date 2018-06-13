import inspect

from tupak.core import utils
import numpy as np


class WaveformGenerator(object):

    def __init__(self, time_duration, sampling_frequency, frequency_domain_source_model=None,
                 time_domain_source_model=None, parameters=None, parameter_conversion=None,
                 non_standard_sampling_parameter_keys=None):
        """ A waveform generator

        Parameters
        ----------
        sampling_frequency: float
            The sampling frequency
        time_duration: float
            Time duration of data
        frequency_domain_source_model: func, optional
            A python function taking some arguments and returning the frequency
            domain strain. Note the first argument must be the frequencies at
            which to compute the strain
        time_domain_source_model: func, optional
            A python function taking some arguments and returning the time
            domain strain. Note the first argument must be the times at
            which to compute the strain
        parameters: dict, optional
            Initial values for the parameters
        parameter_conversion: func, optional
            Function to convert from sampled parameters to parameters of the
            waveform generator
        non_standard_sampling_parameter_keys: list, optional
            List of parameter name for *non-standard* sampling parameters.

        Note: the arguments of frequency_domain_source_model (except the first,
        which is the frequencies at which to compute the strain) will be added to
        the WaveformGenerator object and initialised to `None`.

        """
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
        """ Rapper to source_model.

        Converts self.parameters with self.parameter_conversion before handing it off to the source model.
        Automatically refers to the time_domain_source model via NFFT if no frequency_domain_source_model is given.

        Returns
        -------
        array_like: The frequency domain strain for the given set of parameters

        Raises
        -------
        RuntimeError: If no source model is given

        """
        added_keys = []
        if self.parameter_conversion is not None:
            self.parameters, added_keys = self.parameter_conversion(self.parameters,
                                                                    self.non_standard_sampling_parameter_keys)

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

        for key in added_keys:
            self.parameters.pop(key)
        return model_frequency_strain

    def time_domain_strain(self):
        """ Rapper to source_model.

        Converts self.parameters with self.parameter_conversion before handing it off to the source model.
        Automatically refers to the frequency_domain_source model via INFFT if no frequency_domain_source_model is
        given.

        Returns
        -------
        array_like: The time domain strain for the given set of parameters

        Raises
        -------
        RuntimeError: If no source model is given

        """
        added_keys = []
        if self.parameter_conversion is not None:
            self.parameters, added_keys = self.parameter_conversion(self.parameters,
                                                                    self.non_standard_sampling_parameter_keys)
        if self.time_domain_source_model is not None:
            model_time_strain = self.time_domain_source_model(self.time_array, **self.parameters)
        elif self.frequency_domain_source_model is not None:
            model_time_strain = dict()
            frequency_domain_strain = self.frequency_domain_source_model(self.frequency_array, **self.parameters)
            if isinstance(frequency_domain_strain, np.ndarray):
                return utils.infft(frequency_domain_strain, self.sampling_frequency)
            for key in frequency_domain_strain:
                model_time_strain[key] = utils.infft(frequency_domain_strain[key], self.sampling_frequency)
        else:
            raise RuntimeError("No source model given")

        for key in added_keys:
            self.parameters.pop(key)
        return model_time_strain

    @property
    def frequency_array(self):
        """ Frequency array for the waveforms. Automatically updates if sampling_frequency or time_duration are updated.

        Returns
        -------
        array_like: The frequency array
        """
        if self.__frequency_array_updated is False:
            self.frequency_array = utils.create_frequency_series(
                                        self.sampling_frequency,
                                        self.time_duration)
        return self.__frequency_array

    @frequency_array.setter
    def frequency_array(self, frequency_array):
        self.__frequency_array = frequency_array
        self.__frequency_array_updated = True

    @property
    def time_array(self):
        """ Time array for the waveforms. Automatically updates if sampling_frequency or time_duration are updated.

        Returns
        -------
        array_like: The time array
        """

        if self.__time_array_updated is False:
            self.__time_array = utils.create_time_series(
                                        self.sampling_frequency,
                                        self.time_duration)

            self.__time_array_updated = True
        return self.__time_array

    @property
    def parameters(self):
        """ The dictionary of parameters for source model.

        Does some introspection into the source_model to figure out the parameters if none are given.

        Returns
        -------
        dict: The dictionary of parameter key-value pairs

        """
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
        """ Allows one to set the time duration and automatically updates the frequency and time array.

        Returns
        -------
        float: The time duration.

        """
        return self.__time_duration

    @time_duration.setter
    def time_duration(self, time_duration):
        self.__time_duration = time_duration
        self.__frequency_array_updated = False
        self.__time_array_updated = False

    @property
    def sampling_frequency(self):
        """ Allows one to set the sampling frequency and automatically updates the frequency and time array.

        Returns
        -------
        float: The sampling frequency.

        """
        return self.__sampling_frequency

    @sampling_frequency.setter
    def sampling_frequency(self, sampling_frequency):
        self.__sampling_frequency = sampling_frequency
        self.__frequency_array_updated = False
        self.__time_array_updated = False
