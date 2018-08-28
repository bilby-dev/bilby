from tupak.core import utils
import numpy as np


class WaveformGenerator(object):

    def __init__(self, duration=None, sampling_frequency=None, start_time=0, frequency_domain_source_model=None,
                 time_domain_source_model=None, parameters=None,
                 parameter_conversion=lambda parameters, search_keys: (parameters, []),
                 non_standard_sampling_parameter_keys=None,
                 waveform_arguments=None):
        """ A waveform generator

    Parameters
    ----------
    sampling_frequency: float, optional
        The sampling frequency
    duration: float, optional
        Time duration of data
    start_time: float, optional
        Starting time of the time array
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
        waveform generator. Default value is the identity, i.e. it leaves
        the parameters unaffected.
    non_standard_sampling_parameter_keys: list, optional
        List of parameter name for *non-standard* sampling parameters.
    waveform_arguments: dict, optional
        A dictionary of fixed keyword arguments to pass to either
        `frequency_domain_source_model` or `time_domain_source_model`.

        Note: the arguments of frequency_domain_source_model (except the first,
        which is the frequencies at which to compute the strain) will be added to
        the WaveformGenerator object and initialised to `None`.

        """
        self.duration = duration
        self.sampling_frequency = sampling_frequency
        self.start_time = start_time
        self.frequency_domain_source_model = frequency_domain_source_model
        self.time_domain_source_model = time_domain_source_model
        self.__parameters_from_source_model()
        self.duration = duration
        self.sampling_frequency = sampling_frequency
        self.parameter_conversion = parameter_conversion
        self.non_standard_sampling_parameter_keys = non_standard_sampling_parameter_keys
        self.parameters = parameters
        if waveform_arguments is not None:
            self.waveform_arguments = waveform_arguments
        else:
            self.waveform_arguments = dict()
        self.__frequency_array_updated = False
        self.__time_array_updated = False
        self.__full_source_model_keyword_arguments = {}
        self.__full_source_model_keyword_arguments.update(self.waveform_arguments)
        self.__full_source_model_keyword_arguments.update(self.parameters)
        self.__added_keys = []

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
        return self._calculate_strain(model=self.frequency_domain_source_model,
                                      model_data_points=self.frequency_array,
                                      transformation_function=utils.nfft,
                                      transformed_model=self.time_domain_source_model,
                                      transformed_model_data_points=self.time_array)

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
        return self._calculate_strain(model=self.time_domain_source_model,
                                      model_data_points=self.time_array,
                                      transformation_function=utils.infft,
                                      transformed_model=self.frequency_domain_source_model,
                                      transformed_model_data_points=self.frequency_array)

    def _calculate_strain(self, model, model_data_points, transformation_function, transformed_model,
                          transformed_model_data_points):
        self._apply_parameter_conversion()
        if model is not None:
            model_strain = self._strain_from_model(model_data_points, model)
        elif transformed_model is not None:
            model_strain = self._strain_from_transformed_model(transformed_model_data_points, transformed_model,
                                                               transformation_function)
        else:
            raise RuntimeError("No source model given")
        self._remove_added_keys()
        return model_strain

    def _apply_parameter_conversion(self):
        self.parameters, self.__added_keys = self.parameter_conversion(self.parameters,
                                                                       self.non_standard_sampling_parameter_keys)
        self.__full_source_model_keyword_arguments.update(self.parameters)

    def _strain_from_model(self, model_data_points, model):
        return model(model_data_points, **self.__full_source_model_keyword_arguments)

    def _strain_from_transformed_model(self, transformed_model_data_points, transformed_model, transformation_function):
        transformed_model_strain = self._strain_from_model(transformed_model_data_points, transformed_model)

        if isinstance(transformed_model_strain, np.ndarray):
            return transformation_function(transformed_model_strain, self.sampling_frequency)

        model_strain = dict()
        for key in transformed_model_strain:
            if transformation_function == utils.nfft:
                model_strain[key], self.frequency_array = \
                    transformation_function(transformed_model_strain[key], self.sampling_frequency)
            else:
                model_strain[key] = transformation_function(transformed_model_strain[key], self.sampling_frequency)
        return model_strain

    def _remove_added_keys(self):
        for key in self.__added_keys:
            self.parameters.pop(key)

    @property
    def frequency_array(self):
        """ Frequency array for the waveforms. Automatically updates if sampling_frequency or duration are updated.

        Returns
        -------
        array_like: The frequency array
        """
        if self.__frequency_array_updated is False:
            self.frequency_array = utils.create_frequency_series(
                self.sampling_frequency,
                self.duration)
        return self.__frequency_array

    @frequency_array.setter
    def frequency_array(self, frequency_array):
        self.__frequency_array = frequency_array
        self.__frequency_array_updated = True

    @property
    def time_array(self):
        """ Time array for the waveforms. Automatically updates if sampling_frequency or duration are updated.

        Returns
        -------
        array_like: The time array
        """

        if self.__time_array_updated is False:
            self.__time_array = utils.create_time_series(
                self.sampling_frequency,
                self.duration,
                self.start_time)

            self.__time_array_updated = True
        return self.__time_array

    @time_array.setter
    def time_array(self, time_array):
        self.__time_array = time_array
        self.__time_array_updated = True

    @property
    def parameters(self):
        """ The dictionary of parameters for source model.

        Returns
        -------
        dict: The dictionary of parameter key-value pairs

        """
        return self.__parameters

    @parameters.setter
    def parameters(self, parameters):
        if isinstance(parameters, dict):
            for key in parameters.keys():
                self.__parameters[key] = parameters[key]

    def __parameters_from_source_model(self):
        if self.frequency_domain_source_model is not None:
            self.__parameters = dict.fromkeys(utils.infer_parameters_from_function(self.frequency_domain_source_model))
        elif self.time_domain_source_model is not None:
            self.__parameters = dict.fromkeys(utils.infer_parameters_from_function(self.time_domain_source_model))

    @property
    def duration(self):
        """ Allows one to set the time duration and automatically updates the frequency and time array.

        Returns
        -------
        float: The time duration.

        """
        return self.__duration

    @duration.setter
    def duration(self, duration):
        self.__duration = duration
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

    @property
    def start_time(self):
        return self.__start_time

    @start_time.setter
    def start_time(self, starting_time):
        self.__start_time = starting_time
        self.__time_array_updated = False
