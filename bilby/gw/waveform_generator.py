import numpy as np

from ..core import utils
from ..core.series import CoupledTimeAndFrequencySeries
from ..core.utils import PropertyAccessor
from .conversion import convert_to_lal_binary_black_hole_parameters
from .utils import lalsim_GetApproximantFromString


class WaveformGenerator(object):
    """
    The base waveform generator class.

    Waveform generators provide a unified method to call disparate source models.
    """

    duration = PropertyAccessor('_times_and_frequencies', 'duration')
    sampling_frequency = PropertyAccessor('_times_and_frequencies', 'sampling_frequency')
    start_time = PropertyAccessor('_times_and_frequencies', 'start_time')
    frequency_array = PropertyAccessor('_times_and_frequencies', 'frequency_array')
    time_array = PropertyAccessor('_times_and_frequencies', 'time_array')

    def __init__(self, duration=None, sampling_frequency=None, start_time=0, frequency_domain_source_model=None,
                 time_domain_source_model=None, parameters=None,
                 parameter_conversion=None,
                 waveform_arguments=None):
        """
        The base waveform generator class.

        Parameters
        ==========
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
        waveform_arguments: dict, optional
            A dictionary of fixed keyword arguments to pass to either
            `frequency_domain_source_model` or `time_domain_source_model`.

            Note: the arguments of frequency_domain_source_model (except the first,
            which is the frequencies at which to compute the strain) will be added to
            the WaveformGenerator object and initialised to `None`.

        """
        self._times_and_frequencies = CoupledTimeAndFrequencySeries(duration=duration,
                                                                    sampling_frequency=sampling_frequency,
                                                                    start_time=start_time)
        self.frequency_domain_source_model = frequency_domain_source_model
        self.time_domain_source_model = time_domain_source_model
        self.source_parameter_keys = self.__parameters_from_source_model()
        if parameter_conversion is None:
            self.parameter_conversion = convert_to_lal_binary_black_hole_parameters
        else:
            self.parameter_conversion = parameter_conversion
        if waveform_arguments is not None:
            self.waveform_arguments = waveform_arguments
        else:
            self.waveform_arguments = dict()
        if isinstance(parameters, dict):
            self.parameters = parameters
        self._cache = dict(parameters=None, waveform=None, model=None)
        utils.logger.info(
            "Waveform generator initiated with\n"
            "  frequency_domain_source_model: {}\n"
            "  time_domain_source_model: {}\n"
            "  parameter_conversion: {}"
            .format(utils.get_function_path(self.frequency_domain_source_model),
                    utils.get_function_path(self.time_domain_source_model),
                    utils.get_function_path(self.parameter_conversion))
        )

    def __repr__(self):
        if self.frequency_domain_source_model is not None:
            fdsm_name = self.frequency_domain_source_model.__name__
        else:
            fdsm_name = None
        if self.time_domain_source_model is not None:
            tdsm_name = self.time_domain_source_model.__name__
        else:
            tdsm_name = None
        if self.parameter_conversion is None:
            param_conv_name = None
        else:
            param_conv_name = self.parameter_conversion.__name__

        return self.__class__.__name__ + '(duration={}, sampling_frequency={}, start_time={}, ' \
                                         'frequency_domain_source_model={}, time_domain_source_model={}, ' \
                                         'parameter_conversion={}, ' \
                                         'waveform_arguments={})'\
            .format(self.duration, self.sampling_frequency, self.start_time, fdsm_name, tdsm_name,
                    param_conv_name, self.waveform_arguments)

    def frequency_domain_strain(self, parameters=None):
        """ Wrapper to source_model.

        Converts self.parameters with self.parameter_conversion before handing it off to the source model.
        Automatically refers to the time_domain_source model via NFFT if no frequency_domain_source_model is given.

        Parameters
        ==========
        parameters: dict, optional
            Parameters to evaluate the waveform for, this overwrites
            `self.parameters`.
            If not provided will fall back to `self.parameters`.

        Returns
        =======
        array_like: The frequency domain strain for the given set of parameters

        Raises
        ======
        RuntimeError: If no source model is given

        """
        return self._calculate_strain(model=self.frequency_domain_source_model,
                                      model_data_points=self.frequency_array,
                                      parameters=parameters,
                                      transformation_function=utils.nfft,
                                      transformed_model=self.time_domain_source_model,
                                      transformed_model_data_points=self.time_array)

    def time_domain_strain(self, parameters=None):
        """ Wrapper to source_model.

        Converts self.parameters with self.parameter_conversion before handing it off to the source model.
        Automatically refers to the frequency_domain_source model via INFFT if no frequency_domain_source_model is
        given.

        Parameters
        ==========
        parameters: dict, optional
            Parameters to evaluate the waveform for, this overwrites
            `self.parameters`.
            If not provided will fall back to `self.parameters`.

        Returns
        =======
        array_like: The time domain strain for the given set of parameters

        Raises
        ======
        RuntimeError: If no source model is given

        """
        return self._calculate_strain(model=self.time_domain_source_model,
                                      model_data_points=self.time_array,
                                      parameters=parameters,
                                      transformation_function=utils.infft,
                                      transformed_model=self.frequency_domain_source_model,
                                      transformed_model_data_points=self.frequency_array)

    def _calculate_strain(self, model, model_data_points, transformation_function, transformed_model,
                          transformed_model_data_points, parameters):
        if parameters is not None:
            self.parameters = parameters
        if self.parameters == self._cache['parameters'] and self._cache['model'] == model and \
                self._cache['transformed_model'] == transformed_model:
            return self._cache['waveform']
        if model is not None:
            model_strain = self._strain_from_model(model_data_points, model)
        elif transformed_model is not None:
            model_strain = self._strain_from_transformed_model(transformed_model_data_points, transformed_model,
                                                               transformation_function)
        else:
            raise RuntimeError("No source model given")
        self._cache['waveform'] = model_strain
        self._cache['parameters'] = self.parameters.copy()
        self._cache['model'] = model
        self._cache['transformed_model'] = transformed_model
        return model_strain

    def _strain_from_model(self, model_data_points, model):
        return model(model_data_points, **self.parameters)

    def _strain_from_transformed_model(self, transformed_model_data_points, transformed_model, transformation_function):
        transformed_model_strain = self._strain_from_model(transformed_model_data_points, transformed_model)

        if isinstance(transformed_model_strain, np.ndarray):
            return transformation_function(transformed_model_strain, self.sampling_frequency)

        model_strain = dict()
        for key in transformed_model_strain:
            if transformation_function == utils.nfft:
                model_strain[key], _ = \
                    transformation_function(transformed_model_strain[key], self.sampling_frequency)
            else:
                model_strain[key] = transformation_function(transformed_model_strain[key], self.sampling_frequency)
        return model_strain

    @property
    def parameters(self):
        """ The dictionary of parameters for source model.

        Returns
        =======
        dict: The dictionary of parameter key-value pairs

        """
        return self.__parameters

    @parameters.setter
    def parameters(self, parameters):
        """
        Set parameters, this applies the conversion function and then removes
        any parameters which aren't required by the source function.

        (set.symmetric_difference is the opposite of set.intersection)

        Parameters
        ==========
        parameters: dict
            Input parameter dictionary, this is copied, passed to the conversion
            function and has self.waveform_arguments added to it.
        """
        if not isinstance(parameters, dict):
            raise TypeError('"parameters" must be a dictionary.')
        new_parameters = parameters.copy()
        new_parameters, _ = self.parameter_conversion(new_parameters)
        for key in self.source_parameter_keys.symmetric_difference(
                new_parameters):
            new_parameters.pop(key)
        self.__parameters = new_parameters
        self.__parameters.update(self.waveform_arguments)

    def __parameters_from_source_model(self):
        """
        Infer the named arguments of the source model.

        Returns
        =======
        set: The names of the arguments of the source model.
        """
        if self.frequency_domain_source_model is not None:
            model = self.frequency_domain_source_model
        elif self.time_domain_source_model is not None:
            model = self.time_domain_source_model
        else:
            raise AttributeError('Either time or frequency domain source '
                                 'model must be provided.')
        return set(utils.infer_parameters_from_function(model))


class LALCBCWaveformGenerator(WaveformGenerator):
    """ A waveform generator with specific checks for LAL CBC waveforms """
    LAL_SIM_INSPIRAL_SPINS_FLOW = 1

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.validate_reference_frequency()

    def validate_reference_frequency(self):
        from lalsimulation import SimInspiralGetSpinFreqFromApproximant
        waveform_approximant = self.waveform_arguments["waveform_approximant"]
        waveform_approximant_number = lalsim_GetApproximantFromString(waveform_approximant)
        if SimInspiralGetSpinFreqFromApproximant(waveform_approximant_number) == self.LAL_SIM_INSPIRAL_SPINS_FLOW:
            if self.waveform_arguments["reference_frequency"] != self.waveform_arguments["minimum_frequency"]:
                raise ValueError(f"For {waveform_approximant}, reference_frequency must equal minimum_frequency")
