import numpy as np

from ..core import utils
from ..core.series import CoupledTimeAndFrequencySeries
from ..core.utils import PropertyAccessor
from ..core.utils import logger
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
        self.source_parameter_keys = self._parameters_from_source_model()
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
        logger.info(f"Waveform generator instantiated: {self}")

    def __repr__(self):
        if self.frequency_domain_source_model is not None:
            fdsm_name = utils.get_function_path(self.frequency_domain_source_model)
        else:
            fdsm_name = None
        if self.time_domain_source_model is not None:
            tdsm_name = utils.get_function_path(self.time_domain_source_model)
        else:
            tdsm_name = None
        if self.parameter_conversion is None:
            param_conv_name = None
        else:
            param_conv_name = utils.get_function_path(self.parameter_conversion)

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
        if parameters is None:
            parameters = self.parameters
        if parameters == self._cache['parameters'] and self._cache['model'] == model and \
                self._cache['transformed_model'] == transformed_model:
            return self._cache['waveform']
        else:
            self._cache['parameters'] = parameters.copy()
            self._cache['model'] = model
            self._cache['transformed_model'] = transformed_model
        parameters = self._format_parameters(parameters)
        if model is not None:
            model_strain = self._strain_from_model(model_data_points, model, parameters)
        elif transformed_model is not None:
            model_strain = self._strain_from_transformed_model(transformed_model_data_points, transformed_model,
                                                               transformation_function, parameters)
        else:
            raise RuntimeError("No source model given")
        self._cache['waveform'] = model_strain
        return model_strain

    def _strain_from_model(self, model_data_points, model, parameters):
        return model(model_data_points, **parameters)

    def _strain_from_transformed_model(
        self, transformed_model_data_points, transformed_model, transformation_function, parameters
    ):
        transformed_model_strain = self._strain_from_model(
            transformed_model_data_points, transformed_model, parameters
        )

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
        if hasattr(self, "_parameters"):
            return self._parameters
        else:
            return self._cache.get("parameters", None)

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
        new_parameters = self._format_parameters(parameters)
        self._parameters = new_parameters

    def _format_parameters(self, parameters):
        if not isinstance(parameters, dict):
            raise TypeError('"parameters" must be a dictionary.')
        new_parameters = parameters.copy()
        new_parameters, _ = self.parameter_conversion(new_parameters)
        for key in self.source_parameter_keys.symmetric_difference(
                new_parameters):
            new_parameters.pop(key)
        new_parameters.update(self.waveform_arguments)
        return new_parameters

    def _parameters_from_source_model(self):
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


class GWSignalWaveformGenerator(WaveformGenerator):
    """
    A wrapper to the `gwsignal waveform generator`_ that allows for use of arbitrary
    waveforms implemented as gwsignal generators and caches the generator between calls.

    This wrapper allows sets of parameters to be enabled/disabled.
    When, e.g., :code:`spinning=False` all spin parameters will be fixed to zero.

    Parameters
    ==========
    spinning: bool
        Whether to model spins of the individual objects (default=True)
    eccentric: bool
        Whether to model orbital eccentricity (default=False)
    tidal: bool
        Whether to model tidal deformability of the individual objects (default=False)
    sampling_frequency: float, optional
        The sampling frequency
    duration: float, optional
        Time duration of data
    start_time: float, optional
        Starting time of the time array
    parameters: dict, optional
        Initial values for the parameters
    parameter_conversion: func, optional
        Function to convert from sampled parameters to parameters of the
        waveform generator. The default value is the identity, i.e., it leaves
        the parameters unaffected.
    waveform_arguments: dict, optional
        A dictionary of fixed keyword arguments to pass to the waveform generator.
        There is one required waveform argument :code:`waveform_approximant`.

    .. gwsignal waveform generator: https://docs.ligo.org/lscsoft/lalsuite/lalsimulation/classlalsimulation_1_1gwsignal_1_1core_1_1waveform_1_1_gravitational_wave_generator.html  # noqa
    """

    generator_pickles = False

    def __init__(self, spinning=True, eccentric=False, tidal=False, **kwargs):
        self.spinning = spinning
        self.tidal = tidal
        self.eccentric = eccentric
        super().__init__(**kwargs)
        self.waveform_approximant = self.waveform_arguments["waveform_approximant"]
        self.generator = self._create_generator()

    def _create_generator(self, waveform_approximant=None):
        try:
            from lalsimulation.gwsignal import gwsignal_get_waveform_generator
        except ImportError:
            raise ImportError("lalsimulation is not installed. Cannot use the GWSignal waveform generator.")

        if waveform_approximant is None:
            waveform_approximant = self.waveform_approximant
        return gwsignal_get_waveform_generator(waveform_approximant)

    def __getstate__(self):
        # the waveform generator can't be pickled add a placeholder
        # so it will be re-instantiated on unpickling
        state = self.__dict__.copy()
        if "generator" in state and not self.generator_pickles:
            state["generator"] = "<unpickleable generator>"
        return state

    def __setstate__(self, state):
        # if the waveform generator can't be pickled, will reinstantiate it
        if state.get("generator", None) == "<unpickleable generator>":
            state["generator"] = self._create_generator(waveform_approximant=state["waveform_approximant"])
        self.__dict__.update(state)

    def __repr__(self):
        if self.parameter_conversion is None:
            param_conv_name = None
        else:
            param_conv_name = utils.get_function_path(self.parameter_conversion)

        return (
            f"{self.__class__.__name__}(duration={self.duration}, "
            f"sampling_frequency={self.duration}, start_time={self.start_time}, "
            f"parameter_conversion={param_conv_name}, "
            f"waveform_arguments={self.waveform_arguments}, "
            f"spinning={self.spinning}, eccentric={self.eccentric}, tidal={self.tidal}"
            ")"
        )

    @property
    def defaults(self):
        keys = list()
        if not self.eccentric:
            keys += ["eccentricity", "mean_per_ano"]
        if not self.tidal:
            keys += ["lambda_1", "lambda_2"]
        if not self.spinning:
            keys += ["a_1", "a_2", "tilt_1", "tilt_2", "phi_12", "phi_jl"]

        output = {key: 0.0 for key in keys}
        return output

    def _from_bilby_parameters(self, **parameters):
        from .conversion import bilby_to_lalsimulation_spins

        parameters = self._format_parameters(parameters)

        waveform_kwargs = dict(
            reference_frequency=50.0,
            minimum_frequency=20.0,
            maximum_frequency=self.frequency_array[-1],
            mode_array=None,
            pn_amplitude_order=0,
        )
        waveform_kwargs.update(self.waveform_arguments)
        reference_frequency = waveform_kwargs['reference_frequency']
        minimum_frequency = waveform_kwargs['minimum_frequency']
        maximum_frequency = waveform_kwargs['maximum_frequency']
        mode_array = waveform_kwargs['mode_array']
        pn_amplitude_order = waveform_kwargs['pn_amplitude_order']

        if pn_amplitude_order != 0:
            # This is to mimic the behaviour in
            # https://git.ligo.org/lscsoft/lalsuite/-/blob/master/lalsimulation/lib/LALSimInspiral.c#L5542
            if pn_amplitude_order == -1:
                if self.waveform_approximant in ["SpinTaylorT4", "SpinTaylorT5"]:
                    pn_amplitude_order = 3  # Equivalent to MAX_PRECESSING_AMP_PN_ORDER in LALSimulation
                else:
                    pn_amplitude_order = 6  # Equivalent to MAX_NONPRECESSING_AMP_PN_ORDER in LALSimulation
            start_frequency = minimum_frequency * 2. / (pn_amplitude_order + 2)
        else:
            start_frequency = minimum_frequency

        params = self.defaults.copy()
        params.update(parameters)
        parameters = params

        iota, spin_1x, spin_1y, spin_1z, spin_2x, spin_2y, spin_2z = bilby_to_lalsimulation_spins(
            theta_jn=parameters["theta_jn"],
            phi_jl=parameters["phi_jl"],
            tilt_1=parameters["tilt_1"],
            tilt_2=parameters["tilt_2"],
            phi_12=parameters["phi_12"],
            a_1=parameters["a_1"],
            a_2=parameters["a_2"],
            mass_1=parameters["mass_1"] * utils.solar_mass,
            mass_2=parameters["mass_2"] * utils.solar_mass,
            reference_frequency=reference_frequency,
            phase=parameters["phase"],
        )

        gwsignal_dict = {
            'mass1': parameters["mass_1"],
            'mass2': parameters["mass_2"],
            'spin1x': spin_1x,
            'spin1y': spin_1y,
            'spin1z': spin_1z,
            'spin2x': spin_2x,
            'spin2y': spin_2y,
            'spin2z': spin_2z,
            'lambda1': parameters["lambda_1"],
            'lambda2': parameters["lambda_2"],
            'deltaF': 1 / self.duration,
            'deltaT': 1 / self.sampling_frequency,
            'f22_start': start_frequency,
            'f_max': maximum_frequency,
            'f22_ref': reference_frequency,
            'phi_ref': parameters["phase"],
            'distance': parameters["luminosity_distance"] * 1e6,
            'inclination': iota,
            'eccentricity': parameters["eccentricity"],
            'meanPerAno': parameters["mean_per_ano"],
            'condition': int(self.generator.metadata["implemented_domain"] == 'time'),
        }

        # add astropy units to the parameters using the defaults from gwsignal
        from lalsimulation.gwsignal.core.parameter_conventions import Cosmo_units_dictionary

        gwsignal_dict = {
            key: val << Cosmo_units_dictionary.get(key, 0)
            for key, val in gwsignal_dict.items()
        }

        if mode_array is not None:
            gwsignal_dict.update(ModeArray=mode_array)

        extra_args = waveform_kwargs.copy()

        for key in [
                "waveform_approximant",
                "reference_frequency",
                "minimum_frequency",
                "maximum_frequency",
                "catch_waveform_errors",
                "mode_array",
                "pn_spin_order",
                "pn_amplitude_order",
                "pn_tidal_order",
                "pn_phase_order",
                "numerical_relativity_file",
        ]:
            if key in extra_args.keys():
                del extra_args[key]

        gwsignal_dict.update(extra_args)
        return gwsignal_dict

    def frequency_domain_strain(self, parameters):
        from lalsimulation.gwsignal import GenerateFDWaveform

        if parameters is None:
            parameters = self.parameters

        hpc = _try_waveform_call(
            GenerateFDWaveform,
            self._from_bilby_parameters(**parameters),
            self.generator,
            self.waveform_arguments.get("catch_waveform_errors", False)
        )

        wf = self._extract_waveform(hpc, "frequency")
        if wf is None:
            return None

        minimum_frequency = self.waveform_arguments.get("minimum_frequency", 20.0)
        maximum_frequency = self.waveform_arguments.get("maximum_frequency", self.frequency_array[-1])
        frequency_bounds = (
            (self.frequency_array >= minimum_frequency)
            * (self.frequency_array <= maximum_frequency)
        )
        for key in wf:
            wf[key] *= frequency_bounds

        if self.generator.metadata["implemented_domain"] == 'time':
            dt = 1 / hpc.hp.df.value + hpc.hp.epoch.value
            time_shift = np.exp(-1j * 2 * np.pi * dt * self.frequency_array[frequency_bounds])
            for key in wf:
                wf[key][frequency_bounds] *= time_shift

        return wf

    def time_domain_strain(self, parameters):
        from lalsimulation.gwsignal import GenerateTDWaveform

        if parameters is None:
            parameters = self.parameters

        hpc = _try_waveform_call(
            GenerateTDWaveform,
            self._from_bilby_parameters(**parameters),
            self.generator,
            self.waveform_arguments.get("catch_waveform_errors", False)
        )
        return self._extract_waveform(hpc, "time")

    def _extract_waveform(self, hpc, kind):
        # pass through waveform errors
        if hpc is None:
            return None

        if kind == "frequency":
            dtype = complex
            array = self.frequency_array
        else:
            dtype = float
            array = self.time_array

        h_plus = np.zeros(array.shape, dtype=dtype)
        h_cross = np.zeros(array.shape, dtype=dtype)

        if len(hpc.hp) > len(array):
            logger.debug(
                f"GWsignal waveform longer than bilby's `{kind}_array`({len(hpc.hp)} "
                f"vs {len(array)}). Truncating GWsignal array.")
            # set slice to force the output into a numpy array
            h_plus[:] = hpc.hp[:len(h_plus)]
            h_cross[:] = hpc.hc[:len(h_cross)]
        else:
            h_plus[:len(hpc.hp)] = hpc.hp
            h_cross[:len(hpc.hc)] = hpc.hc

        return dict(plus=h_plus, cross=h_cross)

    _all_parameters = {
        "mass_1",
        "mass_2",
        "luminosity_distance",
        "a_1",
        "tilt_1",
        "phi_12",
        "a_2",
        "tilt_2",
        "phi_jl",
        "theta_jn",
        "phase",
        "eccentricity",
        "mean_per_ano",
        "lambda_1",
        "lambda_2",
    }

    def _parameters_from_source_model(self):
        return self._all_parameters.difference(self.defaults.keys())


def _try_waveform_call(func, parameters, generator, catch_waveform_errors):
    try:
        return func(parameters, generator)
    except Exception as e:
        if not catch_waveform_errors:
            raise
        else:
            EDOM = "Input domain error" in e.args[0]
            if EDOM:
                logger.warning(
                    f"Evaluating the waveform failed with error: {e}\nThe parameters "
                    f"were {parameters}\nLikelihood will be set to -inf."
                )
                return None
            else:
                raise
