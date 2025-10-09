import unittest
from unittest import mock

import bilby
import lalsimulation
import numpy as np


def dummy_func_array_return_value(
    frequency_array, amplitude, mu, sigma, ra, dec, geocent_time, psi, **kwargs
):
    return amplitude + mu + frequency_array + sigma + ra + dec + geocent_time + psi


def dummy_func_dict_return_value(
    frequency_array, amplitude, mu, sigma, ra, dec, geocent_time, psi, **kwargs
):
    ht = {
        "plus": amplitude
        + mu
        + frequency_array
        + sigma
        + ra
        + dec
        + geocent_time
        + psi,
        "cross": amplitude
        + mu
        + frequency_array
        + sigma
        + ra
        + dec
        + geocent_time
        + psi,
    }
    return ht


def dummy_func_array_return_value_2(
    array, amplitude, mu, sigma, ra, dec, geocent_time, psi
):
    return dict(plus=np.array(array), cross=np.array(array))


class TestWaveformGeneratorInstantiationWithoutOptionalParameters(unittest.TestCase):
    def setUp(self):
        self.waveform_generator = bilby.gw.waveform_generator.WaveformGenerator(
            1, 4096, frequency_domain_source_model=dummy_func_dict_return_value
        )
        self.simulation_parameters = dict(
            amplitude=1e-21,
            mu=100,
            sigma=1,
            ra=1.375,
            dec=-1.2108,
            geocent_time=1126259642.413,
            psi=2.659,
        )

    def tearDown(self):
        del self.waveform_generator
        del self.simulation_parameters

    def test_repr(self):
        expected = (
            "WaveformGenerator(duration={}, sampling_frequency={}, start_time={}, "
            "frequency_domain_source_model={}, time_domain_source_model={}, "
            "parameter_conversion={}, waveform_arguments={})".format(
                self.waveform_generator.duration,
                self.waveform_generator.sampling_frequency,
                self.waveform_generator.start_time,
                bilby.core.utils.get_function_path(self.waveform_generator.frequency_domain_source_model),
                bilby.core.utils.get_function_path(self.waveform_generator.time_domain_source_model),
                bilby.core.utils.get_function_path(bilby.gw.conversion.convert_to_lal_binary_black_hole_parameters),
                self.waveform_generator.waveform_arguments,
            )
        )
        self.assertEqual(expected, repr(self.waveform_generator))

    def test_repr_with_time_domain_source_model(self):
        self.waveform_generator = bilby.gw.waveform_generator.WaveformGenerator(
            1, 4096, time_domain_source_model=dummy_func_dict_return_value
        )
        expected = (
            "WaveformGenerator(duration={}, sampling_frequency={}, start_time={}, "
            "frequency_domain_source_model={}, time_domain_source_model={}, "
            "parameter_conversion={}, waveform_arguments={})".format(
                self.waveform_generator.duration,
                self.waveform_generator.sampling_frequency,
                self.waveform_generator.start_time,
                bilby.core.utils.get_function_path(self.waveform_generator.frequency_domain_source_model),
                bilby.core.utils.get_function_path(self.waveform_generator.time_domain_source_model),
                bilby.core.utils.get_function_path(bilby.gw.conversion.convert_to_lal_binary_black_hole_parameters),
                self.waveform_generator.waveform_arguments,
            )
        )
        self.assertEqual(expected, repr(self.waveform_generator))

    def test_repr_with_param_conversion(self):
        def conversion_func():
            pass

        self.waveform_generator.parameter_conversion = conversion_func
        expected = (
            "WaveformGenerator(duration={}, sampling_frequency={}, start_time={}, "
            "frequency_domain_source_model={}, time_domain_source_model={}, "
            "parameter_conversion={}, waveform_arguments={})".format(
                self.waveform_generator.duration,
                self.waveform_generator.sampling_frequency,
                self.waveform_generator.start_time,
                bilby.core.utils.get_function_path(self.waveform_generator.frequency_domain_source_model),
                bilby.core.utils.get_function_path(self.waveform_generator.time_domain_source_model),
                bilby.core.utils.get_function_path(conversion_func),
                self.waveform_generator.waveform_arguments,
            )
        )
        self.assertEqual(expected, repr(self.waveform_generator))

    def test_duration(self):
        self.assertEqual(self.waveform_generator.duration, 1)

    def test_sampling_frequency(self):
        self.assertEqual(self.waveform_generator.sampling_frequency, 4096)

    def test_source_model(self):
        self.assertEqual(
            self.waveform_generator.frequency_domain_source_model,
            dummy_func_dict_return_value,
        )

    def test_frequency_array_type(self):
        self.assertIsInstance(self.waveform_generator.frequency_array, np.ndarray)

    def test_time_array_type(self):
        self.assertIsInstance(self.waveform_generator.time_array, np.ndarray)

    def test_source_model_parameters(self):
        self.waveform_generator.parameters = self.simulation_parameters.copy()
        self.assertListEqual(
            sorted(list(self.waveform_generator.parameters.keys())),
            sorted(list(self.simulation_parameters.keys())),
        )


class TestWaveformArgumentsSetting(unittest.TestCase):
    def setUp(self):
        self.waveform_generator = bilby.gw.waveform_generator.WaveformGenerator(
            1,
            4096,
            frequency_domain_source_model=dummy_func_dict_return_value,
            waveform_arguments=dict(test="test", arguments="arguments"),
        )

    def tearDown(self):
        del self.waveform_generator

    def test_waveform_arguments_init_setting(self):
        self.assertDictEqual(
            self.waveform_generator.waveform_arguments,
            dict(test="test", arguments="arguments"),
        )


class TestLALCBCWaveformArgumentsSetting(unittest.TestCase):
    def setUp(self):
        self.kwargs = dict(
            duration=4,
            frequency_domain_source_model=bilby.gw.source.lal_binary_black_hole,
            sampling_frequency=2048,
        )

    def tearDown(self):
        del self.kwargs

    def test_spin_reference_enumeration(self):
        """
        Verify that the value of the reference enumerator hasn't changed by comparing
        against a known approximant.
        """
        self.assertEqual(
            lalsimulation.SimInspiralGetSpinFreqFromApproximant(lalsimulation.SEOBNRv3),
            bilby.gw.waveform_generator.LALCBCWaveformGenerator.LAL_SIM_INSPIRAL_SPINS_FLOW,
        )

    def test_create_waveform_generator_non_precessing(self):
        self.kwargs["waveform_arguments"] = dict(
            minimum_frequency=20.0,
            reference_frequency=50.0,
            waveform_approximant="TaylorF2",
        )
        wfg = bilby.gw.waveform_generator.LALCBCWaveformGenerator(**self.kwargs)
        self.assertDictEqual(
            wfg.waveform_arguments,
            dict(
                minimum_frequency=20.0,
                reference_frequency=50.0,
                waveform_approximant="TaylorF2",
            ),
        )

    def test_create_waveform_generator_eob_succeeds(self):
        self.kwargs["waveform_arguments"] = dict(
            minimum_frequency=20.0,
            reference_frequency=20.0,
            waveform_approximant="SEOBNRv3",
        )
        wfg = bilby.gw.waveform_generator.LALCBCWaveformGenerator(**self.kwargs)
        self.assertDictEqual(
            wfg.waveform_arguments,
            dict(
                minimum_frequency=20.0,
                reference_frequency=20.0,
                waveform_approximant="SEOBNRv3",
            ),
        )

    def test_create_waveform_generator_eob_fails(self):
        self.kwargs["waveform_arguments"] = dict(
            minimum_frequency=20.0,
            reference_frequency=50.0,
            waveform_approximant="SEOBNRv3",
        )
        with self.assertRaises(ValueError):
            _ = bilby.gw.waveform_generator.LALCBCWaveformGenerator(**self.kwargs)


class TestSetters(unittest.TestCase):
    def setUp(self):
        self.waveform_generator = bilby.gw.waveform_generator.WaveformGenerator(
            1, 4096, frequency_domain_source_model=dummy_func_dict_return_value
        )
        self.simulation_parameters = dict(
            amplitude=1e-21,
            mu=100,
            sigma=1,
            ra=1.375,
            dec=-1.2108,
            geocent_time=1126259642.413,
            psi=2.659,
        )

    def tearDown(self):
        del self.waveform_generator
        del self.simulation_parameters

    def test_parameter_setter_sets_expected_values_with_expected_keys(self):
        self.waveform_generator.parameters = self.simulation_parameters.copy()
        for key in self.simulation_parameters:
            self.assertEqual(
                self.waveform_generator.parameters[key], self.simulation_parameters[key]
            )

    def test_parameter_setter_none_handling(self):
        with self.assertRaises(TypeError):
            self.waveform_generator.parameters = None
        # self.assertListEqual(sorted(list(self.waveform_generator.parameters.keys())),
        #                      sorted(list(self.simulation_parameters.keys())))

    def test_frequency_array_setter(self):
        new_frequency_array = np.arange(1, 100)
        self.waveform_generator.frequency_array = new_frequency_array
        self.assertTrue(
            np.array_equal(new_frequency_array, self.waveform_generator.frequency_array)
        )

    def test_time_array_setter(self):
        new_time_array = np.arange(1, 100)
        self.waveform_generator.time_array = new_time_array
        self.assertTrue(
            np.array_equal(new_time_array, self.waveform_generator.time_array)
        )

    def test_parameters_set_from_frequency_domain_source_model(self):
        self.waveform_generator.frequency_domain_source_model = (
            dummy_func_dict_return_value
        )
        self.waveform_generator.parameters = self.simulation_parameters.copy()
        self.assertListEqual(
            sorted(list(self.waveform_generator.parameters.keys())),
            sorted(list(self.simulation_parameters.keys())),
        )

    def test_parameters_set_from_time_domain_source_model(self):
        self.waveform_generator.time_domain_source_model = dummy_func_dict_return_value
        self.waveform_generator.parameters = self.simulation_parameters.copy()
        self.assertListEqual(
            sorted(list(self.waveform_generator.parameters.keys())),
            sorted(list(self.simulation_parameters.keys())),
        )

    def test_set_parameter_conversion_at_init(self):
        def conversion_func():
            pass

        self.waveform_generator = bilby.gw.waveform_generator.WaveformGenerator(
            1,
            4096,
            frequency_domain_source_model=dummy_func_dict_return_value,
            parameter_conversion=conversion_func,
        )
        self.assertEqual(conversion_func, self.waveform_generator.parameter_conversion)


class TestFrequencyDomainStrainMethod(unittest.TestCase):
    def setUp(self):
        self.waveform_generator = bilby.gw.waveform_generator.WaveformGenerator(
            duration=1,
            sampling_frequency=4096,
            frequency_domain_source_model=dummy_func_dict_return_value,
        )
        self.simulation_parameters = dict(
            amplitude=1e-2,
            mu=100,
            sigma=1,
            ra=1.375,
            dec=-1.2108,
            geocent_time=1126259642.413,
            psi=2.659,
        )

    def tearDown(self):
        del self.waveform_generator
        del self.simulation_parameters

    def test_parameter_conversion_is_called(self):
        self.waveform_generator.parameter_conversion = mock.MagicMock(
            side_effect=KeyError("test")
        )
        with self.assertRaises(KeyError):
            self.waveform_generator.frequency_domain_strain(
                parameters=self.simulation_parameters
            )

    def test_frequency_domain_source_model_call(self):
        expected = self.waveform_generator.frequency_domain_source_model(
            self.waveform_generator.frequency_array,
            self.simulation_parameters["amplitude"],
            self.simulation_parameters["mu"],
            self.simulation_parameters["sigma"],
            self.simulation_parameters["ra"],
            self.simulation_parameters["dec"],
            self.simulation_parameters["geocent_time"],
            self.simulation_parameters["psi"],
        )
        actual = self.waveform_generator.frequency_domain_strain(
            parameters=self.simulation_parameters
        )
        self.assertTrue(np.array_equal(expected["plus"], actual["plus"]))
        self.assertTrue(np.array_equal(expected["cross"], actual["cross"]))

    def test_time_domain_source_model_call_with_ndarray(self):
        self.waveform_generator.frequency_domain_source_model = None
        self.waveform_generator.time_domain_source_model = dummy_func_array_return_value

        def side_effect(value, value2):
            return value

        with mock.patch("bilby.core.utils.nfft") as m:
            m.side_effect = side_effect
            expected = self.waveform_generator.time_domain_strain(
                parameters=self.simulation_parameters
            )
            actual = self.waveform_generator.frequency_domain_strain(
                parameters=self.simulation_parameters
            )
            self.assertTrue(np.array_equal(expected, actual))

    def test_time_domain_source_model_call_with_dict(self):
        self.waveform_generator.frequency_domain_source_model = None
        self.waveform_generator.time_domain_source_model = dummy_func_dict_return_value

        def side_effect(value, value2):
            return value, self.waveform_generator.frequency_array

        with mock.patch("bilby.core.utils.nfft") as m:
            m.side_effect = side_effect
            expected = self.waveform_generator.time_domain_strain(
                parameters=self.simulation_parameters
            )
            actual = self.waveform_generator.frequency_domain_strain(
                parameters=self.simulation_parameters
            )
            self.assertTrue(np.array_equal(expected["plus"], actual["plus"]))
            self.assertTrue(np.array_equal(expected["cross"], actual["cross"]))

    def test_no_source_model_given(self):
        self.waveform_generator.time_domain_source_model = None
        self.waveform_generator.frequency_domain_source_model = None
        with self.assertRaises(RuntimeError):
            self.waveform_generator.frequency_domain_strain(
                parameters=self.simulation_parameters
            )

    def test_key_popping(self):
        self.waveform_generator.parameter_conversion = mock.MagicMock(
            return_value=(
                dict(
                    amplitude=1e-21,
                    mu=100,
                    sigma=1,
                    ra=1.375,
                    dec=-1.2108,
                    geocent_time=1126259642.413,
                    psi=2.659,
                    c=None,
                    d=None,
                ),
                ["c", "d"],
            )
        )
        try:
            self.waveform_generator.frequency_domain_strain(
                parameters=self.simulation_parameters
            )
        except RuntimeError:
            pass
        self.assertListEqual(
            sorted(self.waveform_generator.parameters.keys()),
            sorted(["amplitude", "mu", "sigma", "ra", "dec", "geocent_time", "psi"]),
        )

    def test_caching_with_parameters(self):
        original_waveform = self.waveform_generator.frequency_domain_strain(
            parameters=self.simulation_parameters
        )
        new_waveform = self.waveform_generator.frequency_domain_strain(
            parameters=self.simulation_parameters
        )
        self.assertDictEqual(original_waveform, new_waveform)

    def test_caching_without_parameters(self):
        original_waveform = self.waveform_generator.frequency_domain_strain(
            parameters=self.simulation_parameters
        )
        new_waveform = self.waveform_generator.frequency_domain_strain()
        self.assertDictEqual(original_waveform, new_waveform)

    def test_frequency_domain_caching_and_using_time_domain_strain_without_parameters(
        self,
    ):
        self.assertFalse(_test_caching_different_domain(
            self.waveform_generator.frequency_domain_strain,
            self.waveform_generator.time_domain_strain,
            self.simulation_parameters,
            None,
        ))

    def test_frequency_domain_caching_and_using_time_domain_strain_with_parameters(
        self,
    ):
        self.assertFalse(_test_caching_different_domain(
            self.waveform_generator.frequency_domain_strain,
            self.waveform_generator.time_domain_strain,
            self.simulation_parameters,
            self.simulation_parameters,
        ))

    def test_time_domain_caching_and_using_frequency_domain_strain_without_parameters(
        self,
    ):
        self.assertFalse(_test_caching_different_domain(
            self.waveform_generator.time_domain_strain,
            self.waveform_generator.frequency_domain_strain,
            self.simulation_parameters,
            None,
        ))

    def test_time_domain_caching_and_using_frequency_domain_strain_with_parameters(
        self,
    ):
        self.assertFalse(_test_caching_different_domain(
            self.waveform_generator.time_domain_strain,
            self.waveform_generator.frequency_domain_strain,
            self.simulation_parameters,
            self.simulation_parameters,
        ))

    def test_frequency_domain_caching_changing_model(self):
        original_waveform = self.waveform_generator.frequency_domain_strain(
            parameters=self.simulation_parameters
        )
        self.waveform_generator.frequency_domain_source_model = (
            dummy_func_array_return_value_2
        )
        new_waveform = self.waveform_generator.frequency_domain_strain(
            parameters=self.simulation_parameters
        )
        self.assertFalse(
            np.array_equal(original_waveform["plus"], new_waveform["plus"])
        )

    def test_time_domain_caching_changing_model(self):
        self.waveform_generator = bilby.gw.waveform_generator.WaveformGenerator(
            duration=1,
            sampling_frequency=4096,
            time_domain_source_model=dummy_func_dict_return_value,
        )
        original_waveform = self.waveform_generator.frequency_domain_strain(
            parameters=self.simulation_parameters
        )
        self.waveform_generator.time_domain_source_model = (
            dummy_func_array_return_value_2
        )
        new_waveform = self.waveform_generator.frequency_domain_strain(
            parameters=self.simulation_parameters
        )
        self.assertFalse(
            np.array_equal(original_waveform["plus"], new_waveform["plus"])
        )


class TestTimeDomainStrainMethod(unittest.TestCase):
    def setUp(self):
        self.waveform_generator = bilby.gw.waveform_generator.WaveformGenerator(
            1, 4096, time_domain_source_model=dummy_func_dict_return_value
        )
        self.simulation_parameters = dict(
            amplitude=1e-21,
            mu=100,
            sigma=1,
            ra=1.375,
            dec=-1.2108,
            geocent_time=1126259642.413,
            psi=2.659,
        )

    def tearDown(self):
        del self.waveform_generator
        del self.simulation_parameters

    def test_parameter_conversion_is_called(self):
        self.waveform_generator.parameter_conversion = mock.MagicMock(
            side_effect=KeyError("test")
        )
        with self.assertRaises(KeyError):
            self.waveform_generator.time_domain_strain(
                parameters=self.simulation_parameters
            )

    def test_time_domain_source_model_call(self):
        expected = self.waveform_generator.time_domain_source_model(
            self.waveform_generator.time_array,
            self.simulation_parameters["amplitude"],
            self.simulation_parameters["mu"],
            self.simulation_parameters["sigma"],
            self.simulation_parameters["ra"],
            self.simulation_parameters["dec"],
            self.simulation_parameters["geocent_time"],
            self.simulation_parameters["psi"],
        )
        actual = self.waveform_generator.time_domain_strain(
            parameters=self.simulation_parameters
        )
        self.assertTrue(np.array_equal(expected["plus"], actual["plus"]))
        self.assertTrue(np.array_equal(expected["cross"], actual["cross"]))

    def test_frequency_domain_source_model_call_with_ndarray(self):
        self.waveform_generator.time_domain_source_model = None
        self.waveform_generator.frequency_domain_source_model = (
            dummy_func_array_return_value
        )

        def side_effect(value, value2):
            return value

        with mock.patch("bilby.core.utils.infft") as m:
            m.side_effect = side_effect
            expected = self.waveform_generator.frequency_domain_strain(
                parameters=self.simulation_parameters
            )
            actual = self.waveform_generator.time_domain_strain(
                parameters=self.simulation_parameters
            )
            self.assertTrue(np.array_equal(expected, actual))

    def test_frequency_domain_source_model_call_with_dict(self):
        self.waveform_generator.time_domain_source_model = None
        self.waveform_generator.frequency_domain_source_model = (
            dummy_func_dict_return_value
        )

        def side_effect(value, value2):
            return value

        with mock.patch("bilby.core.utils.infft") as m:
            m.side_effect = side_effect
            expected = self.waveform_generator.frequency_domain_strain(
                parameters=self.simulation_parameters
            )
            actual = self.waveform_generator.time_domain_strain(
                parameters=self.simulation_parameters
            )
            self.assertTrue(np.array_equal(expected["plus"], actual["plus"]))
            self.assertTrue(np.array_equal(expected["cross"], actual["cross"]))

    def test_no_source_model_given(self):
        self.waveform_generator.time_domain_source_model = None
        self.waveform_generator.frequency_domain_source_model = None
        with self.assertRaises(RuntimeError):
            self.waveform_generator.time_domain_strain(
                parameters=self.simulation_parameters
            )

    def test_key_popping(self):
        self.waveform_generator.parameter_conversion = mock.MagicMock(
            return_value=(
                dict(
                    amplitude=1e-2,
                    mu=100,
                    sigma=1,
                    ra=1.375,
                    dec=-1.2108,
                    geocent_time=1126259642.413,
                    psi=2.659,
                    c=None,
                    d=None,
                ),
                ["c", "d"],
            )
        )
        try:
            self.waveform_generator.time_domain_strain(
                parameters=self.simulation_parameters
            )
        except RuntimeError:
            pass
        self.assertListEqual(
            sorted(self.waveform_generator.parameters.keys()),
            sorted(["amplitude", "mu", "sigma", "ra", "dec", "geocent_time", "psi"]),
        )

    def test_caching_with_parameters(self):
        original_waveform = self.waveform_generator.time_domain_strain(
            parameters=self.simulation_parameters
        )
        new_waveform = self.waveform_generator.time_domain_strain(
            parameters=self.simulation_parameters
        )
        self.assertDictEqual(original_waveform, new_waveform)

    def test_caching_without_parameters(self):
        original_waveform = self.waveform_generator.time_domain_strain(
            parameters=self.simulation_parameters
        )
        new_waveform = self.waveform_generator.time_domain_strain()
        self.assertDictEqual(original_waveform, new_waveform)

    def test_frequency_domain_caching_and_using_time_domain_strain_without_parameters(
        self,
    ):
        self.assertFalse(_test_caching_different_domain(
            self.waveform_generator.frequency_domain_strain,
            self.waveform_generator.time_domain_strain,
            self.simulation_parameters,
            None,
        ))

    def test_frequency_domain_caching_and_using_time_domain_strain_with_parameters(
        self,
    ):
        self.assertFalse(_test_caching_different_domain(
            self.waveform_generator.frequency_domain_strain,
            self.waveform_generator.time_domain_strain,
            self.simulation_parameters,
            self.simulation_parameters,
        ))

    def test_time_domain_caching_and_using_frequency_domain_strain_without_parameters(
        self,
    ):
        self.assertFalse(_test_caching_different_domain(
            self.waveform_generator.time_domain_strain,
            self.waveform_generator.frequency_domain_strain,
            self.simulation_parameters,
            None,
        ))

    def test_time_domain_caching_and_using_frequency_domain_strain_with_parameters(
        self,
    ):
        self.assertFalse(_test_caching_different_domain(
            self.waveform_generator.time_domain_strain,
            self.waveform_generator.frequency_domain_strain,
            self.simulation_parameters,
            self.simulation_parameters,
        ))


def _test_caching_different_domain(func1, func2, params1, params2):
    original_waveform = func1(parameters=params1)
    new_waveform = func2(parameters=params2)
    output = True
    for key in original_waveform:
        output &= np.array_equal(original_waveform[key], new_waveform[key])
    return output


class TestGWSignalGenerator(unittest.TestCase):

    def get_wfgen(self, **kwargs):
        default_kwargs = dict(
            duration=4,
            sampling_frequency=256,
            waveform_arguments=dict(waveform_approximant=kwargs.pop("waveform_approximant", "IMRPhenomXP")),
        )
        default_kwargs.update(kwargs)
        return bilby.gw.waveform_generator.GWSignalWaveformGenerator(**default_kwargs)

    def _test_defaults(self, kind, keys):
        # test included parameters are not in the defaults
        kwargs = {kind: True}
        wfg = self.get_wfgen(**kwargs)
        defaults = wfg.defaults
        for key in keys:
            assert key not in defaults

        # test omitted parameters are in the defaults
        kwargs = {kind: False}
        wfg = self.get_wfgen(**kwargs)
        defaults = wfg.defaults
        for key in keys:
            assert key in defaults

        # test an error is raised when the requested parameters aren't provided
        kwargs = {kind: True}
        wfg = self.get_wfgen(**kwargs)
        parameters = dict(
            mass_1=10,
            mass_2=10,
            theta_jn=1,
            luminosity_distance=1,
            phase=0,
        )
        with self.assertRaises(KeyError):
            wfg.frequency_domain_strain(parameters)

    def test_spin_defaults(self):
        spin_parameters = ["a_1", "a_2", "tilt_1", "tilt_2", "phi_12", "phi_jl"]
        self._test_defaults("spinning", spin_parameters)

    def test_eccentric_defaults(self):
        eccentric_parameters = ["eccentricity", "mean_per_ano"]
        self._test_defaults("eccentric", eccentric_parameters)

    def test_tidal_defaults(self):
        tidal_parameters = ["lambda_1", "lambda_2"]
        self._test_defaults("tidal", tidal_parameters)

    def test_default_parameters_are_ignored(self):
        wfg = self.get_wfgen(spinning=False)
        parameters_1 = dict(
            mass_1=10.0,
            mass_2=10.0,
            theta_jn=1.0,
            luminosity_distance=1.0,
            phase=0.0,
        )
        parameters_2 = dict(
            mass_1=10.0,
            mass_2=10.0,
            theta_jn=1.0,
            luminosity_distance=1.0,
            phase=0.0,
            a_1=0.99,
        )
        wf_1 = wfg.frequency_domain_strain(parameters_1)
        wf_2 = wfg.frequency_domain_strain(parameters_2)
        np.testing.assert_equal(wf_1["plus"], wf_2["plus"])

    def test_eccentric_parameters_work(self):
        wfg = self.get_wfgen(
            eccentric=True, spinning=False, waveform_approximant="EccentricFD",
        )
        parameters_1 = dict(
            mass_1=10,
            mass_2=10,
            theta_jn=1.0,
            luminosity_distance=1.0,
            phase=0.0,
            eccentricity=0.0,
            mean_per_ano=0.0,
        )
        parameters_2 = dict(
            mass_1=10,
            mass_2=10,
            theta_jn=1.0,
            luminosity_distance=1.0,
            phase=0.0,
            eccentricity=0.3,
            mean_per_ano=1.0,
        )
        wf_1 = wfg.frequency_domain_strain(parameters_1)
        wf_2 = wfg.frequency_domain_strain(parameters_2)
        assert not all(wf_1["plus"] == wf_2["plus"])

    def test_tidal_parameters_work(self):
        wfg = self.get_wfgen(
            tidal=True, spinning=False, waveform_approximant="IMRPhenomD_NRTidalv2",
            sampling_frequency=16384,
        )
        parameters_1 = dict(
            mass_1=1.4,
            mass_2=1.3,
            theta_jn=1.0,
            luminosity_distance=1.0,
            phase=0.0,
            lambda_1=0.0,
            lambda_2=0.0,
        )
        parameters_2 = dict(
            mass_1=1.4,
            mass_2=1.3,
            theta_jn=1.0,
            luminosity_distance=1.0,
            phase=0.0,
            lambda_1=10000.0,
            lambda_2=1000.0,
        )
        wf_1 = wfg.frequency_domain_strain(parameters_1)
        wf_2 = wfg.frequency_domain_strain(parameters_2)
        assert not all(wf_1["plus"] == wf_2["plus"])

    def test_bilby_to_gwsignal_parameters(self):
        from astropy import units as u

        wfg = self.get_wfgen(spinning=False)
        expected = dict(
            mass1=10.0 * u.M_sun,
            mass2=10.0 * u.M_sun,
            spin1x=0.0 * u.dimensionless_unscaled,
            spin1y=0.0 * u.dimensionless_unscaled,
            spin1z=0.0 * u.dimensionless_unscaled,
            spin2x=0.0 * u.dimensionless_unscaled,
            spin2y=0.0 * u.dimensionless_unscaled,
            spin2z=0.0 * u.dimensionless_unscaled,
            lambda1=0.0 * u.dimensionless_unscaled,
            lambda2=0.0 * u.dimensionless_unscaled,
            deltaF=0.25 * u.Hz,
            deltaT=1 / 256 * u.s,
            f22_start=20 * u.Hz,
            f_max=128 * u.Hz,
            f22_ref=50.0 * u.Hz,
            phi_ref=0 * u.rad,
            distance=1 * u.Mpc,
            inclination=1 * u.rad,
            eccentricity=0 * u.dimensionless_unscaled,
            meanPerAno=0 * u.rad,
            condition=0,
        )
        parameters = wfg._from_bilby_parameters(
            mass_1=10.0,
            mass_2=10.0,
            theta_jn=1.0,
            luminosity_distance=1.0,
            phase=0.0,
        )
        print(parameters)
        print(expected)
        assert expected == parameters

    def test_gwsignal_generator_matches_base(self):
        base = bilby.gw.waveform_generator.WaveformGenerator(
            duration=4,
            sampling_frequency=256,
            waveform_arguments=dict(waveform_approximant="IMRPhenomXP"),
            parameter_conversion=bilby.gw.conversion.convert_to_lal_binary_black_hole_parameters,
            frequency_domain_source_model=bilby.gw.source.lal_binary_black_hole,
        )
        wfg = self.get_wfgen()

        parameters = dict(
            mass_1=10.0,
            mass_2=8.0,
            a_1=0.6,
            a_2=0.4,
            tilt_1=3.0,
            tilt_2=1.2,
            phi_12=0.2,
            phi_jl=1.2,
            theta_jn=1.0,
            luminosity_distance=1.0,
            phase=0.0,
        )

        base_wf = base.frequency_domain_strain(parameters)
        new_wf = wfg.frequency_domain_strain(parameters)
        assert base_wf.keys() == new_wf.keys()
        for key in base_wf:
            np.testing.assert_almost_equal(base_wf[key], new_wf[key])

        base_wf = base.time_domain_strain(parameters)
        new_wf = wfg.time_domain_strain(parameters)
        assert base_wf.keys() == new_wf.keys()
        for key in base_wf:
            np.testing.assert_almost_equal(base_wf[key], new_wf[key])

    def test_raises_error_with_extra_kwargs(self):
        wfg = self.get_wfgen(
            waveform_arguments=dict(waveform_approximant="IMRPhenomD", foo="bar"),
            spinning=False,
        )
        parameters = dict(
            mass_1=10.0,
            mass_2=10.0,
            theta_jn=1.0,
            luminosity_distance=1.0,
            phase=0.0,
        )

        with self.assertRaises(TypeError):
            wfg.frequency_domain_strain(parameters)

    def test_caught_error_return_none(self):
        wfg = self.get_wfgen(
            spinning=False,
            waveform_arguments=dict(waveform_approximant="IMRPhenomD", catch_waveform_errors=True),
        )
        parameters = dict(
            mass_1=10.0,
            mass_2=-10.0,
            theta_jn=1.0,
            luminosity_distance=1.0,
            phase=0.0,
        )

        assert wfg.frequency_domain_strain(parameters) is None


if __name__ == "__main__":
    unittest.main()
