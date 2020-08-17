from __future__ import absolute_import
import unittest
import bilby
import numpy as np
import mock
from mock import MagicMock
import lal
import lalsimulation as lalsim


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
                self.waveform_generator.frequency_domain_source_model.__name__,
                self.waveform_generator.time_domain_source_model,
                bilby.gw.conversion.convert_to_lal_binary_black_hole_parameters.__name__,
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
                self.waveform_generator.frequency_domain_source_model,
                self.waveform_generator.time_domain_source_model.__name__,
                bilby.gw.conversion.convert_to_lal_binary_black_hole_parameters.__name__,
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
                self.waveform_generator.frequency_domain_source_model.__name__,
                self.waveform_generator.time_domain_source_model,
                conversion_func.__name__,
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
        self.waveform_generator.parameter_conversion = MagicMock(
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
        self.waveform_generator.parameter_conversion = MagicMock(
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
        original_waveform = self.waveform_generator.frequency_domain_strain(
            parameters=self.simulation_parameters
        )
        new_waveform = self.waveform_generator.time_domain_strain()
        self.assertNotEqual(original_waveform, new_waveform)

    def test_frequency_domain_caching_and_using_time_domain_strain_with_parameters(
        self,
    ):
        original_waveform = self.waveform_generator.frequency_domain_strain(
            parameters=self.simulation_parameters
        )
        new_waveform = self.waveform_generator.time_domain_strain(
            parameters=self.simulation_parameters
        )
        self.assertNotEqual(original_waveform, new_waveform)

    def test_time_domain_caching_and_using_frequency_domain_strain_without_parameters(
        self,
    ):
        original_waveform = self.waveform_generator.time_domain_strain(
            parameters=self.simulation_parameters
        )
        new_waveform = self.waveform_generator.frequency_domain_strain()
        self.assertNotEqual(original_waveform, new_waveform)

    def test_time_domain_caching_and_using_frequency_domain_strain_with_parameters(
        self,
    ):
        original_waveform = self.waveform_generator.time_domain_strain(
            parameters=self.simulation_parameters
        )
        new_waveform = self.waveform_generator.frequency_domain_strain(
            parameters=self.simulation_parameters
        )
        self.assertNotEqual(original_waveform, new_waveform)

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
        self.waveform_generator.parameter_conversion = MagicMock(
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
        self.waveform_generator.parameter_conversion = MagicMock(
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
        original_waveform = self.waveform_generator.frequency_domain_strain(
            parameters=self.simulation_parameters
        )
        new_waveform = self.waveform_generator.time_domain_strain()
        self.assertNotEqual(original_waveform, new_waveform)

    def test_frequency_domain_caching_and_using_time_domain_strain_with_parameters(
        self,
    ):
        original_waveform = self.waveform_generator.frequency_domain_strain(
            parameters=self.simulation_parameters
        )
        new_waveform = self.waveform_generator.time_domain_strain(
            parameters=self.simulation_parameters
        )
        self.assertNotEqual(original_waveform, new_waveform)

    def test_time_domain_caching_and_using_frequency_domain_strain_without_parameters(
        self,
    ):
        original_waveform = self.waveform_generator.time_domain_strain(
            parameters=self.simulation_parameters
        )
        new_waveform = self.waveform_generator.frequency_domain_strain()
        self.assertNotEqual(original_waveform, new_waveform)

    def test_time_domain_caching_and_using_frequency_domain_strain_with_parameters(
        self,
    ):
        original_waveform = self.waveform_generator.time_domain_strain(
            parameters=self.simulation_parameters
        )
        new_waveform = self.waveform_generator.frequency_domain_strain(
            parameters=self.simulation_parameters
        )
        self.assertNotEqual(original_waveform, new_waveform)


class TestWaveformDirectAgainstLALSIM(unittest.TestCase):
    def setUp(self):
        self.BBH_precessing_injection_parameters = dict(
            mass_1=36.0,
            mass_2=32.0,
            a_1=0.2,
            a_2=0.4,
            tilt_1=0.0,
            tilt_2=0.0,
            phi_12=0.0,
            phi_jl=0.0,
            luminosity_distance=4000.0,
            theta_jn=0.4,
            psi=2.659,
            phase=1.3 + np.pi / 2.0,
            geocent_time=1126259642.413,
            ra=1.375,
            dec=0.2108,
        )

        self.BNS_precessing_injection_parameters = dict(
            mass_1=36.0,
            mass_2=32.0,
            a_1=0.2,
            a_2=0.4,
            tilt_1=0.0,
            tilt_2=0.0,
            phi_12=0.0,
            phi_jl=0.0,
            luminosity_distance=4000.0,
            theta_jn=0.4,
            psi=2.659,
            phase=1.3 + np.pi / 2.0,
            geocent_time=1126259642.413,
            ra=1.375,
            dec=0.2108,
            lambda_1=1000,
            lambda_2=1500,
        )

    def test_IMRPhenomPv2(self):
        waveform_approximant = "IMRPhenomPv2"
        self.run_for_approximant(waveform_approximant, source="bbh")

    def test_IMRPhenomD(self):
        waveform_approximant = "IMRPhenomD"
        self.run_for_approximant(waveform_approximant, source="bbh")

    def test_IMRPhenomPv2_NRTidal(self):
        waveform_approximant = "IMRPhenomPv2_NRTidal"
        self.run_for_approximant(waveform_approximant, source="bns")

    def test_IMRPhenomD_NRTidal(self):
        waveform_approximant = "IMRPhenomD_NRTidal"
        self.run_for_approximant(waveform_approximant, source="bns")

    def test_TaylorF2(self):
        waveform_approximant = "TaylorF2"
        self.run_for_approximant(waveform_approximant, source="bns")

    def run_for_approximant(self, waveform_approximant, source):

        if source == "bbh":
            injection_parameters = self.BBH_precessing_injection_parameters
            frequency_domain_source_model = bilby.gw.source.lal_binary_black_hole
        elif source == "bns":
            injection_parameters = self.BNS_precessing_injection_parameters
            frequency_domain_source_model = bilby.gw.source.lal_binary_neutron_star

        # create a waveform generator for bilby
        duration = 4.0
        sampling_frequency = 2048.0
        reference_frequency = 20.0
        minimum_frequency = 20.0
        # Fixed arguments passed into the source model

        waveform_arguments = dict(
            waveform_approximant=waveform_approximant,
            reference_frequency=reference_frequency,
            minimum_frequency=minimum_frequency,
        )

        (
            iota,
            spin_1x,
            spin_1y,
            spin_1z,
            spin_2x,
            spin_2y,
            spin_2z,
        ) = bilby.gw.conversion.bilby_to_lalsimulation_spins(
            theta_jn=injection_parameters["theta_jn"],
            phi_jl=injection_parameters["phi_jl"],
            tilt_1=injection_parameters["tilt_1"],
            tilt_2=injection_parameters["tilt_2"],
            phi_12=injection_parameters["phi_12"],
            a_1=injection_parameters["a_1"],
            a_2=injection_parameters["a_2"],
            mass_1=injection_parameters["mass_1"],
            mass_2=injection_parameters["mass_2"],
            reference_frequency=reference_frequency,
            phase=injection_parameters["phase"],
        )

        waveform_generator = bilby.gw.WaveformGenerator(
            duration=duration,
            sampling_frequency=sampling_frequency,
            frequency_domain_source_model=frequency_domain_source_model,
            waveform_arguments=waveform_arguments,
        )

        bilby_strain = waveform_generator.frequency_domain_strain(
            parameters=injection_parameters
        )

        # LALSIM Waveform

        lambda_1 = injection_parameters.get("lambda_1", None)
        lambda_2 = injection_parameters.get("lambda_2", None)

        get_lalsim_waveform = lalsim_FD_waveform(
            injection_parameters["mass_1"],
            injection_parameters["mass_2"],
            spin_1x,
            spin_1y,
            spin_1z,
            spin_2x,
            spin_2y,
            spin_2z,
            iota,
            injection_parameters["phase"],
            duration,
            injection_parameters["luminosity_distance"],
            (waveform_generator.frequency_array)[-1],
            lambda_1,
            lambda_2,
            **waveform_arguments
        )

        h_plus = get_lalsim_waveform["plus"]
        h_cross = get_lalsim_waveform["cross"]

        if waveform_approximant == "TaylorF2":
            upper_freq = ISCO(
                injection_parameters["mass_1"], injection_parameters["mass_2"]
            )

        else:
            upper_freq = waveform_generator.frequency_array[-1]

        # Frequency resolution
        delta_f = 1.0 / duration
        # length of PSD
        f_len = int((2 * sampling_frequency) / delta_f)

        # PSD aLIGO
        psd_aLIGO = generate_PSD(
            psd_name="aLIGOZeroDetHighPower", length=f_len, delta_f=delta_f
        )

        norm_hp_bilby = normalize_strain(
            bilby_strain["plus"],
            psd=psd_aLIGO.data.data,
            delta_f=delta_f,
            lower_cut_off=minimum_frequency,
            upper_cut_off=upper_freq,
        )

        norm_hc_bilby = normalize_strain(
            bilby_strain["cross"],
            psd=psd_aLIGO.data.data,
            delta_f=delta_f,
            lower_cut_off=minimum_frequency,
            upper_cut_off=upper_freq,
        )

        norm_hp_lalsim = normalize_strain(
            h_plus,
            psd=psd_aLIGO.data.data,
            delta_f=delta_f,
            lower_cut_off=minimum_frequency,
            upper_cut_off=upper_freq,
        )

        norm_hc_lalsim = normalize_strain(
            h_cross,
            psd=psd_aLIGO.data.data,
            delta_f=delta_f,
            lower_cut_off=minimum_frequency,
            upper_cut_off=upper_freq,
        )

        # Match/Overpal between polarizations of lalsim and Bilby
        match_Hplus = overlap(
            bilby_strain["plus"],
            h_plus,
            psd=psd_aLIGO.data.data,
            delta_f=delta_f,
            lower_cut_off=minimum_frequency,
            upper_cut_off=upper_freq,
            norm1=norm_hp_bilby,
            norm2=norm_hp_lalsim,
        )

        match_Hcross = overlap(
            bilby_strain["cross"],
            h_cross,
            psd=psd_aLIGO.data.data,
            delta_f=delta_f,
            lower_cut_off=minimum_frequency,
            upper_cut_off=upper_freq,
            norm1=norm_hc_bilby,
            norm2=norm_hc_lalsim,
        )

        self.assertAlmostEqual(match_Hplus, 1, places=4)
        self.assertAlmostEqual(match_Hcross, 1, places=4)


def ISCO(m1, m2):
    return 1.0 / (6.0 * np.sqrt(6.0) * np.pi * (m1 + m2) * lal.MTSUN_SI)


def lalsim_FD_waveform(
    m1,
    m2,
    s1x,
    s1y,
    s1z,
    s2x,
    s2y,
    s2z,
    theta_jn,
    phase,
    duration,
    dL,
    fmax,
    lambda_1=None,
    lambda_2=None,
    **kwarg
):
    mass1 = m1 * lal.MSUN_SI
    mass2 = m2 * lal.MSUN_SI
    spin_1x = s1x
    spin_1y = s1y
    spin_1z = s1z
    spin_2x = s2x
    spin_2y = s2y
    spin_2z = s2z
    iota = theta_jn
    phaseC = phase  # Phase is hard coded to be zero

    eccentricity = 0
    longitude_ascending_nodes = 0
    mean_per_ano = 0

    waveform_arg = dict(minimum_freq=20.0, reference_frequency=20)
    waveform_arg.update(kwarg)
    dL = dL * lal.PC_SI * 1e6  # MPC --> Km
    approximant = lalsim.GetApproximantFromString(waveform_arg["waveform_approximant"])
    flow = waveform_arg["minimum_freq"]
    delta_freq = 1.0 / duration
    maximum_frequency = fmax  # 1024.0 # ISCO(m1, m2)
    fref = waveform_arg["reference_frequency"]
    waveform_dictionary = lal.CreateDict()

    if lambda_1 is not None:
        lalsim.SimInspiralWaveformParamsInsertTidalLambda1(
            waveform_dictionary, float(lambda_1)
        )
    if lambda_2 is not None:
        lalsim.SimInspiralWaveformParamsInsertTidalLambda2(
            waveform_dictionary, float(lambda_2)
        )

    hplus, hcross = lalsim.SimInspiralChooseFDWaveform(
        mass1,
        mass2,
        spin_1x,
        spin_1y,
        spin_1z,
        spin_2x,
        spin_2y,
        spin_2z,
        dL,
        iota,
        phaseC,
        longitude_ascending_nodes,
        eccentricity,
        mean_per_ano,
        delta_freq,
        flow,
        maximum_frequency,
        fref,
        waveform_dictionary,
        approximant,
    )

    h_plus = hplus.data.data[:]
    h_cross = hcross.data.data[:]

    return {"plus": h_plus, "cross": h_cross}


# Function for PSD list
def get_lalsim_psd_list():
    PSD_prefix = "SimNoisePSD"
    PSD_suffix = "Ptr"
    blacklist = [
        "FromFile",
        "MirrorTherm",
        "Quantum",
        "Seismic",
        "Shot",
        "SuspTherm",
        "TAMA",
        "GEO",
        "GEOHF",
        "aLIGOThermal",
    ]
    psd_list = []
    # Avoid the string 'SimNoisePSD'
    for name in lalsim.__dict__:
        if (
            name != PSD_prefix
            and name.startswith(PSD_prefix)
            and not name.endswith(PSD_suffix)
        ):
            # if name in blacklist:
            name = name[len(PSD_prefix) :]
            if (
                name not in blacklist
                and not name.startswith("iLIGO")
                and not name.startswith("eLIGO")
            ):
                psd_list.append(name)
    return sorted(psd_list)


# Function te generate PSDs
def generate_PSD(psd_name="aLIGOZeroDetHighPower", length=None, delta_f=None):
    psd_list = get_lalsim_psd_list()

    if psd_name in psd_list:
        # print (psd_name)
        # Function for PSD
        func = lalsim.__dict__["SimNoisePSD" + psd_name + "Ptr"]
        # Generate a lal frequency series
        PSDseries = lal.CreateREAL8FrequencySeries(
            "", lal.LIGOTimeGPS(0), 0, delta_f, lal.DimensionlessUnit, length
        )
        # func(PSDseries)
        lalsim.SimNoisePSD(PSDseries, 0, func)
    return PSDseries


# Functions to compute match/overlap
def overlap(
    signal1,
    signal2,
    psd=None,
    delta_f=None,
    lower_cut_off=None,
    upper_cut_off=None,
    norm1=None,
    norm2=None,
):
    low_index = int(lower_cut_off / delta_f)
    up_index = int(upper_cut_off / delta_f)
    integrand = np.conj(signal1) * signal2
    integrand = integrand[low_index:up_index] / psd[low_index:up_index]
    integral = (4 * delta_f * integrand) / norm1 / norm2
    return sum(integral).real


# Normalizing a waveform
def normalize_strain(
    signal, psd=None, delta_f=None, lower_cut_off=None, upper_cut_off=None
):
    low_index = int(lower_cut_off / delta_f)
    up_index = int(upper_cut_off / delta_f)
    integrand = np.conj(signal) * signal
    integrand = integrand[low_index:up_index] / psd[low_index:up_index]
    integral = sum(4 * delta_f * integrand)
    return np.sqrt(integral).real


if __name__ == "__main__":
    unittest.main()
