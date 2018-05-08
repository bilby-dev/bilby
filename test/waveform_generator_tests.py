import unittest
from context import tupak
import numpy as np


def gaussian_frequency_domain_strain(frequency_array, amplitude, mu, sigma, ra, dec, geocent_time, psi):
    ht = {'plus': amplitude * np.exp(-(mu - frequency_array) ** 2 / sigma ** 2 / 2),
          'cross': amplitude * np.exp(-(mu - frequency_array) ** 2 / sigma ** 2 / 2)}
    return ht

def gaussian_frequency_domain_strain_2(frequency_array, a, m, s, ra, dec, geocent_time, psi):
    ht = {'plus': a * np.exp(-(m - frequency_array) ** 2 / s ** 2 / 2),
          'cross': a * np.exp(-(m - frequency_array) ** 2 / s ** 2 / 2)}
    return ht


class TestWaveformGeneratorInstantiationWithoutOptionalParameters(unittest.TestCase):

    def setUp(self):
        self.waveform_generator = tupak.waveform_generator.WaveformGenerator(
            frequency_domain_source_model=gaussian_frequency_domain_strain)
        self.simulation_parameters = dict(amplitude=1e-21, mu=100, sigma=1,
                                     ra=1.375,
                                     dec=-1.2108,
                                     geocent_time=1126259642.413,
                                     psi=2.659)

    def tearDown(self):
        del self.waveform_generator
        del self.simulation_parameters

    def test_time_duration(self):
        self.assertEqual(self.waveform_generator.time_duration, 1)

    def test_sampling_frequency(self):
        self.assertEqual(self.waveform_generator.sampling_frequency, 4096)

    def test_source_model(self):
        self.assertEqual(self.waveform_generator.source_model, gaussian_frequency_domain_strain)

    def test_frequency_array_type(self):
        self.assertIsInstance(self.waveform_generator.frequency_array, np.ndarray)

    def test_time_array_type(self):
        self.assertIsInstance(self.waveform_generator.time_array, np.ndarray)

    def test_source_model_parameters(self):
        params = self.simulation_parameters.keys()
        self.assertItemsEqual(self.waveform_generator.parameters, params)


class TestParameterSetter(unittest.TestCase):

    def setUp(self):
        self.waveform_generator = tupak.waveform_generator.WaveformGenerator(
            frequency_domain_source_model=gaussian_frequency_domain_strain)
        self.simulation_parameters = dict(amplitude=1e-21, mu=100, sigma=1,
                                     ra=1.375,
                                     dec=-1.2108,
                                     geocent_time=1126259642.413,
                                     psi=2.659)

    def tearDown(self):
        del self.waveform_generator
        del self.simulation_parameters

    def test_parameter_setter_sets_expected_values_with_expected_keys(self):
        self.waveform_generator.parameters = self.simulation_parameters
        for key in self.simulation_parameters:
            self.assertEqual(self.waveform_generator.parameters[key], self.simulation_parameters[key])

    def test_parameter_setter_with_unexpected_keys(self):
        self.waveform_generator.parameters['foo'] = 1337

        def parameter_setter(wg_params, sim_params):
            wg_params = sim_params
        self.assertRaises(KeyError, parameter_setter(self.waveform_generator.parameters, self.simulation_parameters))

    def test_parameter_setter_raises_type_error(self):
        a = 4

        def parameter_setter(wg_params, sim_params):
            wg_params = sim_params
        self.assertRaises(TypeError, parameter_setter(self.waveform_generator.parameters, a))

    def test_parameter_setter_none_handling(self):
        self.waveform_generator.parameters = None
        self.assertItemsEqual(self.waveform_generator.parameters.keys(), self.simulation_parameters.keys())


class TestSourceModelSetter(unittest.TestCase):

    def setUp(self):
        self.waveform_generator = tupak.waveform_generator.WaveformGenerator(
            frequency_domain_source_model=gaussian_frequency_domain_strain)
        self.waveform_generator.source_model = gaussian_frequency_domain_strain_2
        self.simulation_parameters = dict(a=1e-21, m=100, s=1,
                                     ra=1.375,
                                     dec=-1.2108,
                                     geocent_time=1126259642.413,
                                     psi=2.659)

    def tearDown(self):
        del self.waveform_generator
        del self.simulation_parameters

    def test_parameters_are_set_correctly(self):
        self.assertItemsEqual(self.waveform_generator.parameters, self.simulation_parameters.keys())


if __name__ == '__main__':
    unittest.main()