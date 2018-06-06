from __future__ import absolute_import
import unittest
import tupak
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
        self.waveform_generator = \
            tupak.gw.waveform_generator.WaveformGenerator(1, 4096,
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
        self.assertEqual(self.waveform_generator.frequency_domain_source_model, gaussian_frequency_domain_strain)

    def test_frequency_array_type(self):
        self.assertIsInstance(self.waveform_generator.frequency_array, np.ndarray)

    def test_time_array_type(self):
        self.assertIsInstance(self.waveform_generator.time_array, np.ndarray)

    def test_source_model_parameters(self):
        self.assertListEqual(sorted(list(self.waveform_generator.parameters.keys())),
                             sorted(list(self.simulation_parameters.keys())))


class TestParameterSetter(unittest.TestCase):

    def setUp(self):
        self.waveform_generator = \
            tupak.gw.waveform_generator.WaveformGenerator(1, 4096,
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

    def test_parameter_setter_none_handling(self):
        self.waveform_generator.parameters = None
        self.assertListEqual(sorted(list(self.waveform_generator.parameters.keys())),
                             sorted(list(self.simulation_parameters.keys())))


class TestSourceModelSetter(unittest.TestCase):

    def setUp(self):
        self.waveform_generator = tupak.gw.waveform_generator.WaveformGenerator(1, 4096,
                                                                                frequency_domain_source_model=gaussian_frequency_domain_strain)
        self.waveform_generator.frequency_domain_source_model = gaussian_frequency_domain_strain_2
        self.simulation_parameters = dict(amplitude=1e-21, mu=100, sigma=1,
                                          ra=1.375,
                                          dec=-1.2108,
                                          geocent_time=1126259642.413,
                                          psi=2.659)

    def tearDown(self):
        del self.waveform_generator
        del self.simulation_parameters

    def test_parameters_are_set_correctly(self):
        self.assertListEqual(sorted(list(self.waveform_generator.parameters.keys())),
                             sorted(list(self.simulation_parameters.keys())))


if __name__ == '__main__':
    unittest.main()
