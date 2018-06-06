from __future__ import absolute_import
from test.context import tupak
import unittest
import mock
from mock import MagicMock
import numpy as np
import logging


class TestDetector(unittest.TestCase):

    def setUp(self):
        self.name = 'name'
        self.power_spectral_density = MagicMock()
        self.minimum_frequency = 10
        self.maximum_frequency = 20
        self.length = 30
        self.latitude = 1
        self.longitude = 2
        self.elevation = 3
        self.xarm_azimuth = 4
        self.yarm_azimuth = 5
        self.xarm_tilt = 0.
        self.yarm_tilt = 0.
        # noinspection PyTypeChecker
        self.ifo = tupak.detector.Interferometer(name=self.name, power_spectral_density=self.power_spectral_density,
                                           minimum_frequency=self.minimum_frequency,
                                           maximum_frequency=self.maximum_frequency, length=self.length,
                                           latitude=self.latitude, longitude=self.longitude, elevation=self.elevation,
                                           xarm_azimuth=self.xarm_azimuth, yarm_azimuth=self.yarm_azimuth,
                                           xarm_tilt=self.xarm_tilt, yarm_tilt=self.yarm_tilt)

    def tearDown(self):
        del self.name
        del self.power_spectral_density
        del self.minimum_frequency
        del self.maximum_frequency
        del self.length
        del self.latitude
        del self.longitude
        del self.elevation
        del self.xarm_azimuth
        del self.yarm_azimuth
        del self.xarm_tilt
        del self.yarm_tilt
        del self.ifo

    def test_name_setting(self):
        self.assertEqual(self.ifo.name, self.name)

    def test_psd_setting(self):
        self.assertEqual(self.ifo.power_spectral_density, self.power_spectral_density)

    def test_min_freq_setting(self):
        self.assertEqual(self.ifo.minimum_frequency, self.minimum_frequency)

    def test_max_freq_setting(self):
        self.assertEqual(self.ifo.maximum_frequency, self.maximum_frequency)

    def test_length_setting(self):
        self.assertEqual(self.ifo.length, self.length)

    def test_latitude_setting(self):
        self.assertEqual(self.ifo.latitude, self.latitude)

    def test_longitude_setting(self):
        self.assertEqual(self.ifo.longitude, self.longitude)

    def test_elevation_setting(self):
        self.assertEqual(self.ifo.elevation, self.elevation)

    def test_xarm_azi_setting(self):
        self.assertEqual(self.ifo.xarm_azimuth, self.xarm_azimuth)

    def test_yarm_azi_setting(self):
        self.assertEqual(self.ifo.yarm_azimuth, self.yarm_azimuth)

    def test_xarm_tilt_setting(self):
        self.assertEqual(self.ifo.xarm_tilt, self.xarm_tilt)

    def test_yarm_tilt_setting(self):
        self.assertEqual(self.ifo.yarm_tilt, self.yarm_tilt)

    def test_data_init(self):
        self.assertTrue(np.array_equal(self.ifo.data, np.array([])))

    def test_frequency_array_init(self):
        self.assertTrue(np.array_equal(self.ifo.frequency_array, np.array([])))

    def test_sampling_frequency_init(self):
        self.assertIsNone(self.ifo.sampling_frequency)

    def test_sampling_duration_init(self):
        self.assertIsNone(self.ifo.duration)

    def test_epoch_init(self):
        self.assertEqual(self.ifo.epoch, 0)

    def test_frequency_mask(self):
        self.ifo.frequency_array = np.array([8, 10, 13, 15, 20, 22])
        self.assertTrue(np.array_equal(self.ifo.frequency_mask, [False, False, True, True, False, False]))

    def test_vertex_without_update(self):
        _ = self.ifo.vertex
        with mock.patch('tupak.utils.get_vertex_position_geocentric') as m:
            m.return_value = np.array([1])
            self.assertFalse(np.array_equal(self.ifo.vertex, np.array([1])))

    def test_vertex_with_latitude_update(self):
        with mock.patch('tupak.utils.get_vertex_position_geocentric') as m:
            m.return_value = np.array([1])
            self.ifo.latitude = 5
            self.assertEqual(self.ifo.vertex, np.array([1]))

    def test_vertex_with_longitude_update(self):
        with mock.patch('tupak.utils.get_vertex_position_geocentric') as m:
            m.return_value = np.array([1])
            self.ifo.longitude = 5
            self.assertEqual(self.ifo.vertex, np.array([1]))

    def test_vertex_with_elevation_update(self):
        with mock.patch('tupak.utils.get_vertex_position_geocentric') as m:
            m.return_value = np.array([1])
            self.ifo.elevation = 5
            self.assertEqual(self.ifo.vertex, np.array([1]))

    def test_x_without_update(self):
        _ = self.ifo.x
        self.ifo.unit_vector_along_arm = MagicMock(return_value=np.array([1]))

        self.assertFalse(np.array_equal(self.ifo.x,
                                        np.array([1])))

    def test_x_with_xarm_tilt_update(self):
        self.ifo.unit_vector_along_arm = MagicMock(return_value=np.array([1]))
        self.ifo.xarm_tilt = 0
        self.assertTrue(np.array_equal(self.ifo.x,
                                       np.array([1])))

    def test_x_with_xarm_azimuth_update(self):
        self.ifo.unit_vector_along_arm = MagicMock(return_value=np.array([1]))
        self.ifo.xarm_azimuth = 0
        self.assertTrue(np.array_equal(self.ifo.x,
                                       np.array([1])))

    def test_x_with_longitude_update(self):
        self.ifo.unit_vector_along_arm = MagicMock(return_value=np.array([1]))
        self.ifo.longitude = 0
        self.assertTrue(np.array_equal(self.ifo.x,
                                       np.array([1])))

    def test_x_with_latitude_update(self):
        self.ifo.unit_vector_along_arm = MagicMock(return_value=np.array([1]))
        self.ifo.latitude = 0
        self.assertTrue(np.array_equal(self.ifo.x,
                                       np.array([1])))

    def test_y_without_update(self):
        _ = self.ifo.y
        self.ifo.unit_vector_along_arm = MagicMock(return_value=np.array([1]))

        self.assertFalse(np.array_equal(self.ifo.y,
                                        np.array([1])))

    def test_y_with_yarm_tilt_update(self):
        self.ifo.unit_vector_along_arm = MagicMock(return_value=np.array([1]))
        self.ifo.yarm_tilt = 0
        self.assertTrue(np.array_equal(self.ifo.y,
                                       np.array([1])))

    def test_y_with_yarm_azimuth_update(self):
        self.ifo.unit_vector_along_arm = MagicMock(return_value=np.array([1]))
        self.ifo.yarm_azimuth = 0
        self.assertTrue(np.array_equal(self.ifo.y,
                                       np.array([1])))

    def test_y_with_longitude_update(self):
        self.ifo.unit_vector_along_arm = MagicMock(return_value=np.array([1]))
        self.ifo.longitude = 0
        self.assertTrue(np.array_equal(self.ifo.y,
                                       np.array([1])))

    def test_y_with_latitude_update(self):
        self.ifo.unit_vector_along_arm = MagicMock(return_value=np.array([1]))
        self.ifo.latitude = 0
        self.assertTrue(np.array_equal(self.ifo.y,
                                       np.array([1])))

    def test_detector_tensor_without_update(self):
        _ = self.ifo.detector_tensor
        with mock.patch('numpy.einsum') as m:
            m.return_value = 1
            self.assertIsInstance(self.ifo.detector_tensor, np.ndarray)

    def test_detector_tensor_with_x_update(self):
        with mock.patch('numpy.einsum') as m:
            m.return_value = 1
            self.ifo.xarm_azimuth = 12
            self.assertEqual(self.ifo.detector_tensor, 0)

    def test_detector_tensor_with_y_update(self):
        with mock.patch('numpy.einsum') as m:
            m.return_value = 1
            self.ifo.yarm_azimuth = 12
            self.assertEqual(self.ifo.detector_tensor, 0)

    def test_amplitude_spectral_density_array(self):
        self.ifo.power_spectral_density.power_spectral_density_interpolated = MagicMock(return_value=np.array([1, 4]))
        self.assertTrue(np.array_equal(self.ifo.amplitude_spectral_density_array, np.array([1, 2])))

    def test_power_spectral_density_array(self):
        self.ifo.power_spectral_density.power_spectral_density_interpolated = MagicMock(return_value=np.array([1, 4]))
        self.assertTrue(np.array_equal(self.ifo.power_spectral_density_array, np.array([1, 4])))

    def test_antenna_response_default(self):
        with mock.patch('tupak.utils.get_polarization_tensor') as m:
            with mock.patch('numpy.einsum') as n:
                m.return_value = 0
                n.return_value = 1
                self.assertEqual(self.ifo.antenna_response(234, 52, 54, 76, 'plus'), 1)

    def test_antenna_response_einsum(self):
        with mock.patch('tupak.utils.get_polarization_tensor') as m:
            m.return_value = np.ones((3, 3))
            self.assertAlmostEqual(self.ifo.antenna_response(234, 52, 54, 76, 'plus'), self.ifo.detector_tensor.sum())

    def test_get_detector_response_default_behaviour(self):
        self.ifo.antenna_response = MagicMock(return_value=1)
        self.ifo.time_delay_from_geocenter = MagicMock(return_value = 0)
        self.ifo.epoch = 0
        self.minimum_frequency = 10
        self.maximum_frequency = 20
        self.ifo.frequency_array = np.array([8, 12, 16, 20, 24])
        plus = np.array([1, 2, 3, 4, 5])
        response = self.ifo.get_detector_response(
            waveform_polarizations=dict(plus=plus),
            parameters=dict(ra=0, dec=0, geocent_time=0, psi=0))
        self.assertTrue(np.array_equal(response, plus*self.ifo.frequency_mask*np.exp(-0j)))

    def test_get_detector_response_with_dt(self):
        self.ifo.antenna_response = MagicMock(return_value=1)
        self.ifo.time_delay_from_geocenter = MagicMock(return_value = 0)
        self.ifo.epoch = 1
        self.minimum_frequency = 10
        self.maximum_frequency = 20
        self.ifo.frequency_array = np.array([8, 12, 16, 20, 24])
        plus = np.array([1, 2, 3, 4, 5])
        response = self.ifo.get_detector_response(
            waveform_polarizations=dict(plus=plus),
            parameters=dict(ra=0, dec=0, geocent_time=0, psi=0))
        self.assertTrue(np.array_equal(response, plus*self.ifo.frequency_mask*np.exp(-1j*2*np.pi*self.ifo.frequency_array)))

    def test_get_detector_response_multiple_modes(self):
        self.ifo.antenna_response = MagicMock(return_value=1)
        self.ifo.time_delay_from_geocenter = MagicMock(return_value = 0)
        self.ifo.epoch = 0
        self.minimum_frequency = 10
        self.maximum_frequency = 20
        self.ifo.frequency_array = np.array([8, 12, 16, 20, 24])
        plus = np.array([1, 2, 3, 4, 5])
        cross = np.array([6, 7, 8, 9, 10])
        response = self.ifo.get_detector_response(
            waveform_polarizations=dict(plus=plus, cross=cross),
            parameters=dict(ra=0, dec=0, geocent_time=0, psi=0))
        self.assertTrue(np.array_equal(response, (plus+cross)*self.ifo.frequency_mask*np.exp(-0j)))

    def test_inject_signal_no_waveform_polarizations(self):
        with mock.patch('logging.warning') as m:
            m.side_effect = KeyError('foo')
            with self.assertRaises(KeyError):
                self.ifo.inject_signal(waveform_polarizations=None, parameters=None)

    def test_inject_signal_sets_data_with_existing_data_array(self):
        self.ifo.get_detector_response = MagicMock(return_value=np.array([1]))
        self.ifo.frequency_array = np.array([0, 1])
        with mock.patch('tupak.utils.optimal_snr_squared') as m:
            with mock.patch('tupak.utils.matched_filter_snr_squared') as n:
                m.return_value = 0
                n.return_value = 0
                self.ifo.data = np.array([1])
                self.ifo.inject_signal(waveform_polarizations='foo', parameters=None)
                self.assertEqual(self.ifo.data, np.array([2]))

    def test_inject_signal_sets_data_without_data_array(self):
        self.ifo.get_detector_response = MagicMock(return_value=np.array([1]))
        self.ifo.frequency_array = np.array([0, 1])
        with mock.patch('tupak.utils.optimal_snr_squared') as m:
            with mock.patch('tupak.utils.matched_filter_snr_squared') as n:
                m.return_value = 0
                n.return_value = 0
                self.ifo.data = 1
                self.ifo.data = np.array([])
                self.ifo.inject_signal(waveform_polarizations='foo', parameters=None)
                self.assertEqual(self.ifo.data, np.array([1]))

    def test_unit_vector_along_arm_default(self):
        with mock.patch('logging.warning') as m:
            m.side_effect = KeyError('foo')
            with self.assertRaises(KeyError):
                self.ifo.unit_vector_along_arm('z')

    def test_unit_vector_along_arm_x(self):
        with mock.patch('numpy.array') as m:
            m.return_value = 1
            self.ifo.xarm_tilt = 0
            self.ifo.xarm_azimuth = 0
            self.ifo.yarm_tilt = 0
            self.ifo.yarm_azimuth = np.pi
            self.assertAlmostEqual(self.ifo.unit_vector_along_arm('x'), 1)

    def test_unit_vector_along_arm_y(self):
        with mock.patch('numpy.array') as m:
            m.return_value = 1
            self.ifo.xarm_tilt = 0
            self.ifo.xarm_azimuth = 0
            self.ifo.yarm_tilt = 0
            self.ifo.yarm_azimuth = np.pi
            self.assertAlmostEqual(self.ifo.unit_vector_along_arm('y'), -1)

    def test_set_data_raises_value_error(self):
        with self.assertRaises(ValueError):
            self.ifo.set_data(sampling_frequency=1, duration=1)

    def test_set_data_sets_data_from_frequency_domain_strain(self):
        with mock.patch('tupak.utils.create_frequency_series') as m:
            m.return_value = np.array([1])
            self.ifo.minimum_frequency = 0
            self.ifo.maximum_frequency = 3
            self.ifo.power_spectral_density.get_noise_realisation = MagicMock(return_value=(1, np.array([2])))
            self.ifo.set_data(sampling_frequency=1, duration=1, frequency_domain_strain=np.array([1]))
            self.assertTrue(np.array_equal(self.ifo.data, np.array([1])))

    def test_set_data_sets_frequencies_from_frequency_domain_strain(self):
        with mock.patch('tupak.utils.create_frequency_series') as m:
            m.return_value = np.array([1])
            self.ifo.minimum_frequency = 0
            self.ifo.maximum_frequency = 3
            self.ifo.power_spectral_density.get_noise_realisation = MagicMock(return_value=(1, np.array([2])))
            self.ifo.set_data(sampling_frequency=1, duration=1, frequency_domain_strain=np.array([1]))
            self.assertTrue(np.array_equal(self.ifo.frequency_array, np.array([1])))

    def test_set_data_sets_frequencies_from_spectral_density(self):
        with mock.patch('tupak.utils.create_frequency_series') as m:
            m.return_value = np.array([1])
            self.ifo.minimum_frequency = 0
            self.ifo.maximum_frequency = 3
            self.ifo.power_spectral_density.get_noise_realisation = MagicMock(return_value=(1, np.array([2])))
            self.ifo.set_data(sampling_frequency=1, duration=1, from_power_spectral_density=True)
            self.assertTrue(np.array_equal(self.ifo.frequency_array, np.array([2])))

    def test_set_data_sets_epoch(self):
        with mock.patch('tupak.utils.create_frequency_series') as m:
            m.return_value = np.array([1])
            self.ifo.minimum_frequency = 0
            self.ifo.maximum_frequency = 3
            self.ifo.power_spectral_density.get_noise_realisation = MagicMock(return_value=(1, np.array([2])))
            self.ifo.set_data(sampling_frequency=1, duration=1, from_power_spectral_density=True, epoch=4)
            self.assertEqual(self.ifo.epoch, 4)

    def test_set_data_sets_sampling_frequency(self):
        with mock.patch('tupak.utils.create_frequency_series') as m:
            m.return_value = np.array([1])
            self.ifo.minimum_frequency = 0
            self.ifo.maximum_frequency = 3
            self.ifo.power_spectral_density.get_noise_realisation = MagicMock(return_value=(1, np.array([2])))
            self.ifo.set_data(sampling_frequency=1, duration=1, from_power_spectral_density=True, epoch=4)
            self.assertEqual(self.ifo.sampling_frequency, 1)

    def test_set_data_sets_duration(self):
        with mock.patch('tupak.utils.create_frequency_series') as m:
            m.return_value = np.array([1])
            self.ifo.minimum_frequency = 0
            self.ifo.maximum_frequency = 3
            self.ifo.power_spectral_density.get_noise_realisation = MagicMock(return_value=(1, np.array([2])))
            self.ifo.set_data(sampling_frequency=1, duration=1, from_power_spectral_density=True, epoch=4)
            self.assertEqual(self.ifo.duration, 1)

    def test_time_delay_from_geocenter(self):
        with mock.patch('tupak.utils.time_delay_geocentric') as m:
            m.return_value = 1
            self.assertEqual(self.ifo.time_delay_from_geocenter(1, 2, 3), 1)

    def test_vertex_position_geocentric(self):
        with mock.patch('tupak.utils.get_vertex_position_geocentric') as m:
            m.return_value = 1
            self.assertEqual(self.ifo.vertex_position_geocentric(), 1)

    def test_whitened_data(self):
        self.ifo.data = np.array([2])
        self.ifo.power_spectral_density.power_spectral_density_interpolated = MagicMock(return_value=np.array([1]))
        self.assertTrue(np.array_equal(self.ifo.whitened_data, np.array([2])))


if __name__ == '__main__':
    unittest.main()
