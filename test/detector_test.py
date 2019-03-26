from __future__ import absolute_import

import bilby
import unittest
import mock
from mock import MagicMock
from mock import patch
import numpy as np
import scipy.signal.windows
import os
import sys
from shutil import rmtree
import logging
import deepdish as dd


class TestInterferometer(unittest.TestCase):

    def setUp(self):
        self.name = 'name'
        self.power_spectral_density = bilby.gw.detector.PowerSpectralDensity.from_aligo()
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
        self.ifo = bilby.gw.detector.Interferometer(name=self.name, power_spectral_density=self.power_spectral_density,
                                                    minimum_frequency=self.minimum_frequency,
                                                    maximum_frequency=self.maximum_frequency, length=self.length,
                                                    latitude=self.latitude, longitude=self.longitude,
                                                    elevation=self.elevation,
                                                    xarm_azimuth=self.xarm_azimuth, yarm_azimuth=self.yarm_azimuth,
                                                    xarm_tilt=self.xarm_tilt, yarm_tilt=self.yarm_tilt)
        self.ifo.strain_data.set_from_frequency_domain_strain(
            np.linspace(0, 4096, 4097), sampling_frequency=4096, duration=2)
        self.outdir = 'outdir'
        bilby.core.utils.check_directory_exists_and_if_not_mkdir(self.outdir)

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
        rmtree(self.outdir)

    def test_name_setting(self):
        self.assertEqual(self.ifo.name, self.name)

    def test_psd_setting(self):
        self.assertEqual(self.ifo.power_spectral_density, self.power_spectral_density)

    def test_min_freq_setting(self):
        self.assertEqual(self.ifo.strain_data.minimum_frequency, self.minimum_frequency)

    def test_max_freq_setting(self):
        self.assertEqual(self.ifo.strain_data.maximum_frequency, self.maximum_frequency)

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

    def test_vertex_without_update(self):
        _ = self.ifo.vertex
        with mock.patch('bilby.gw.utils.get_vertex_position_geocentric') as m:
            m.return_value = np.array([1])
            self.assertFalse(np.array_equal(self.ifo.vertex, np.array([1])))

    def test_vertex_with_latitude_update(self):
        with mock.patch('bilby.gw.utils.get_vertex_position_geocentric') as m:
            m.return_value = np.array([1])
            self.ifo.latitude = 5
            self.assertEqual(self.ifo.vertex, np.array([1]))

    def test_vertex_with_longitude_update(self):
        with mock.patch('bilby.gw.utils.get_vertex_position_geocentric') as m:
            m.return_value = np.array([1])
            self.ifo.longitude = 5
            self.assertEqual(self.ifo.vertex, np.array([1]))

    def test_vertex_with_elevation_update(self):
        with mock.patch('bilby.gw.utils.get_vertex_position_geocentric') as m:
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
            expected = np.array([[-9.24529394e-06, 1.02425803e-04, 3.24550668e-04],
                                 [1.02425803e-04, 1.37390844e-03, -8.61137566e-03],
                                 [3.24550668e-04, -8.61137566e-03, -1.36466315e-03]])
            self.assertTrue(np.allclose(expected, self.ifo.detector_tensor))

    def test_detector_tensor_with_x_azimuth_update(self):
        _ = self.ifo.detector_tensor
        with mock.patch('numpy.einsum') as m:
            m.return_value = 1
            self.ifo.xarm_azimuth = 1
            self.assertEqual(0, self.ifo.detector_tensor)

    def test_detector_tensor_with_y_azimuth_update(self):
        _ = self.ifo.detector_tensor
        with mock.patch('numpy.einsum') as m:
            m.return_value = 1
            self.ifo.yarm_azimuth = 1
            self.assertEqual(0, self.ifo.detector_tensor)

    def test_detector_tensor_with_x_tilt_update(self):
        _ = self.ifo.detector_tensor
        with mock.patch('numpy.einsum') as m:
            m.return_value = 1
            self.ifo.xarm_tilt = 1
            self.assertEqual(0, self.ifo.detector_tensor)

    def test_detector_tensor_with_y_tilt_update(self):
        _ = self.ifo.detector_tensor
        with mock.patch('numpy.einsum') as m:
            m.return_value = 1
            self.ifo.yarm_tilt = 1
            self.assertEqual(0, self.ifo.detector_tensor)

    def test_detector_tensor_with_longitude_update(self):
        with mock.patch('numpy.einsum') as m:
            m.return_value = 1
            self.ifo.longitude = 1
            self.assertEqual(0, self.ifo.detector_tensor)

    def test_detector_tensor_with_latitude_update(self):
        with mock.patch('numpy.einsum') as m:
            _ = self.ifo.detector_tensor
            m.return_value = 1
            self.ifo.latitude = 1
            self.assertEqual(self.ifo.detector_tensor, 0)

    def test_antenna_response_default(self):
        with mock.patch('bilby.gw.utils.get_polarization_tensor') as m:
            with mock.patch('numpy.einsum') as n:
                m.return_value = 0
                n.return_value = 1
                self.assertEqual(self.ifo.antenna_response(234, 52, 54, 76, 'plus'), 1)

    def test_antenna_response_einsum(self):
        with mock.patch('bilby.gw.utils.get_polarization_tensor') as m:
            m.return_value = np.ones((3, 3))
            self.assertAlmostEqual(self.ifo.antenna_response(234, 52, 54, 76, 'plus'), self.ifo.detector_tensor.sum())

    def test_get_detector_response_default_behaviour(self):
        self.ifo.antenna_response = MagicMock(return_value=1)
        self.ifo.time_delay_from_geocenter = MagicMock(return_value=0)
        self.ifo.epoch = 0
        self.minimum_frequency = 10
        self.maximum_frequency = 20
        # self.ifo.frequency_array = np.array([8, 12, 16, 20, 24])
        plus = np.linspace(0, 4096, 4097)
        response = self.ifo.get_detector_response(
            waveform_polarizations=dict(plus=plus),
            parameters=dict(ra=0, dec=0, geocent_time=0, psi=0))
        self.assertTrue(np.array_equal(response, plus * self.ifo.frequency_mask * np.exp(-0j)))

    def test_get_detector_response_with_dt(self):
        self.ifo.antenna_response = MagicMock(return_value=1)
        self.ifo.time_delay_from_geocenter = MagicMock(return_value=0)
        self.ifo.epoch = 1
        self.minimum_frequency = 10
        self.maximum_frequency = 20
        # self.ifo.frequency_array = np.array([8, 12, 16, 20, 24])
        plus = np.linspace(0, 4096, 4097)
        response = self.ifo.get_detector_response(
            waveform_polarizations=dict(plus=plus),
            parameters=dict(ra=0, dec=0, geocent_time=0, psi=0))
        expected_response = plus * self.ifo.frequency_mask * np.exp(-1j * 2 * np.pi * self.ifo.frequency_array)
        self.assertTrue(np.allclose(abs(response),
                                    abs(plus * self.ifo.frequency_mask * np.exp(
                                        -1j * 2 * np.pi * self.ifo.frequency_array))))

    def test_get_detector_response_multiple_modes(self):
        self.ifo.antenna_response = MagicMock(return_value=1)
        self.ifo.time_delay_from_geocenter = MagicMock(return_value=0)
        self.ifo.epoch = 0
        self.minimum_frequency = 10
        self.maximum_frequency = 20
        # self.ifo.frequency_array = np.array([8, 12, 16, 20, 24])
        plus = np.linspace(0, 4096, 4097)
        cross = np.linspace(0, 4096, 4097)
        response = self.ifo.get_detector_response(
            waveform_polarizations=dict(plus=plus, cross=cross),
            parameters=dict(ra=0, dec=0, geocent_time=0, psi=0))
        self.assertTrue(np.array_equal(response, (plus + cross) * self.ifo.frequency_mask * np.exp(-0j)))

    def test_inject_signal_no_waveform_polarizations(self):
        with self.assertRaises(ValueError):
            self.ifo.inject_signal(injection_polarizations=None, parameters=None)

    def test_unit_vector_along_arm_default(self):
        with self.assertRaises(ValueError):
            self.ifo.unit_vector_along_arm('z')

    def test_unit_vector_along_arm_x(self):
        with mock.patch('numpy.array') as m:
            m.return_value = 1
            self.ifo.xarm_tilt = 0
            self.ifo.xarm_azimuth = 0
            self.ifo.yarm_tilt = 0
            self.ifo.yarm_azimuth = 90
            self.assertAlmostEqual(self.ifo.unit_vector_along_arm('x'), 1)

    def test_unit_vector_along_arm_y(self):
        with mock.patch('numpy.array') as m:
            m.return_value = 1
            self.ifo.xarm_tilt = 0
            self.ifo.xarm_azimuth = 90
            self.ifo.yarm_tilt = 0
            self.ifo.yarm_azimuth = 180
            self.assertAlmostEqual(self.ifo.unit_vector_along_arm('y'), -1)

    def test_time_delay_from_geocenter(self):
        with mock.patch('bilby.gw.utils.time_delay_geocentric') as m:
            m.return_value = 1
            self.assertEqual(self.ifo.time_delay_from_geocenter(1, 2, 3), 1)

    def test_vertex_position_geocentric(self):
        with mock.patch('bilby.gw.utils.get_vertex_position_geocentric') as m:
            m.return_value = 1
            self.assertEqual(self.ifo.vertex_position_geocentric(), 1)

    def test_optimal_snr_squared(self):
        """
        Merely checks parameters are given in the right order and the frequency
        mask is applied.
        """
        with mock.patch('bilby.gw.utils.noise_weighted_inner_product') as m:
            m.side_effect = lambda a, b, c, d: [a, b, c, d]
            signal = np.ones_like(self.ifo.power_spectral_density_array)
            mask = self.ifo.frequency_mask
            expected = [signal[mask], signal[mask],
                        self.ifo.power_spectral_density_array[mask],
                        self.ifo.strain_data.duration]
            actual = self.ifo.optimal_snr_squared(signal=signal)
            self.assertTrue(np.array_equal(expected[0], actual[0]))
            self.assertTrue(np.array_equal(expected[1], actual[1]))
            self.assertTrue(np.array_equal(expected[2], actual[2]))
            self.assertEqual(expected[3], actual[3])

    def test_repr(self):
        expected = 'Interferometer(name=\'{}\', power_spectral_density={}, minimum_frequency={}, ' \
                   'maximum_frequency={}, length={}, latitude={}, longitude={}, elevation={}, xarm_azimuth={}, ' \
                   'yarm_azimuth={}, xarm_tilt={}, yarm_tilt={})' \
            .format(self.name, self.power_spectral_density, float(self.minimum_frequency),
                    float(self.maximum_frequency), float(self.length), float(self.latitude), float(self.longitude),
                    float(self.elevation), float(self.xarm_azimuth), float(self.yarm_azimuth), float(self.xarm_tilt),
                    float(self.yarm_tilt))
        self.assertEqual(expected, repr(self.ifo))

    def test_to_and_from_hdf5_loading(self):
        if sys.version_info[0] < 3:
            with self.assertRaises(NotImplementedError):
                self.ifo.to_hdf5(outdir='outdir', label='test')
        else:
            self.ifo.to_hdf5(outdir='outdir', label='test')
            filename = self.ifo._hdf5_filename_from_outdir_label(outdir='outdir', label='test')
            recovered_ifo = bilby.gw.detector.Interferometer.from_hdf5(filename)
            self.assertEqual(self.ifo, recovered_ifo)

    def test_to_and_from_hdf5_wrong_class(self):
        if sys.version_info[0] < 3:
            pass
        else:
            bilby.core.utils.check_directory_exists_and_if_not_mkdir('outdir')
            dd.io.save('./outdir/psd.h5', self.power_spectral_density)
            filename = self.ifo._hdf5_filename_from_outdir_label(outdir='outdir', label='psd')
            with self.assertRaises(TypeError):
                bilby.gw.detector.Interferometer.from_hdf5(filename)


class TestInterferometerEquals(unittest.TestCase):

    def setUp(self):
        self.name = 'name'
        self.power_spectral_density_1 = bilby.gw.detector.PowerSpectralDensity.from_aligo()
        self.power_spectral_density_2 = bilby.gw.detector.PowerSpectralDensity.from_aligo()
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
        self.duration = 1
        self.sampling_frequency = 200
        self.frequency_array = bilby.utils.create_frequency_series(sampling_frequency=self.sampling_frequency,
                                                                   duration=self.duration)
        self.strain = self.frequency_array
        self.ifo_1 = bilby.gw.detector.Interferometer(name=self.name,
                                                      power_spectral_density=self.power_spectral_density_1,
                                                      minimum_frequency=self.minimum_frequency,
                                                      maximum_frequency=self.maximum_frequency, length=self.length,
                                                      latitude=self.latitude, longitude=self.longitude,
                                                      elevation=self.elevation,
                                                      xarm_azimuth=self.xarm_azimuth, yarm_azimuth=self.yarm_azimuth,
                                                      xarm_tilt=self.xarm_tilt, yarm_tilt=self.yarm_tilt)
        self.ifo_2 = bilby.gw.detector.Interferometer(name=self.name,
                                                      power_spectral_density=self.power_spectral_density_2,
                                                      minimum_frequency=self.minimum_frequency,
                                                      maximum_frequency=self.maximum_frequency, length=self.length,
                                                      latitude=self.latitude, longitude=self.longitude,
                                                      elevation=self.elevation,
                                                      xarm_azimuth=self.xarm_azimuth, yarm_azimuth=self.yarm_azimuth,
                                                      xarm_tilt=self.xarm_tilt, yarm_tilt=self.yarm_tilt)
        self.ifo_1.set_strain_data_from_frequency_domain_strain(frequency_array=self.frequency_array,
                                                                frequency_domain_strain=self.strain)
        self.ifo_2.set_strain_data_from_frequency_domain_strain(frequency_array=self.frequency_array,
                                                                frequency_domain_strain=self.strain)

    def tearDown(self):
        del self.name
        del self.power_spectral_density_1
        del self.power_spectral_density_2
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
        del self.ifo_1
        del self.ifo_2
        del self.sampling_frequency
        del self.duration
        del self.frequency_array
        del self.strain

    def test_eq_true(self):
        self.assertEqual(self.ifo_1, self.ifo_2)

    def test_eq_false_different_psd(self):
        self.ifo_1.power_spectral_density.psd_array[0] = 1234
        self.assertNotEqual(self.ifo_1, self.ifo_2)

    def test_eq_false_different_minimum_frequency(self):
        self.ifo_1.minimum_frequency -= 1
        self.assertNotEqual(self.ifo_1, self.ifo_2)

    def test_eq_false_different_maximum_frequency(self):
        self.ifo_1.minimum_frequency -= 1
        self.assertNotEqual(self.ifo_1, self.ifo_2)

    def test_eq_false_different_length(self):
        self.ifo_1.length -= 1
        self.assertNotEqual(self.ifo_1, self.ifo_2)

    def test_eq_false_different_latitude(self):
        self.ifo_1.latitude -= 1
        self.assertNotEqual(self.ifo_1, self.ifo_2)

    def test_eq_false_different_longitude(self):
        self.ifo_1.longitude -= 1
        self.assertNotEqual(self.ifo_1, self.ifo_2)

    def test_eq_false_different_elevation(self):
        self.ifo_1.elevation -= 1
        self.assertNotEqual(self.ifo_1, self.ifo_2)

    def test_eq_false_different_xarm_azimuth(self):
        self.ifo_1.xarm_azimuth -= 1
        self.assertNotEqual(self.ifo_1, self.ifo_2)

    def test_eq_false_different_xarmtilt(self):
        self.ifo_1.xarm_tilt -= 1
        self.assertNotEqual(self.ifo_1, self.ifo_2)

    def test_eq_false_different_yarm_azimuth(self):
        self.ifo_1.yarm_azimuth -= 1
        self.assertNotEqual(self.ifo_1, self.ifo_2)

    def test_eq_false_different_yarm_tilt(self):
        self.ifo_1.yarm_tilt -= 1
        self.assertNotEqual(self.ifo_1, self.ifo_2)

    def test_eq_false_different_ifo_strain_data(self):
        self.strain = bilby.utils.create_frequency_series(sampling_frequency=self.sampling_frequency/2,
                                                          duration=self.duration*2)
        self.ifo_1.set_strain_data_from_frequency_domain_strain(frequency_array=self.frequency_array,
                                                                frequency_domain_strain=self.strain)
        self.assertNotEqual(self.ifo_1, self.ifo_2)


class TestInterferometerStrainData(unittest.TestCase):

    def setUp(self):
        self.minimum_frequency = 10
        self.maximum_frequency = 20
        self.ifosd = bilby.gw.detector.InterferometerStrainData(
            minimum_frequency=self.minimum_frequency,
            maximum_frequency=self.maximum_frequency)

    def tearDown(self):
        del self.minimum_frequency
        del self.maximum_frequency
        del self.ifosd

    def test_frequency_mask(self):
        self.minimum_frequency = 10
        self.maximum_frequency = 20
        with mock.patch('bilby.core.utils.create_frequency_series') as m:
            m.return_value = np.array([5, 15, 25])
            self.ifosd.set_from_frequency_domain_strain(
                frequency_domain_strain=np.array([0, 1, 2]), frequency_array=np.array([5, 15, 25]))
            self.assertTrue(np.array_equal(self.ifosd.frequency_mask, [False, True, False]))

    def test_set_data_fails(self):
        with mock.patch('bilby.core.utils.create_frequency_series') as m:
            m.return_value = [1, 2, 3]
            with self.assertRaises(ValueError):
                self.ifosd.set_from_frequency_domain_strain(
                    frequency_domain_strain=np.array([0, 1, 2]))

    def test_set_data_fails_too_much(self):
        with mock.patch('bilby.core.utils.create_frequency_series') as m:
            m.return_value = [1, 2, 3]
            with self.assertRaises(ValueError):
                self.ifosd.set_from_frequency_domain_strain(
                    frequency_domain_strain=np.array([0, 1, 2]), frequency_array=np.array([1, 2, 3]),
                    duration=3, sampling_frequency=1)

    def test_start_time_init(self):
        with mock.patch('bilby.core.utils.create_frequency_series') as m:
            m.return_value = [1, 2, 3]
            duration = 3
            sampling_frequency = 1
            self.ifosd.set_from_frequency_domain_strain(
                frequency_domain_strain=np.array([0, 1, 2]), duration=duration,
                sampling_frequency=sampling_frequency)
            self.assertTrue(self.ifosd.start_time == 0)

    def test_start_time_set(self):
        with mock.patch('bilby.core.utils.create_frequency_series') as m:
            m.return_value = [1, 2, 3]
            duration = 3
            sampling_frequency = 1
            self.ifosd.set_from_frequency_domain_strain(
                frequency_domain_strain=np.array([0, 1, 2]), duration=duration,
                sampling_frequency=sampling_frequency, start_time=10)
            self.assertTrue(self.ifosd.start_time == 10)

    def test_time_array_frequency_array_consistency(self):
        duration = 1
        sampling_frequency = 10
        time_array = bilby.core.utils.create_time_series(
            sampling_frequency=sampling_frequency, duration=duration)
        time_domain_strain = np.random.normal(
            0, duration - 1 / sampling_frequency, len(time_array))
        self.ifosd.roll_off = 0
        self.ifosd.set_from_time_domain_strain(
            time_domain_strain=time_domain_strain, duration=duration,
            sampling_frequency=sampling_frequency)

        frequency_domain_strain, freqs = bilby.core.utils.nfft(
            time_domain_strain, sampling_frequency)

        self.assertTrue(np.all(
            self.ifosd.frequency_domain_strain == frequency_domain_strain * self.ifosd.frequency_mask))

    def test_time_within_data_before(self):
        self.ifosd.start_time = 3
        self.ifosd.duration = 2
        self.assertFalse(self.ifosd.time_within_data(2))

    def test_time_within_data_during(self):
        self.ifosd.start_time = 3
        self.ifosd.duration = 2
        self.assertTrue(self.ifosd.time_within_data(3))
        self.assertTrue(self.ifosd.time_within_data(4))
        self.assertTrue(self.ifosd.time_within_data(5))

    def test_time_within_data_after(self):
        self.ifosd.start_time = 3
        self.ifosd.duration = 2
        self.assertFalse(self.ifosd.time_within_data(6))

    def test_time_domain_window_no_roll_off_no_alpha(self):
        self.ifosd._time_domain_strain = np.array([3])
        self.ifosd.duration = 5
        self.ifosd.roll_off = 2
        expected_window = scipy.signal.windows.tukey(len(self.ifosd._time_domain_strain), alpha=self.ifosd.alpha)
        self.assertEqual(expected_window,
                         self.ifosd.time_domain_window())
        self.assertEqual(np.mean(expected_window ** 2), self.ifosd.window_factor)

    def test_time_domain_window_sets_roll_off_directly(self):
        self.ifosd._time_domain_strain = np.array([3])
        self.ifosd.duration = 5
        self.ifosd.roll_off = 2
        expected_roll_off = 6
        self.ifosd.time_domain_window(roll_off=expected_roll_off)
        self.assertEqual(expected_roll_off, self.ifosd.roll_off)

    def test_time_domain_window_sets_roll_off_indirectly(self):
        self.ifosd._time_domain_strain = np.array([3])
        self.ifosd.duration = 5
        self.ifosd.roll_off = 2
        alpha = 4
        expected_roll_off = alpha * self.ifosd.duration / 2
        self.ifosd.time_domain_window(alpha=alpha)
        self.assertEqual(expected_roll_off, self.ifosd.roll_off)

    def test_time_domain_strain_when_set(self):
        expected_strain = 5
        self.ifosd._time_domain_strain = expected_strain
        self.assertEqual(expected_strain, self.ifosd.time_domain_strain)

    @patch('bilby.core.utils.infft')
    def test_time_domain_strain_from_frequency_domain_strain(self, m):
        m.return_value = 5
        self.ifosd.sampling_frequency = 200
        self.ifosd.duration = 4
        self.ifosd.sampling_frequency = 123
        self.ifosd.frequency_domain_strain = self.ifosd.frequency_array
        self.assertEqual(m.return_value, self.ifosd.time_domain_strain)

    def test_time_domain_strain_not_set(self):
        self.ifosd._time_domain_strain = None
        self.ifosd._frequency_domain_strain = None
        with self.assertRaises(ValueError):
            test = self.ifosd.time_domain_strain

    def test_frequency_domain_strain_when_set(self):
        self.ifosd.sampling_frequency = 200
        self.ifosd.duration = 4
        expected_strain = self.ifosd.frequency_array * self.ifosd.frequency_mask
        self.ifosd._frequency_domain_strain = expected_strain
        self.assertTrue(np.array_equal(expected_strain,
                                       self.ifosd.frequency_domain_strain))

    @patch('bilby.core.utils.nfft')
    def test_frequency_domain_strain_from_frequency_domain_strain(self, m):
        self.ifosd.start_time = 0
        self.ifosd.duration = 4
        self.ifosd.sampling_frequency = 200
        m.return_value = self.ifosd.frequency_array, self.ifosd.frequency_array
        self.ifosd._time_domain_strain = self.ifosd.time_array
        self.assertTrue(np.array_equal(self.ifosd.frequency_array * self.ifosd.frequency_mask,
                                       self.ifosd.frequency_domain_strain))

    def test_frequency_domain_strain_not_set(self):
        self.ifosd._time_domain_strain = None
        self.ifosd._frequency_domain_strain = None
        with self.assertRaises(ValueError):
            test = self.ifosd.frequency_domain_strain

    def test_set_frequency_domain_strain(self):
        self.ifosd.duration = 4
        self.ifosd.sampling_frequency = 200
        self.ifosd.frequency_domain_strain = np.ones(len(self.ifosd.frequency_array))
        self.assertTrue(np.array_equal(np.ones(len(self.ifosd.frequency_array)),
                                       self.ifosd._frequency_domain_strain))

    def test_set_frequency_domain_strain_wrong_length(self):
        self.ifosd.duration = 4
        self.ifosd.sampling_frequency = 200
        with self.assertRaises(ValueError):
            self.ifosd.frequency_domain_strain = np.array([1])


class TestInterferometerStrainDataEquals(unittest.TestCase):

    def setUp(self):
        self.minimum_frequency = 10
        self.maximum_frequency = 20
        self.roll_off = 0.2
        self.sampling_frequency = 100
        self.duration = 2
        self.frequency_array = bilby.utils.create_frequency_series(sampling_frequency=self.sampling_frequency,
                                                                   duration=self.duration)
        self.strain = self.frequency_array
        self.ifosd_1 = bilby.gw.detector.InterferometerStrainData(minimum_frequency=self.minimum_frequency,
                                                                  maximum_frequency=self.maximum_frequency,
                                                                  roll_off=self.roll_off)
        self.ifosd_2 = bilby.gw.detector.InterferometerStrainData(minimum_frequency=self.minimum_frequency,
                                                                  maximum_frequency=self.maximum_frequency,
                                                                  roll_off=self.roll_off)
        self.ifosd_1.set_from_frequency_domain_strain(frequency_domain_strain=self.strain,
                                                      frequency_array=self.frequency_array)
        self.ifosd_2.set_from_frequency_domain_strain(frequency_domain_strain=self.strain,
                                                      frequency_array=self.frequency_array)

    def tearDown(self):
        del self.minimum_frequency
        del self.maximum_frequency
        del self.roll_off
        del self.sampling_frequency
        del self.duration
        del self.frequency_array
        del self.strain
        del self.ifosd_1
        del self.ifosd_2

    def test_eq_true(self):
        self.assertEqual(self.ifosd_1, self.ifosd_2)

    def test_eq_different_minimum_frequency(self):
        self.ifosd_1.minimum_frequency -= 1
        self.assertNotEqual(self.ifosd_1, self.ifosd_2)

    def test_eq_different_maximum_frequency(self):
        self.ifosd_1.maximum_frequency -= 1
        self.assertNotEqual(self.ifosd_1, self.ifosd_2)

    def test_eq_different_roll_off(self):
        self.ifosd_1.roll_off -= 0.1
        self.assertNotEqual(self.ifosd_1, self.ifosd_2)

    def test_eq_different_window_factor(self):
        self.ifosd_1.roll_off -= 0.1
        self.assertNotEqual(self.ifosd_1, self.ifosd_2)

    def test_eq_different_sampling_frequency(self):
        self.ifosd_1.sampling_frequency *= 2
        self.assertNotEqual(self.ifosd_1, self.ifosd_2)

    def test_eq_different_sampling_duration(self):
        self.ifosd_1.duration *= 2
        self.assertNotEqual(self.ifosd_1, self.ifosd_2)

    def test_eq_different_start_time(self):
        self.ifosd_1.start_time -= 0.1
        self.assertNotEqual(self.ifosd_1, self.ifosd_2)

    def test_eq_different_frequency_array(self):
        new_frequency_array = bilby.utils.create_frequency_series(sampling_frequency=self.sampling_frequency/2,
                                                                  duration=self.duration*2)
        self.ifosd_1.frequency_array = new_frequency_array
        self.assertNotEqual(self.ifosd_1, self.ifosd_2)

    def test_eq_different_frequency_domain_strain(self):
        new_strain = bilby.utils.create_frequency_series(sampling_frequency=self.sampling_frequency/2,
                                                         duration=self.duration*2)
        self.ifosd_1._frequency_domain_strain = new_strain
        self.assertNotEqual(self.ifosd_1, self.ifosd_2)

    def test_eq_different_time_array(self):
        new_time_array = bilby.utils.create_time_series(sampling_frequency=self.sampling_frequency/2,
                                                        duration=self.duration*2)
        self.ifosd_1.time_array = new_time_array
        self.assertNotEqual(self.ifosd_1, self.ifosd_2)

    def test_eq_different_time_domain_strain(self):
        new_strain = bilby.utils.create_time_series(sampling_frequency=self.sampling_frequency/2,
                                                    duration=self.duration*2)
        self.ifosd_1._time_domain_strain= new_strain
        self.assertNotEqual(self.ifosd_1, self.ifosd_2)


class TestInterferometerList(unittest.TestCase):

    def setUp(self):
        self.frequency_arrays = np.linspace(0, 4096, 4097)
        self.name1 = 'name1'
        self.name2 = 'name2'
        self.power_spectral_density1 = bilby.gw.detector.PowerSpectralDensity.from_aligo()
        self.power_spectral_density2 = bilby.gw.detector.PowerSpectralDensity.from_aligo()
        self.minimum_frequency1 = 10
        self.minimum_frequency2 = 10
        self.maximum_frequency1 = 20
        self.maximum_frequency2 = 20
        self.length1 = 30
        self.length2 = 30
        self.latitude1 = 1
        self.latitude2 = 1
        self.longitude1 = 2
        self.longitude2 = 2
        self.elevation1 = 3
        self.elevation2 = 3
        self.xarm_azimuth1 = 4
        self.xarm_azimuth2 = 4
        self.yarm_azimuth1 = 5
        self.yarm_azimuth2 = 5
        self.xarm_tilt1 = 0.
        self.xarm_tilt2 = 0.
        self.yarm_tilt1 = 0.
        self.yarm_tilt2 = 0.
        # noinspection PyTypeChecker
        self.ifo1 = bilby.gw.detector.Interferometer(name=self.name1,
                                                     power_spectral_density=self.power_spectral_density1,
                                                     minimum_frequency=self.minimum_frequency1,
                                                     maximum_frequency=self.maximum_frequency1, length=self.length1,
                                                     latitude=self.latitude1, longitude=self.longitude1,
                                                     elevation=self.elevation1,
                                                     xarm_azimuth=self.xarm_azimuth1, yarm_azimuth=self.yarm_azimuth1,
                                                     xarm_tilt=self.xarm_tilt1, yarm_tilt=self.yarm_tilt1)
        self.ifo2 = bilby.gw.detector.Interferometer(name=self.name2,
                                                     power_spectral_density=self.power_spectral_density2,
                                                     minimum_frequency=self.minimum_frequency2,
                                                     maximum_frequency=self.maximum_frequency2, length=self.length2,
                                                     latitude=self.latitude2, longitude=self.longitude2,
                                                     elevation=self.elevation2,
                                                     xarm_azimuth=self.xarm_azimuth2, yarm_azimuth=self.yarm_azimuth2,
                                                     xarm_tilt=self.xarm_tilt2, yarm_tilt=self.yarm_tilt2)
        self.ifo1.strain_data.set_from_frequency_domain_strain(
            self.frequency_arrays, sampling_frequency=4096, duration=2)
        self.ifo2.strain_data.set_from_frequency_domain_strain(
            self.frequency_arrays, sampling_frequency=4096, duration=2)
        self.ifo_list = bilby.gw.detector.InterferometerList([self.ifo1, self.ifo2])
        self.outdir = 'outdir'
        bilby.core.utils.check_directory_exists_and_if_not_mkdir(self.outdir)

    def tearDown(self):
        del self.frequency_arrays
        del self.name1
        del self.name2
        del self.power_spectral_density1
        del self.power_spectral_density2
        del self.minimum_frequency1
        del self.minimum_frequency2
        del self.maximum_frequency1
        del self.maximum_frequency2
        del self.length1
        del self.length2
        del self.latitude1
        del self.latitude2
        del self.longitude1
        del self.longitude2
        del self.elevation1
        del self.elevation2
        del self.xarm_azimuth1
        del self.xarm_azimuth2
        del self.yarm_azimuth1
        del self.yarm_azimuth2
        del self.xarm_tilt1
        del self.xarm_tilt2
        del self.yarm_tilt1
        del self.yarm_tilt2
        del self.ifo1
        del self.ifo2
        del self.ifo_list
        rmtree(self.outdir)

    def test_init_with_string(self):
        with self.assertRaises(TypeError):
            bilby.gw.detector.InterferometerList("string")

    def test_init_with_string_list(self):
        """ Merely checks if this ends up in the right bracket """
        with mock.patch('bilby.gw.detector.get_empty_interferometer') as m:
            m.side_effect = TypeError
            with self.assertRaises(TypeError):
                bilby.gw.detector.InterferometerList(['string'])

    def test_init_with_other_object(self):
        with self.assertRaises(TypeError):
            bilby.gw.detector.InterferometerList([object()])

    def test_init_with_actual_ifos(self):
        ifo_list = bilby.gw.detector.InterferometerList([self.ifo1, self.ifo2])
        self.assertEqual(self.ifo1, ifo_list[0])
        self.assertEqual(self.ifo2, ifo_list[1])

    def test_init_inconsistent_duration(self):
        self.frequency_arrays = np.linspace(0, 2048, 2049)
        self.ifo2 = bilby.gw.detector.Interferometer(name=self.name2,
                                                     power_spectral_density=self.power_spectral_density2,
                                                     minimum_frequency=self.minimum_frequency2,
                                                     maximum_frequency=self.maximum_frequency2, length=self.length2,
                                                     latitude=self.latitude2, longitude=self.longitude2,
                                                     elevation=self.elevation2,
                                                     xarm_azimuth=self.xarm_azimuth2, yarm_azimuth=self.yarm_azimuth2,
                                                     xarm_tilt=self.xarm_tilt2, yarm_tilt=self.yarm_tilt2)
        self.ifo2.strain_data.set_from_frequency_domain_strain(
            self.frequency_arrays, sampling_frequency=4096, duration=1)
        with self.assertRaises(ValueError):
            bilby.gw.detector.InterferometerList([self.ifo1, self.ifo2])

    def test_init_inconsistent_sampling_frequency(self):
        self.frequency_arrays = np.linspace(0, 2048, 2049)
        self.ifo2 = bilby.gw.detector.Interferometer(name=self.name2,
                                                     power_spectral_density=self.power_spectral_density2,
                                                     minimum_frequency=self.minimum_frequency2,
                                                     maximum_frequency=self.maximum_frequency2, length=self.length2,
                                                     latitude=self.latitude2, longitude=self.longitude2,
                                                     elevation=self.elevation2,
                                                     xarm_azimuth=self.xarm_azimuth2, yarm_azimuth=self.yarm_azimuth2,
                                                     xarm_tilt=self.xarm_tilt2, yarm_tilt=self.yarm_tilt2)
        self.ifo2.strain_data.set_from_frequency_domain_strain(
            self.frequency_arrays, sampling_frequency=2048, duration=2)
        with self.assertRaises(ValueError):
            bilby.gw.detector.InterferometerList([self.ifo1, self.ifo2])

    def test_init_inconsistent_start_time(self):
        self.ifo2.strain_data.start_time = 1
        with self.assertRaises(ValueError):
            bilby.gw.detector.InterferometerList([self.ifo1, self.ifo2])

    @patch.object(bilby.gw.detector.Interferometer, 'set_strain_data_from_power_spectral_density')
    def test_set_strain_data_from_power_spectral_density(self, m):
        self.ifo_list.set_strain_data_from_power_spectral_densities(sampling_frequency=123, duration=6.2, start_time=3)
        m.assert_called_with(sampling_frequency=123, duration=6.2, start_time=3)
        self.assertEqual(len(self.ifo_list), m.call_count)

    def test_inject_signal_pol_and_wg_none(self):
        with self.assertRaises(ValueError):
            self.ifo_list.inject_signal(injection_polarizations=None, waveform_generator=None)

    def test_meta_data(self):
        ifos_list = [self.ifo1, self.ifo2]
        ifos = bilby.gw.detector.InterferometerList(ifos_list)
        self.assertTrue(isinstance(ifos.meta_data, dict))
        meta_data = {ifo.name: ifo.meta_data for ifo in ifos_list}
        self.assertEqual(ifos.meta_data, meta_data)

    @patch.object(bilby.gw.waveform_generator.WaveformGenerator, 'frequency_domain_strain')
    def test_inject_signal_pol_none_calls_frequency_domain_strain(self, m):
        waveform_generator = bilby.gw.waveform_generator.WaveformGenerator(
            frequency_domain_source_model=lambda x, y, z: x)
        self.ifo1.inject_signal = MagicMock(return_value=None)
        self.ifo2.inject_signal = MagicMock(return_value=None)
        self.ifo_list.inject_signal(parameters=None, waveform_generator=waveform_generator)
        self.assertTrue(m.called)

    @patch.object(bilby.gw.detector.Interferometer, 'inject_signal')
    def test_inject_signal_with_inj_pol(self, m):
        self.ifo_list.inject_signal(injection_polarizations=dict(plus=1))
        m.assert_called_with(parameters=None, injection_polarizations=dict(plus=1))
        self.assertEqual(len(self.ifo_list), m.call_count)

    @patch.object(bilby.gw.detector.Interferometer, 'inject_signal')
    def test_inject_signal_returns_expected_polarisations(self, m):
        m.return_value = dict(plus=1, cross=2)
        injection_polarizations = dict(plus=1, cross=2)
        ifos_pol = self.ifo_list.inject_signal(injection_polarizations=injection_polarizations)
        self.assertDictEqual(self.ifo1.inject_signal(injection_polarizations=injection_polarizations), ifos_pol[0])
        self.assertDictEqual(self.ifo2.inject_signal(injection_polarizations=injection_polarizations), ifos_pol[1])

    @patch.object(bilby.gw.detector.Interferometer, 'save_data')
    def test_save_data(self, m):
        self.ifo_list.save_data(outdir='test_outdir', label='test_outdir')
        m.assert_called_with(outdir='test_outdir', label='test_outdir')
        self.assertEqual(len(self.ifo_list), m.call_count)

    def test_number_of_interferometers(self):
        self.assertEqual(len(self.ifo_list), self.ifo_list.number_of_interferometers)

    def test_duration(self):
        self.assertEqual(self.ifo1.strain_data.duration, self.ifo_list.duration)
        self.assertEqual(self.ifo2.strain_data.duration, self.ifo_list.duration)

    def test_sampling_frequency(self):
        self.assertEqual(self.ifo1.strain_data.sampling_frequency, self.ifo_list.sampling_frequency)
        self.assertEqual(self.ifo2.strain_data.sampling_frequency, self.ifo_list.sampling_frequency)

    def test_start_time(self):
        self.assertEqual(self.ifo1.strain_data.start_time, self.ifo_list.start_time)
        self.assertEqual(self.ifo2.strain_data.start_time, self.ifo_list.start_time)

    def test_frequency_array(self):
        self.assertTrue(np.array_equal(self.ifo1.strain_data.frequency_array, self.ifo_list.frequency_array))
        self.assertTrue(np.array_equal(self.ifo2.strain_data.frequency_array, self.ifo_list.frequency_array))

    def test_append_with_ifo(self):
        self.ifo_list.append(self.ifo2)
        names = [ifo.name for ifo in self.ifo_list]
        self.assertListEqual([self.ifo1.name, self.ifo2.name, self.ifo2.name], names)

    def test_append_with_ifo_list(self):
        self.ifo_list.append(self.ifo_list)
        names = [ifo.name for ifo in self.ifo_list]
        self.assertListEqual([self.ifo1.name, self.ifo2.name, self.ifo1.name, self.ifo2.name], names)

    def test_extend(self):
        self.ifo_list.extend(self.ifo_list)
        names = [ifo.name for ifo in self.ifo_list]
        self.assertListEqual([self.ifo1.name, self.ifo2.name, self.ifo1.name, self.ifo2.name], names)

    def test_insert(self):
        new_ifo = self.ifo1
        new_ifo.name = 'name3'
        self.ifo_list.insert(1, new_ifo)
        names = [ifo.name for ifo in self.ifo_list]
        self.assertListEqual([self.ifo1.name, new_ifo.name, self.ifo2.name], names)

    def test_to_and_from_hdf5_loading(self):
        if sys.version_info[0] < 3:
            with self.assertRaises(NotImplementedError):
                self.ifo_list.to_hdf5(outdir='outdir', label='test')
        else:
            self.ifo_list.to_hdf5(outdir='outdir', label='test')
            filename = 'outdir/test_name1name2.h5'
            recovered_ifo = bilby.gw.detector.InterferometerList.from_hdf5(filename)
            self.assertListEqual(self.ifo_list, recovered_ifo)

    def test_to_and_from_hdf5_wrong_class(self):
        if sys.version_info[0] < 3:
            pass
        else:
            dd.io.save('./outdir/psd.h5', self.ifo_list[0].power_spectral_density)
            filename = self.ifo_list._hdf5_filename_from_outdir_label(
                outdir='outdir', label='psd')
            with self.assertRaises(TypeError):
                bilby.gw.detector.InterferometerList.from_hdf5(filename)

    def test_plot_data(self):
        ifos = bilby.gw.detector.InterferometerList(['H1', 'L1'])
        ifos.set_strain_data_from_power_spectral_densities(2048, 4)
        ifos.plot_data(outdir=self.outdir)

        ifos = bilby.gw.detector.InterferometerList(['H1', 'L1', 'V1'])
        ifos.set_strain_data_from_power_spectral_densities(2048, 4)
        ifos.plot_data(outdir=self.outdir)


class TestPowerSpectralDensityWithoutFiles(unittest.TestCase):

    def setUp(self):
        self.frequency_array = np.array([1., 2., 3.])
        self.psd_array = np.array([16., 25., 36.])
        self.asd_array = np.array([4., 5., 6.])

    def tearDown(self):
        del self.frequency_array
        del self.psd_array
        del self.asd_array

    def test_init_with_asd_array(self):
        psd = bilby.gw.detector.PowerSpectralDensity(frequency_array=self.frequency_array, asd_array=self.asd_array)
        self.assertTrue(np.array_equal(self.frequency_array, psd.frequency_array))
        self.assertTrue(np.array_equal(self.asd_array, psd.asd_array))
        self.assertTrue(np.array_equal(self.psd_array, psd.psd_array))

    def test_init_with_psd_array(self):
        psd = bilby.gw.detector.PowerSpectralDensity(frequency_array=self.frequency_array, psd_array=self.psd_array)
        self.assertTrue(np.array_equal(self.frequency_array, psd.frequency_array))
        self.assertTrue(np.array_equal(self.asd_array, psd.asd_array))
        self.assertTrue(np.array_equal(self.psd_array, psd.psd_array))

    def test_setting_asd_array_after_init(self):
        psd = bilby.gw.detector.PowerSpectralDensity(frequency_array=self.frequency_array)
        psd.asd_array = self.asd_array
        self.assertTrue(np.array_equal(self.frequency_array, psd.frequency_array))
        self.assertTrue(np.array_equal(self.asd_array, psd.asd_array))
        self.assertTrue(np.array_equal(self.psd_array, psd.psd_array))

    def test_setting_psd_array_after_init(self):
        psd = bilby.gw.detector.PowerSpectralDensity(frequency_array=self.frequency_array)
        psd.psd_array = self.psd_array
        self.assertTrue(np.array_equal(self.frequency_array, psd.frequency_array))
        self.assertTrue(np.array_equal(self.asd_array, psd.asd_array))
        self.assertTrue(np.array_equal(self.psd_array, psd.psd_array))

    def test_power_spectral_density_interpolated_from_asd_array(self):
        expected = np.array([25.])
        psd = bilby.gw.detector.PowerSpectralDensity(frequency_array=self.frequency_array, asd_array=self.asd_array)
        self.assertEqual(expected, psd.power_spectral_density_interpolated(2))

    def test_power_spectral_density_interpolated_from_psd_array(self):
        expected = np.array([25.])
        psd = bilby.gw.detector.PowerSpectralDensity(frequency_array=self.frequency_array, psd_array=self.psd_array)
        self.assertEqual(expected, psd.power_spectral_density_interpolated(2))

    def test_from_amplitude_spectral_density_array(self):
        actual = bilby.gw.detector.PowerSpectralDensity.from_amplitude_spectral_density_array(
            frequency_array=self.frequency_array, asd_array=self.asd_array)
        self.assertTrue(np.array_equal(self.psd_array, actual.psd_array))
        self.assertTrue(np.array_equal(self.asd_array, actual.asd_array))

    def test_from_power_spectral_density_array(self):
        actual = bilby.gw.detector.PowerSpectralDensity.from_power_spectral_density_array(
            frequency_array=self.frequency_array, psd_array=self.psd_array)
        self.assertTrue(np.array_equal(self.psd_array, actual.psd_array))
        self.assertTrue(np.array_equal(self.asd_array, actual.asd_array))

    def test_repr(self):
        psd = bilby.gw.detector.PowerSpectralDensity(frequency_array=self.frequency_array, psd_array=self.psd_array)
        expected = 'PowerSpectralDensity(frequency_array={}, psd_array={}, asd_array={})'.format(self.frequency_array,
                                                                                                 self.psd_array,
                                                                                                 self.asd_array)
        self.assertEqual(expected, repr(psd))


class TestPowerSpectralDensityWithFiles(unittest.TestCase):

    def setUp(self):
        self.dir = os.path.join(os.path.dirname(__file__), 'noise_curves')
        os.mkdir(self.dir)
        self.asd_file = os.path.join(os.path.dirname(__file__), 'noise_curves', 'asd_test_file.txt')
        self.psd_file = os.path.join(os.path.dirname(__file__), 'noise_curves', 'psd_test_file.txt')
        with open(self.asd_file, 'w') as f:
            f.write('1.\t1.0e-21\n2.\t2.0e-21\n3.\t3.0e-21')
        with open(self.psd_file, 'w') as f:
            f.write('1.\t1.0e-42\n2.\t4.0e-42\n3.\t9.0e-42')
        self.frequency_array = np.array([1.0, 2.0, 3.0])
        self.asd_array = np.array([1.0e-21, 2.0e-21, 3.0e-21])
        self.psd_array = np.array([1.0e-42, 4.0e-42, 9.0e-42])

    def tearDown(self):
        os.remove(self.asd_file)
        os.remove(self.psd_file)
        os.rmdir(self.dir)
        del self.dir
        del self.asd_array
        del self.psd_array
        del self.asd_file
        del self.psd_file

    def test_init_with_psd_file(self):
        psd = bilby.gw.detector.PowerSpectralDensity(frequency_array=self.frequency_array, psd_file=self.psd_file)
        self.assertEqual(self.psd_file, psd.psd_file)
        self.assertTrue(np.array_equal(self.psd_array, psd.psd_array))
        self.assertTrue(np.allclose(self.asd_array, psd.asd_array, atol=1e-30))

    def test_init_with_asd_file(self):
        psd = bilby.gw.detector.PowerSpectralDensity(frequency_array=self.frequency_array, asd_file=self.asd_file)
        self.assertEqual(self.asd_file, psd.asd_file)
        self.assertTrue(np.allclose(self.psd_array, psd.psd_array, atol=1e-60))
        self.assertTrue(np.array_equal(self.asd_array, psd.asd_array))

    def test_setting_psd_array_after_init(self):
        psd = bilby.gw.detector.PowerSpectralDensity(frequency_array=self.frequency_array)
        psd.psd_file = self.psd_file
        self.assertEqual(self.psd_file, psd.psd_file)
        self.assertTrue(np.array_equal(self.psd_array, psd.psd_array))
        self.assertTrue(np.allclose(self.asd_array, psd.asd_array, atol=1e-30))

    def test_init_with_asd_array_after_init(self):
        psd = bilby.gw.detector.PowerSpectralDensity(frequency_array=self.frequency_array)
        psd.asd_file = self.asd_file
        self.assertEqual(self.asd_file, psd.asd_file)
        self.assertTrue(np.allclose(self.psd_array, psd.psd_array, atol=1e-60))
        self.assertTrue(np.array_equal(self.asd_array, psd.asd_array))

    def test_power_spectral_density_interpolated_from_asd_file(self):
        expected = np.array([4.0e-42])
        psd = bilby.gw.detector.PowerSpectralDensity(frequency_array=self.frequency_array, asd_file=self.asd_file)
        self.assertTrue(np.allclose(expected, psd.power_spectral_density_interpolated(2), atol=1e-60))

    def test_power_spectral_density_interpolated_from_psd_file(self):
        expected = np.array([4.0e-42])
        psd = bilby.gw.detector.PowerSpectralDensity(frequency_array=self.frequency_array, psd_file=self.psd_file)
        self.assertAlmostEqual(expected, psd.power_spectral_density_interpolated(2))

    def test_from_amplitude_spectral_density_file(self):
        psd = bilby.gw.detector.PowerSpectralDensity.from_amplitude_spectral_density_file(asd_file=self.asd_file)
        self.assertEqual(self.asd_file, psd.asd_file)
        self.assertTrue(np.allclose(self.psd_array, psd.psd_array, atol=1e-60))
        self.assertTrue(np.array_equal(self.asd_array, psd.asd_array))

    def test_from_power_spectral_density_file(self):
        psd = bilby.gw.detector.PowerSpectralDensity.from_power_spectral_density_file(psd_file=self.psd_file)
        self.assertEqual(self.psd_file, psd.psd_file)
        self.assertTrue(np.array_equal(self.psd_array, psd.psd_array))
        self.assertTrue(np.allclose(self.asd_array, psd.asd_array, atol=1e-30))

    def test_from_aligo(self):
        psd = bilby.gw.detector.PowerSpectralDensity.from_aligo()
        expected_filename = 'aLIGO_ZERO_DET_high_P_psd.txt'
        expected = bilby.gw.detector.PowerSpectralDensity(psd_file=expected_filename)
        actual_filename = psd.psd_file.split('/')[-1]
        self.assertEqual(expected_filename, actual_filename)
        self.assertTrue(np.allclose(expected.psd_array, psd.psd_array, atol=1e-60))
        self.assertTrue(np.array_equal(expected.asd_array, psd.asd_array))

    def test_check_file_psd_file_set_to_asd_file(self):
        logger = logging.getLogger('bilby')
        m = MagicMock()
        logger.warning = m
        psd = bilby.gw.detector.PowerSpectralDensity(psd_file=self.asd_file)
        self.assertEqual(4, m.call_count)

    def test_check_file_not_called_psd_file_set_to_psd_file(self):
        logger = logging.getLogger('bilby')
        m = MagicMock()
        logger.warning = m
        psd = bilby.gw.detector.PowerSpectralDensity(psd_file=self.psd_file)
        self.assertEqual(0, m.call_count)

    def test_check_file_asd_file_set_to_psd_file(self):
        logger = logging.getLogger('bilby')
        m = MagicMock()
        logger.warning = m
        psd = bilby.gw.detector.PowerSpectralDensity(asd_file=self.psd_file)
        self.assertEqual(4, m.call_count)

    def test_check_file_not_called_asd_file_set_to_asd_file(self):
        logger = logging.getLogger('bilby')
        m = MagicMock()
        logger.warning = m
        psd = bilby.gw.detector.PowerSpectralDensity(asd_file=self.asd_file)
        self.assertEqual(0, m.call_count)

    def test_from_frame_file(self):
        expected_frequency_array = np.array([1., 2., 3.])
        expected_psd_array = np.array([16., 25., 36.])
        with mock.patch('bilby.gw.detector.InterferometerStrainData.set_from_frame_file') as m:
            with mock.patch('bilby.gw.detector.InterferometerStrainData.create_power_spectral_density') as n:
                n.return_value = expected_frequency_array, expected_psd_array
                psd = bilby.gw.detector.PowerSpectralDensity.from_frame_file(frame_file=self.asd_file,
                                                                             psd_start_time=0,
                                                                             psd_duration=4)
                self.assertTrue(np.array_equal(expected_frequency_array, psd.frequency_array))
                self.assertTrue(np.array_equal(expected_psd_array, psd.psd_array))

    def test_repr(self):
        psd = bilby.gw.detector.PowerSpectralDensity(psd_file=self.psd_file)
        expected = 'PowerSpectralDensity(psd_file=\'{}\', asd_file=\'{}\')'.format(self.psd_file, None)
        self.assertEqual(expected, repr(psd))


class TestPowerSpectralDensityEquals(unittest.TestCase):

    def setUp(self):
        self.psd_from_file_1 = bilby.gw.detector.PowerSpectralDensity.from_aligo()
        self.psd_from_file_2 = bilby.gw.detector.PowerSpectralDensity.from_aligo()
        self.frequency_array = np.linspace(1, 100)
        self.psd_array = np.linspace(1, 100)
        self.psd_from_array_1 = bilby.gw.detector.PowerSpectralDensity. \
            from_power_spectral_density_array(frequency_array=self.frequency_array, psd_array= self.psd_array)
        self.psd_from_array_2 = bilby.gw.detector.PowerSpectralDensity. \
            from_power_spectral_density_array(frequency_array=self.frequency_array, psd_array= self.psd_array)

    def tearDown(self):
        del self.psd_from_file_1
        del self.psd_from_file_2
        del self.frequency_array
        del self.psd_array
        del self.psd_from_array_1
        del self.psd_from_array_2

    def test_eq_true_from_array(self):
        self.assertEqual(self.psd_from_array_1, self.psd_from_array_2)

    def test_eq_true_from_file(self):
        self.assertEqual(self.psd_from_file_1, self.psd_from_file_2)

    def test_eq_false_different_psd_file_name(self):
        self.psd_from_file_1._psd_file = 'some_other_name'
        self.assertNotEqual(self.psd_from_file_1, self.psd_from_file_2)

    def test_eq_false_different_asd_file_name(self):
        self.psd_from_file_1._psd_file = None
        self.psd_from_file_2._psd_file = None
        self.psd_from_file_1._asd_file = 'some_name'
        self.psd_from_file_2._asd_file = 'some_other_name'
        self.assertNotEqual(self.psd_from_file_1, self.psd_from_file_2)

    def test_eq_false_different_frequency_array(self):
        self.psd_from_file_1.frequency_array[0] = 0.5
        self.psd_from_array_1.frequency_array[0] = 0.5
        self.assertNotEqual(self.psd_from_file_1, self.psd_from_file_2)
        self.assertNotEqual(self.psd_from_array_1, self.psd_from_array_2)

    def test_eq_false_different_psd(self):
        self.psd_from_file_1.psd_array[0] = 0.53544321
        self.psd_from_array_1.psd_array[0] = 0.53544321
        self.assertNotEqual(self.psd_from_file_1, self.psd_from_file_2)
        self.assertNotEqual(self.psd_from_array_1, self.psd_from_array_2)

    def test_eq_false_different_asd(self):
        self.psd_from_file_1.asd_array[0] = 0.53544321
        self.psd_from_array_1.asd_array[0] = 0.53544321
        self.assertNotEqual(self.psd_from_file_1, self.psd_from_file_2)
        self.assertNotEqual(self.psd_from_array_1, self.psd_from_array_2)


if __name__ == '__main__':
    unittest.main()
