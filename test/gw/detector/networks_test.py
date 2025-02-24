import unittest
from unittest import mock
from shutil import rmtree
from itertools import combinations

import numpy as np

import bilby


class TestInterferometerList(unittest.TestCase):
    def setUp(self):
        self.frequency_arrays = np.linspace(0, 4096, 4097)
        self.name1 = "name1"
        self.name2 = "name2"
        self.power_spectral_density1 = (
            bilby.gw.detector.PowerSpectralDensity.from_aligo()
        )
        self.power_spectral_density2 = (
            bilby.gw.detector.PowerSpectralDensity.from_aligo()
        )
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
        self.xarm_tilt1 = 0.0
        self.xarm_tilt2 = 0.0
        self.yarm_tilt1 = 0.0
        self.yarm_tilt2 = 0.0
        # noinspection PyTypeChecker
        self.ifo1 = bilby.gw.detector.Interferometer(
            name=self.name1,
            power_spectral_density=self.power_spectral_density1,
            minimum_frequency=self.minimum_frequency1,
            maximum_frequency=self.maximum_frequency1,
            length=self.length1,
            latitude=self.latitude1,
            longitude=self.longitude1,
            elevation=self.elevation1,
            xarm_azimuth=self.xarm_azimuth1,
            yarm_azimuth=self.yarm_azimuth1,
            xarm_tilt=self.xarm_tilt1,
            yarm_tilt=self.yarm_tilt1,
        )
        self.ifo2 = bilby.gw.detector.Interferometer(
            name=self.name2,
            power_spectral_density=self.power_spectral_density2,
            minimum_frequency=self.minimum_frequency2,
            maximum_frequency=self.maximum_frequency2,
            length=self.length2,
            latitude=self.latitude2,
            longitude=self.longitude2,
            elevation=self.elevation2,
            xarm_azimuth=self.xarm_azimuth2,
            yarm_azimuth=self.yarm_azimuth2,
            xarm_tilt=self.xarm_tilt2,
            yarm_tilt=self.yarm_tilt2,
        )
        self.ifo1.strain_data.set_from_frequency_domain_strain(
            self.frequency_arrays, sampling_frequency=4096, duration=2
        )
        self.ifo2.strain_data.set_from_frequency_domain_strain(
            self.frequency_arrays, sampling_frequency=4096, duration=2
        )
        self.ifo_list = bilby.gw.detector.InterferometerList([self.ifo1, self.ifo2])
        self.outdir = "outdir"
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
        """Merely checks if this ends up in the right bracket"""
        with mock.patch("bilby.gw.detector.networks.get_empty_interferometer") as m:
            m.side_effect = TypeError
            with self.assertRaises(TypeError):
                bilby.gw.detector.InterferometerList(["string"])

    def test_init_with_other_object(self):
        with self.assertRaises(TypeError):
            bilby.gw.detector.InterferometerList([object()])

    def test_init_with_actual_ifos(self):
        ifo_list = bilby.gw.detector.InterferometerList([self.ifo1, self.ifo2])
        self.assertEqual(self.ifo1, ifo_list[0])
        self.assertEqual(self.ifo2, ifo_list[1])

    def test_init_inconsistent_duration(self):
        self.frequency_arrays = np.linspace(0, 2048, 2049)
        self.ifo2 = bilby.gw.detector.Interferometer(
            name=self.name2,
            power_spectral_density=self.power_spectral_density2,
            minimum_frequency=self.minimum_frequency2,
            maximum_frequency=self.maximum_frequency2,
            length=self.length2,
            latitude=self.latitude2,
            longitude=self.longitude2,
            elevation=self.elevation2,
            xarm_azimuth=self.xarm_azimuth2,
            yarm_azimuth=self.yarm_azimuth2,
            xarm_tilt=self.xarm_tilt2,
            yarm_tilt=self.yarm_tilt2,
        )
        self.ifo2.strain_data.set_from_frequency_domain_strain(
            self.frequency_arrays, sampling_frequency=4096, duration=1
        )
        with self.assertRaises(ValueError):
            bilby.gw.detector.InterferometerList([self.ifo1, self.ifo2])

    def test_init_inconsistent_sampling_frequency(self):
        self.frequency_arrays = np.linspace(0, 2048, 2049)
        self.ifo2 = bilby.gw.detector.Interferometer(
            name=self.name2,
            power_spectral_density=self.power_spectral_density2,
            minimum_frequency=self.minimum_frequency2,
            maximum_frequency=self.maximum_frequency2,
            length=self.length2,
            latitude=self.latitude2,
            longitude=self.longitude2,
            elevation=self.elevation2,
            xarm_azimuth=self.xarm_azimuth2,
            yarm_azimuth=self.yarm_azimuth2,
            xarm_tilt=self.xarm_tilt2,
            yarm_tilt=self.yarm_tilt2,
        )
        self.ifo2.strain_data.set_from_frequency_domain_strain(
            self.frequency_arrays, sampling_frequency=2048, duration=2
        )
        with self.assertRaises(ValueError):
            bilby.gw.detector.InterferometerList([self.ifo1, self.ifo2])

    def test_init_inconsistent_start_time(self):
        self.ifo2.strain_data.start_time = 1
        with self.assertRaises(ValueError):
            bilby.gw.detector.InterferometerList([self.ifo1, self.ifo2])

    @mock.patch.object(bilby.gw.detector.networks.logger, "warning")
    def test_check_interferometers_relative_tolerance(self, mock_warning):
        # Value larger than relative tolerance -- not tolerated
        self.ifo2.strain_data.start_time = self.ifo1.strain_data.start_time + 1e-4
        with self.assertRaises(ValueError):
            bilby.gw.detector.InterferometerList([self.ifo1, self.ifo2])

        # Value smaller than relative tolerance -- tolerated with warning
        self.ifo2.strain_data.start_time = self.ifo1.strain_data.start_time + 1e-6
        ifo_list = bilby.gw.detector.InterferometerList([self.ifo1, self.ifo2])
        self.assertIsNotNone(ifo_list)
        self.assertTrue(mock_warning.called)
        warning_log_str = mock_warning.call_args.args[0].args[0]
        self.assertIsInstance(warning_log_str, str)
        self.assertTrue(
            "The start_time of all interferometers are not the same:" in warning_log_str
        )

    @mock.patch.object(
        bilby.gw.detector.Interferometer, "set_strain_data_from_power_spectral_density"
    )
    def test_set_strain_data_from_power_spectral_density(self, m):
        self.ifo_list.set_strain_data_from_power_spectral_densities(
            sampling_frequency=123, duration=6.2, start_time=3
        )
        m.assert_called_with(sampling_frequency=123, duration=6.2, start_time=3)
        self.assertEqual(len(self.ifo_list), m.call_count)

    def test_inject_signal_pol_and_wg_none(self):
        with self.assertRaises(ValueError):
            self.ifo_list.inject_signal(
                injection_polarizations=None, waveform_generator=None
            )

    def test_meta_data(self):
        ifos_list = [self.ifo1, self.ifo2]
        ifos = bilby.gw.detector.InterferometerList(ifos_list)
        self.assertTrue(isinstance(ifos.meta_data, dict))
        meta_data = {ifo.name: ifo.meta_data for ifo in ifos_list}
        self.assertEqual(ifos.meta_data, meta_data)

    @mock.patch.object(
        bilby.gw.waveform_generator.WaveformGenerator, "frequency_domain_strain"
    )
    def test_inject_signal_pol_none_calls_frequency_domain_strain(self, m):
        waveform_generator = bilby.gw.waveform_generator.WaveformGenerator(
            frequency_domain_source_model=lambda x, y, z: x
        )
        self.ifo1.inject_signal = mock.MagicMock(return_value=None)
        self.ifo2.inject_signal = mock.MagicMock(return_value=None)
        self.ifo_list.inject_signal(
            parameters=None, waveform_generator=waveform_generator
        )
        self.assertTrue(m.called)

    @mock.patch.object(bilby.gw.detector.Interferometer, "inject_signal")
    def test_inject_signal_with_inj_pol(self, m):
        self.ifo_list.inject_signal(
            injection_polarizations=dict(plus=1), raise_error=False
        )
        m.assert_called_with(
            parameters=None, injection_polarizations=dict(plus=1), raise_error=False
        )
        self.assertEqual(len(self.ifo_list), m.call_count)

    @mock.patch.object(bilby.gw.detector.Interferometer, "inject_signal")
    def test_inject_signal_returns_expected_polarisations(self, m):
        m.return_value = dict(plus=1, cross=2)
        injection_polarizations = dict(plus=1, cross=2)
        ifos_pol = self.ifo_list.inject_signal(
            injection_polarizations=injection_polarizations
        )
        self.assertDictEqual(
            self.ifo1.inject_signal(injection_polarizations=injection_polarizations),
            ifos_pol[0],
        )
        self.assertDictEqual(
            self.ifo2.inject_signal(injection_polarizations=injection_polarizations),
            ifos_pol[1],
        )

    @mock.patch.object(bilby.gw.detector.Interferometer, "save_data")
    def test_save_data(self, m):
        self.ifo_list.save_data(outdir="test_outdir", label="test_outdir")
        m.assert_called_with(outdir="test_outdir", label="test_outdir")
        self.assertEqual(len(self.ifo_list), m.call_count)

    def test_number_of_interferometers(self):
        self.assertEqual(len(self.ifo_list), self.ifo_list.number_of_interferometers)

    def test_duration(self):
        self.assertEqual(self.ifo1.strain_data.duration, self.ifo_list.duration)
        self.assertEqual(self.ifo2.strain_data.duration, self.ifo_list.duration)

    def test_sampling_frequency(self):
        self.assertEqual(
            self.ifo1.strain_data.sampling_frequency, self.ifo_list.sampling_frequency
        )
        self.assertEqual(
            self.ifo2.strain_data.sampling_frequency, self.ifo_list.sampling_frequency
        )

    def test_start_time(self):
        self.assertEqual(self.ifo1.strain_data.start_time, self.ifo_list.start_time)
        self.assertEqual(self.ifo2.strain_data.start_time, self.ifo_list.start_time)

    def test_frequency_array(self):
        self.assertTrue(
            np.array_equal(
                self.ifo1.strain_data.frequency_array, self.ifo_list.frequency_array
            )
        )
        self.assertTrue(
            np.array_equal(
                self.ifo2.strain_data.frequency_array, self.ifo_list.frequency_array
            )
        )

    def test_append_with_ifo(self):
        self.ifo_list.append(self.ifo2)
        names = [ifo.name for ifo in self.ifo_list]
        self.assertListEqual([self.ifo1.name, self.ifo2.name, self.ifo2.name], names)

    def test_append_with_ifo_list(self):
        self.ifo_list.append(self.ifo_list)
        names = [ifo.name for ifo in self.ifo_list]
        self.assertListEqual(
            [self.ifo1.name, self.ifo2.name, self.ifo1.name, self.ifo2.name], names
        )

    def test_extend(self):
        self.ifo_list.extend(self.ifo_list)
        names = [ifo.name for ifo in self.ifo_list]
        self.assertListEqual(
            [self.ifo1.name, self.ifo2.name, self.ifo1.name, self.ifo2.name], names
        )

    def test_insert(self):
        new_ifo = self.ifo1
        new_ifo.name = "name3"
        self.ifo_list.insert(1, new_ifo)
        names = [ifo.name for ifo in self.ifo_list]
        self.assertListEqual([self.ifo1.name, new_ifo.name, self.ifo2.name], names)

    def test_to_and_from_pkl_loading(self):
        self.ifo_list.to_pickle(outdir="outdir", label="test")
        filename = "outdir/test_name1name2.pkl"
        recovered_ifo = bilby.gw.detector.InterferometerList.from_pickle(filename)
        self.assertListEqual(self.ifo_list, recovered_ifo)

    def test_to_and_from_pkl_wrong_class(self):
        import dill

        with open("./outdir/psd.pkl", "wb") as ff:
            dill.dump(self.ifo_list[0].power_spectral_density, ff)
        filename = self.ifo_list._filename_from_outdir_label_extension(
            outdir="outdir", label="psd", extension="pkl"
        )
        with self.assertRaises(TypeError):
            bilby.gw.detector.InterferometerList.from_pickle(filename)

    def test_plot_data(self):
        ifos = bilby.gw.detector.InterferometerList(["H1", "L1"])
        ifos.set_strain_data_from_power_spectral_densities(2048, 4)
        ifos.plot_data(outdir=self.outdir)

        ifos = bilby.gw.detector.InterferometerList(["H1", "L1", "V1"])
        ifos.set_strain_data_from_power_spectral_densities(2048, 4)
        ifos.plot_data(outdir=self.outdir)

    def test_plot_time_domain_data(self):
        ifos = bilby.gw.detector.InterferometerList(["H1", "L1"])
        ifos.set_strain_data_from_power_spectral_densities(2048, 4)
        ifos.plot_time_domain_data(outdir=self.outdir)

        ifos = bilby.gw.detector.InterferometerList(["H1", "L1", "V1"])
        ifos.set_strain_data_from_power_spectral_densities(2048, 4)
        ifos.plot_time_domain_data(outdir=self.outdir)


class TriangularInterferometerTest(unittest.TestCase):
    def setUp(self):
        self.triangular_ifo = bilby.gw.detector.get_empty_interferometer("ET")

    def tearDown(self):
        del self.triangular_ifo

    def test_individual_positions(self):
        """
        Check that the distances between the vertices of the three
        individual interferometers is approximately equal to the
        length of the arms of the detector.
        Calculation following:
        https://www.movable-type.co.uk/scripts/latlong.html
        Angles must be in radians
        """

        def a(delta_lat, delta_long, lat_1, lat_2):
            return (
                np.sin(delta_lat / 2) ** 2
                + np.cos(lat_1) * np.cos(lat_2) * np.sin(delta_long / 2) ** 2
            )

        def c(a):
            return 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

        for pair in list(combinations(self.triangular_ifo, 2)):
            delta_lat = np.radians(pair[1].latitude - pair[0].latitude)
            delta_long = np.radians(pair[1].longitude - pair[0].longitude)
            pair_a = a(delta_lat, delta_long, pair[0].latitude, pair[1].latitude)
            pair_c = c(pair_a)
            distance = bilby.core.utils.radius_of_earth * pair_c
            self.assertAlmostEqual(distance / 1000, pair[0].length, delta=1)


if __name__ == "__main__":
    unittest.main()
