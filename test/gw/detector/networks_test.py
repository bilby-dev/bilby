import unittest
from unittest import mock
from shutil import rmtree
from itertools import combinations

import numpy as np
import pytest

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
        m.assert_called_with(
            sampling_frequency=123, duration=6.2, start_time=3, random_state=None
        )
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


class TestLegacyDetectorNameAlias(unittest.TestCase):
    def test_geo600_alias_resolves_to_g1(self):
        """GEO600 was renamed to G1; the old name must keep working."""
        ifo = bilby.gw.detector.get_empty_interferometer("GEO600")
        self.assertEqual(ifo.name, "G1")

    def test_geo600_alias_matches_g1_directly(self):
        alias_ifo = bilby.gw.detector.get_empty_interferometer("GEO600")
        direct_ifo = bilby.gw.detector.get_empty_interferometer("G1")
        self.assertEqual(alias_ifo.name, direct_ifo.name)
        self.assertEqual(
            alias_ifo.power_spectral_density.psd_file,
            direct_ifo.power_spectral_density.psd_file,
        )


@pytest.mark.array_backend
@pytest.mark.usefixtures("xp_class")
class TriangularInterferometerTest(unittest.TestCase):
    def setUp(self):
        self.triangular_ifo = bilby.gw.detector.get_empty_interferometer("ET")
        self.triangular_ifo.array_backend = self.xp

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
        xp = self.xp

        def a(delta_lat, delta_long, lat_1, lat_2):
            return (
                xp.sin(delta_lat / 2) ** 2
                + xp.cos(lat_1) * xp.cos(lat_2) * xp.sin(delta_long / 2) ** 2
            )

        def c(a):
            return 2 * xp.atan2(xp.sqrt(a), xp.sqrt(1 - a))

        for pair in combinations(self.triangular_ifo, 2):
            lat_0 = pair[0].latitude_radians
            lat_1 = pair[1].latitude_radians
            long_0 = pair[0].longitude_radians
            long_1 = pair[1].longitude_radians
            pair_a = a(lat_1 - lat_0, long_1 - long_0, lat_0, lat_1)
            pair_c = c(pair_a)
            distance = bilby.core.utils.radius_of_earth * pair_c
            self.assertAlmostEqual(
                float(distance) / 1000, float(pair[0].length), delta=1
            )

    def test_individual_positions_exact(self):
        """
        Check that the distances between the vertices of the three
        individual interferometers is exactly equal to the
        length of the arms of the detector.
        """
        for pair in combinations(self.triangular_ifo, 2):
            vertex_0 = pair[0].vertex_position_geocentric()
            vertex_1 = pair[1].vertex_position_geocentric()
            self.assertAlmostEqual(
                float(pair[0].length), float(pair[1].length), delta=1e-6
            )
            distance = self.xp.linalg.vector_norm(vertex_0 - vertex_1) / 1000
            self.assertAlmostEqual(float(distance), float(pair[0].length), delta=1e-6)

    def test_arm_endpoints_coincide_with_neighbouring_vertices(self):
        """
        Check that the end of the x-arm of each interferometer coincides
        with the vertex of the next interferometer and the end of the
        y-arm with the vertex of the previous one. This tests the vertex
        positions and arm orientations together.
        """
        n_ifos = len(self.triangular_ifo)
        for ii, ifo in enumerate(self.triangular_ifo):
            vertex = ifo.vertex_position_geocentric()
            arm_length = ifo.length * 1000
            next_vertex = self.triangular_ifo[
                (ii + 1) % n_ifos
            ].vertex_position_geocentric()
            previous_vertex = self.triangular_ifo[
                (ii - 1) % n_ifos
            ].vertex_position_geocentric()
            x_arm_end = vertex + arm_length * ifo.geometry.x
            y_arm_end = vertex + arm_length * ifo.geometry.y
            self.assertLess(
                float(self.xp.linalg.vector_norm(x_arm_end - next_vertex)), 1e-6
            )
            self.assertLess(
                float(self.xp.linalg.vector_norm(y_arm_end - previous_vertex)), 1e-6
            )

    def test_detector_tensors_sum_to_zero(self):
        """
        Check that the detector tensors of the three interferometers sum
        to zero, i.e., that the null stream of the triangular detector
        vanishes for any signal.
        """
        total = sum(ifo.detector_tensor for ifo in self.triangular_ifo)
        self.assertLess(float(self.xp.max(self.xp.abs(total))), 1e-10)


class TestInterferometerListTimeDomain(unittest.TestCase):
    def setUp(self):
        self.gps_start = 1126259640.0
        self.sky = dict(ra=1.375, dec=-1.2108, psi=2.659)
        self.ifos = bilby.gw.detector.InterferometerList(["H1", "L1"])
        self.ifos.set_strain_data_from_zero_noise(
            sampling_frequency=1024.0, duration=8, start_time=self.gps_start
        )

    def tearDown(self):
        del self.ifos

    @staticmethod
    def _burst_at_model_times():
        def polarizations_at(times):
            envelope = 1e-21 * np.exp(-((np.asarray(times) / 0.04) ** 2))
            return dict(
                plus=envelope * np.cos(2 * np.pi * 150 * np.asarray(times)),
                cross=envelope * np.sin(2 * np.pi * 150 * np.asarray(times)),
            )

        return dict(polarizations_at=polarizations_at)

    def test_imr_windows_are_the_same_in_every_interferometer(self):
        self.ifos.set_analysis_windows(duration=2, start_time=self.gps_start + 3)
        for ifo in self.ifos:
            self.assertEqual(
                (ifo.analysis_start_time, ifo.analysis_duration),
                (self.gps_start + 3, 2),
            )
            self.assertIsNone(ifo.analysis_cut_time)
            self.assertEqual(
                ifo.meta_data["analysis_window"]["analysis_segment"], "imr"
            )

    def test_cut_time_includes_the_detector_time_delay(self):
        reference = dict(self.sky, geocent_time=self.gps_start + 4.3)
        self.ifos.set_analysis_windows(
            duration=1,
            analysis_segment="inspiral",
            segment_cut_time=-0.01,
            cut_reference_parameters=reference,
        )
        for ifo in self.ifos:
            delay = ifo.time_delay_from_geocenter(
                reference["ra"], reference["dec"], reference["geocent_time"]
            )
            self.assertAlmostEqual(
                ifo.analysis_cut_time,
                reference["geocent_time"] - 0.01 + delay,
                places=9,
            )
            self.assertAlmostEqual(
                ifo.analysis_start_time, ifo.analysis_cut_time - 1, places=9
            )
            self.assertEqual(
                ifo.meta_data["analysis_window"]["segment_cut_time"], -0.01
            )

    def test_inspiral_and_post_inspiral_windows_meet_at_the_cut(self):
        """For cuts on and off the sample grid, the inspiral window ends where the
        post-inspiral window starts, at the sample nearest the cut"""
        for fraction in [0.0, 0.3, 0.5001, 0.7]:
            reference = dict(
                self.sky, geocent_time=self.gps_start + 4 + fraction / 1024
            )
            slices, cuts = dict(), dict()
            for segment in ["inspiral", "post_inspiral"]:
                self.ifos.set_analysis_windows(
                    duration=1,
                    analysis_segment=segment,
                    segment_cut_time=0.0,
                    cut_reference_parameters=reference,
                )
                slices[segment] = [ifo.analysis_slice for ifo in self.ifos]
                cuts[segment] = [ifo.analysis_cut_time for ifo in self.ifos]
            for ifo, inspiral, post_inspiral, cut in zip(
                self.ifos, slices["inspiral"], slices["post_inspiral"], cuts["inspiral"]
            ):
                self.assertEqual(inspiral.stop, post_inspiral.start)
                self.assertEqual(inspiral.stop - inspiral.start, 1024)
                self.assertLessEqual(
                    abs(ifo.time_array[post_inspiral.start] - cut), 0.5 / 1024 + 1e-9
                )

    def test_unknown_analysis_segment_raises(self):
        with self.assertRaises(ValueError):
            self.ifos.set_analysis_windows(
                duration=1, start_time=self.gps_start, analysis_segment="ringdown"
            )

    def test_strain_data_start_times_may_differ_with_time_domain_windows(self):
        h1, l1 = self.ifos
        l1.set_strain_data_from_zero_noise(
            sampling_frequency=1024.0, duration=8, start_time=self.gps_start + 3
        )
        h1.set_analysis_window(self.gps_start + 4, 2)
        l1.set_analysis_window(self.gps_start + 4.5, 2)
        bilby.gw.detector.InterferometerList([h1, l1])
        h1.analysis_duration = None
        with self.assertRaises(ValueError):
            bilby.gw.detector.InterferometerList([h1, l1])

    def test_interferometers_without_window_attributes(self):
        ifo = bilby.gw.detector.get_empty_interferometer("H1")
        del (
            ifo.analysis_duration
        )  # e.g. unpickled from an older bilby: the class default is used
        bilby.gw.detector.InterferometerList([ifo])

    def test_set_autocovariance_functions(self):
        self.ifos.set_analysis_windows(duration=2, start_time=self.gps_start + 3)
        self.ifos.set_autocovariance_functions_from_power_spectral_densities(
            fill_multiplier=10
        )
        for ifo in self.ifos:
            self.assertIsNotNone(ifo.gohberg_semencul_vectors)
            self.assertEqual(
                ifo.meta_data["autocovariance_function"]["fill_multiplier"], 10
            )

    def test_time_domain_injection(self):
        self.ifos.set_analysis_windows(duration=2, start_time=self.gps_start + 3)
        self.ifos.set_autocovariance_functions_from_power_spectral_densities()
        parameters = dict(self.sky, geocent_time=self.gps_start + 4.3)
        polarizations = self.ifos.inject_signal(
            parameters=parameters,
            injection_polarizations=self._burst_at_model_times(),
            domain="time",
            placement="model_time",
        )
        self.assertEqual(len(polarizations), 2)
        for ifo in self.ifos:
            self.assertGreater(ifo.meta_data["optimal_SNR"], 0)
            self.assertEqual(ifo.meta_data["injection_placement"], "model_time")

    def test_time_domain_injection_errors(self):
        self.ifos.set_analysis_windows(duration=2, start_time=self.gps_start + 3)
        self.ifos.set_autocovariance_functions_from_power_spectral_densities()
        parameters = dict(self.sky, geocent_time=self.gps_start + 4.3)
        generator = bilby.gw.WaveformGenerator(
            duration=8,
            sampling_frequency=2048,
            time_domain_source_model=lambda time_array: dict(),
        )
        for kwargs in [
            dict(domain="time"),
            dict(waveform_generator=generator, domain="time"),
            dict(
                injection_polarizations=self._burst_at_model_times(),
                domain="time",
                raise_error=False,
            ),
            dict(
                injection_polarizations=self._burst_at_model_times(), domain="wavelet"
            ),
            dict(
                injection_polarizations=dict(plus=np.zeros(4097, dtype=complex)),
                placement="fd_shift",
            ),
        ]:
            with self.assertRaises(ValueError):
                self.ifos.inject_signal(parameters=parameters, **kwargs)


if __name__ == "__main__":
    unittest.main()
