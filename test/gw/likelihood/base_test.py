import unittest

import array_api_compat as aac
import bilby
import numpy as np
import pytest

from utils import BackendWaveformGenerator


@pytest.mark.array_backend
@pytest.mark.usefixtures("xp_class")
class TestGWTransient(unittest.TestCase):
    def setUp(self):
        bilby.core.utils.random.seed(500)
        self.duration = self.xp.asarray(4.0)
        self.sampling_frequency = self.xp.asarray(2048.0)
        self.parameters = dict(
            mass_1=31.0,
            mass_2=29.0,
            a_1=0.4,
            a_2=0.3,
            tilt_1=0.0,
            tilt_2=0.0,
            phi_12=1.7,
            phi_jl=0.3,
            luminosity_distance=4000.0,
            theta_jn=0.4,
            psi=self.xp.asarray(2.659),
            phase=self.xp.asarray(1.3),
            geocent_time=self.xp.asarray(1126259642.413),
            ra=self.xp.asarray(1.375),
            dec=self.xp.asarray(-1.2108),
        )
        self.interferometers = bilby.gw.detector.InterferometerList(["H1"])
        self.interferometers.set_strain_data_from_power_spectral_densities(
            sampling_frequency=self.sampling_frequency, duration=self.duration
        )
        self.interferometers.set_array_backend(self.xp)
        wfg = bilby.gw.waveform_generator.WaveformGenerator(
            duration=self.duration,
            sampling_frequency=self.sampling_frequency,
            frequency_domain_source_model=bilby.gw.source.lal_binary_black_hole,
        )
        self.waveform_generator = BackendWaveformGenerator(wfg, self.xp)

        self.prior = bilby.gw.prior.BBHPriorDict()
        self.prior["geocent_time"] = bilby.prior.Uniform(
            minimum=self.parameters["geocent_time"] - self.duration / 2,
            maximum=self.parameters["geocent_time"] + self.duration / 2,
        )

        self.likelihood = bilby.gw.likelihood.GravitationalWaveTransient(
            interferometers=self.interferometers,
            waveform_generator=self.waveform_generator,
            priors=self.prior.copy(),
        )

    def tearDown(self):
        del self.parameters
        del self.interferometers
        del self.waveform_generator
        del self.prior
        del self.likelihood

    def test_noise_log_likelihood(self):
        """Test noise log likelihood matches precomputed value"""
        nll = self.likelihood.noise_log_likelihood()
        self.assertEqual(aac.get_namespace(nll), self.xp)
        self.assertAlmostEqual(
            -4014.1787704539474, float(nll), 3
        )

    def test_log_likelihood(self):
        """Test log likelihood matches precomputed value"""
        logl = self.likelihood.log_likelihood(self.parameters)
        self.assertAlmostEqual(float(logl), -4032.4397343470005, 3)
        self.assertEqual(aac.get_namespace(logl), self.xp)

    def test_log_likelihood_ratio(self):
        """Test log likelihood ratio returns the correct value"""
        llr = self.likelihood.log_likelihood_ratio(self.parameters)
        self.assertAlmostEqual(
            float(self.likelihood.log_likelihood(self.parameters)) - float(self.likelihood.noise_log_likelihood()),
            float(llr),
            3,
        )
        self.assertEqual(aac.get_namespace(llr), self.xp)

    def test_likelihood_zero_when_waveform_is_none(self):
        """Test log likelihood returns np.nan_to_num(-np.inf) when the
        waveform is None"""
        self.likelihood.waveform_generator.frequency_domain_strain = lambda x: None
        self.assertEqual(self.likelihood.log_likelihood_ratio(self.parameters), np.nan_to_num(-np.inf))

    def test_repr(self):
        expected = (
            "GravitationalWaveTransient(interferometers={},\n\twaveform_generator={},\n\t"
            "time_marginalization={}, distance_marginalization={}, phase_marginalization={}, "
            "calibration_marginalization={}, priors={})".format(
                self.interferometers,
                self.waveform_generator,
                False,
                False,
                False,
                False,
                self.prior,
            )
        )
        self.assertEqual(expected, repr(self.likelihood))

    def test_interferometers_setting_list(self):
        ifos = [
            bilby.gw.detector.get_empty_interferometer(name=name)
            for name in ["H1", "L1"]
        ]
        self.likelihood.interferometers = ifos
        self.assertListEqual(
            bilby.gw.detector.InterferometerList(ifos), self.likelihood.interferometers
        )
        self.assertIsInstance(self.likelihood.interferometers, bilby.gw.detector.InterferometerList)

    def test_interferometers_setting_interferometer_list(self):
        ifos = bilby.gw.detector.InterferometerList(
            [
                bilby.gw.detector.get_empty_interferometer(name=name)
                for name in ["H1", "L1"]
            ]
        )
        self.likelihood.interferometers = ifos
        self.assertListEqual(
            bilby.gw.detector.InterferometerList(ifos), self.likelihood.interferometers
        )
        self.assertIsInstance(self.likelihood.interferometers, bilby.gw.detector.InterferometerList)

    def test_meta_data(self):
        expected = dict(
            interferometers=self.interferometers.meta_data,
            time_marginalization=False,
            phase_marginalization=False,
            distance_marginalization=False,
            calibration_marginalization=False,
            waveform_generator_class=self.waveform_generator.__class__,
            waveform_arguments=self.waveform_generator.waveform_arguments,
            frequency_domain_source_model=self.waveform_generator.frequency_domain_source_model,
            time_domain_source_model=self.waveform_generator.time_domain_source_model,
            parameter_conversion=self.waveform_generator.parameter_conversion,
            sampling_frequency=self.waveform_generator.sampling_frequency,
            duration=self.waveform_generator.duration,
            start_time=self.waveform_generator.start_time,
            time_reference="geocent",
            reference_frame="sky",
            lal_version=self.likelihood.lal_version,
            lalsimulation_version=self.likelihood.lalsimulation_version,
            waveform_generator_meta_data=self.waveform_generator.meta_data,
        )
        self.assertDictEqual(expected, self.likelihood.meta_data)

    def test_reference_frame_agrees_with_default(self):
        new_likelihood = bilby.gw.likelihood.GravitationalWaveTransient(
            interferometers=self.interferometers,
            waveform_generator=self.waveform_generator,
            priors=self.prior.copy(),
            reference_frame="H1L1"
        )
        parameters = self.parameters.copy()
        del parameters["ra"], parameters["dec"]
        parameters["zenith"] = self.xp.asarray(1.0)
        parameters["azimuth"] = self.xp.asarray(1.0)
        parameters["ra"], parameters["dec"] = bilby.gw.utils.zenith_azimuth_to_ra_dec(
            zenith=parameters["zenith"],
            azimuth=parameters["azimuth"],
            geocent_time=parameters["geocent_time"],
            ifos=new_likelihood.reference_frame,
        )
        self.assertEqual(aac.get_namespace(parameters["ra"]), self.xp)
        self.assertEqual(aac.get_namespace(parameters["dec"]), self.xp)
        self.assertEqual(
            new_likelihood.log_likelihood_ratio(parameters),
            self.likelihood.log_likelihood_ratio(parameters)
        )

    def test_time_reference_agrees_with_default(self):
        new_likelihood = bilby.gw.likelihood.GravitationalWaveTransient(
            interferometers=self.interferometers,
            waveform_generator=self.waveform_generator,
            priors=self.prior.copy(),
            time_reference="H1"
        )
        ifo = bilby.gw.detector.get_empty_interferometer("H1")
        time_delay = ifo.time_delay_from_geocenter(
            ra=self.parameters["ra"],
            dec=self.parameters["dec"],
            time=self.parameters["geocent_time"]
        )
        parameters = self.parameters.copy()
        parameters["H1_time"] = parameters["geocent_time"] + time_delay
        self.assertAlmostEqual(
            new_likelihood.log_likelihood_ratio(parameters),
            self.likelihood.log_likelihood_ratio(parameters),
            8,
        )


class TestBBHLikelihoodSetUp(unittest.TestCase):
    def setUp(self):
        self.ifos = bilby.gw.detector.InterferometerList(["H1"])

    def tearDown(self):
        del self.ifos

    def test_instantiation(self):
        self.like = bilby.gw.likelihood.get_binary_black_hole_likelihood(self.ifos)
