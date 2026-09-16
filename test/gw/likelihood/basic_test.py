import unittest

import array_api_compat as aac
import bilby
import numpy as np
import pytest

from utils import BackendWaveformGenerator


@pytest.mark.array_backend
@pytest.mark.usefixtures("xp_class")
class TestBasicGWTransient(unittest.TestCase):
    def setUp(self):
        bilby.core.utils.random.seed(500)
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
            phase=1.3,
            geocent_time=self.xp.asarray(1126259642.413),
            ra=self.xp.asarray(1.375),
            dec=self.xp.asarray(-1.2108),
        )
        self.interferometers = bilby.gw.detector.InterferometerList(["H1"])
        self.interferometers.set_strain_data_from_power_spectral_densities(
            sampling_frequency=self.xp.asarray(2048.0), duration=self.xp.asarray(4.0)
        )
        self.interferometers.set_array_backend(self.xp)
        base_wfg = bilby.gw.waveform_generator.GWSignalWaveformGenerator(
            duration=self.xp.asarray(4.0),
            sampling_frequency=self.xp.asarray(2048.0),
            waveform_arguments=dict(waveform_approximant="IMRPhenomPv2"),
        )
        self.waveform_generator = BackendWaveformGenerator(base_wfg, self.xp)

        self.likelihood = bilby.gw.likelihood.BasicGravitationalWaveTransient(
            interferometers=self.interferometers,
            waveform_generator=self.waveform_generator,
        )

    def tearDown(self):
        del self.parameters
        del self.interferometers
        del self.waveform_generator
        del self.likelihood

    def test_noise_log_likelihood(self):
        """Test noise log likelihood matches precomputed value"""
        nll = self.likelihood.noise_log_likelihood()
        self.assertAlmostEqual(
            -4014.1787704539474, float(nll), 3
        )
        self.assertEqual(aac.get_namespace(nll), self.xp)

    def test_log_likelihood(self):
        """Test log likelihood matches precomputed value"""
        logl = self.likelihood.log_likelihood(self.parameters)
        self.assertAlmostEqual(float(logl), -4032.4397343470005, 3)
        self.assertEqual(aac.get_namespace(logl), self.xp)

    def test_log_likelihood_ratio(self):
        """Test log likelihood ratio returns the correct value"""
        llr = self.likelihood.log_likelihood_ratio(self.parameters)
        self.assertAlmostEqual(
            self.likelihood.log_likelihood(self.parameters) - self.likelihood.noise_log_likelihood(),
            llr,
            3,
        )
        self.assertEqual(aac.get_namespace(llr), self.xp)

    def test_likelihood_zero_when_waveform_is_none(self):
        """Test log likelihood returns np.nan_to_num(-np.inf) when the
        waveform is None"""
        self.likelihood.waveform_generator.frequency_domain_strain = lambda x: None
        self.assertEqual(self.likelihood.log_likelihood_ratio(self.parameters), np.nan_to_num(-np.inf))

    def test_repr(self):
        expected = "BasicGravitationalWaveTransient(interferometers={},\n\twaveform_generator={})".format(
            self.interferometers, self.waveform_generator
        )
        self.assertEqual(expected, repr(self.likelihood))
