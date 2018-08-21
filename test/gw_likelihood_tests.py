from __future__ import division, absolute_import
import unittest
import mock
import tupak
import numpy as np
from scipy.special import logsumexp


class TestBasicGWTransient(unittest.TestCase):

    def setUp(self):
        np.random.seed(500)
        self.parameters = dict(
            mass_1=31., mass_2=29., a_1=0.4, a_2=0.3, tilt_1=0.0, tilt_2=0.0,
            phi_12=1.7, phi_jl=0.3, luminosity_distance=4000., iota=0.4,
            psi=2.659, phase=1.3, geocent_time=1126259642.413, ra=1.375,
            dec=-1.2108)
        self.interferometers = tupak.gw.detector.InterferometerList(['H1'])
        self.interferometers.set_strain_data_from_power_spectral_densities(
            sampling_frequency=2048, duration=4)
        self.waveform_generator = tupak.gw.waveform_generator.WaveformGenerator(
            duration=4, sampling_frequency=2048, parameters=self.parameters,
            frequency_domain_source_model=tupak.gw.source.lal_binary_black_hole,
            )

        self.likelihood = tupak.gw.likelihood.BasicGravitationalWaveTransient(
            interferometers=self.interferometers,
            waveform_generator=self.waveform_generator
        )

    def tearDown(self):
        del self.parameters
        del self.interferometers
        del self.waveform_generator
        del self.likelihood

    def test_noise_log_likelihood(self):
        """Test noise log likelihood matches precomputed value"""
        self.likelihood.noise_log_likelihood()
        self.assertAlmostEqual(self.likelihood.noise_log_likelihood(),
                               -4036.1064342687155, 3)

    def test_log_likelihood(self):
        """Test log likelihood matches precomputed value"""
        self.likelihood.log_likelihood()
        self.assertAlmostEqual(self.likelihood.log_likelihood(),
                               -4051.754299050376, 3)

    def test_log_likelihood_ratio(self):
        """Test log likelihood ratio returns the correct value"""
        self.assertAlmostEqual(
            self.likelihood.log_likelihood()
            - self.likelihood.noise_log_likelihood(),
            self.likelihood.log_likelihood_ratio(), 3)


class TestGWTransient(unittest.TestCase):

    def setUp(self):
        np.random.seed(500)
        self.duration = 4
        self.sampling_frequency = 2048
        self.parameters = dict(
            mass_1=31., mass_2=29., a_1=0.4, a_2=0.3, tilt_1=0.0, tilt_2=0.0,
            phi_12=1.7, phi_jl=0.3, luminosity_distance=4000., iota=0.4,
            psi=2.659, phase=1.3, geocent_time=1126259642.413, ra=1.375,
            dec=-1.2108)
        self.interferometers = tupak.gw.detector.InterferometerList(['H1'])
        self.interferometers.set_strain_data_from_power_spectral_densities(
            sampling_frequency=self.sampling_frequency, duration=self.duration)
        self.waveform_generator = tupak.gw.waveform_generator.WaveformGenerator(
            duration=self.duration, sampling_frequency=self.sampling_frequency,
            parameters=self.parameters.copy(),
            frequency_domain_source_model=tupak.gw.source.lal_binary_black_hole,
        )

        self.prior = tupak.gw.prior.BBHPriorSet()
        self.prior['geocent_time'] = tupak.prior.Uniform(
            minimum=self.parameters['geocent_time'] - self.duration / 2,
            maximum=self.parameters['geocent_time'] + self.duration / 2)

        self.likelihood = tupak.gw.likelihood.GravitationalWaveTransient(
            interferometers=self.interferometers,
            waveform_generator=self.waveform_generator, prior=self.prior.copy()
        )

        self.phase = tupak.gw.likelihood.GravitationalWaveTransient(
            interferometers=self.interferometers,
            waveform_generator=self.waveform_generator,
            phase_marginalization=True, prior=self.prior.copy()
        )

        self.time = tupak.gw.likelihood.GravitationalWaveTransient(
            interferometers=self.interferometers,
            waveform_generator=self.waveform_generator,
            time_marginalization=True, prior=self.prior.copy()
        )
        # self.distance = tupak.gw.likelihood.GravitationalWaveTransient(
        #     interferometers=self.interferometers,
        #     waveform_generator=self.waveform_generator,
        #     distance_marginalization=True, prior=self.prior
        # )

    def tearDown(self):
        del self.parameters
        del self.interferometers
        del self.waveform_generator
        del self.prior
        del self.likelihood
        del self.phase
        # del self.distance

    def test_noise_log_likelihood(self):
        """Test noise log likelihood matches precomputed value"""
        self.likelihood.noise_log_likelihood()
        self.assertAlmostEqual(self.likelihood.noise_log_likelihood(),
                               -4036.1064342687155, 3)

    def test_log_likelihood(self):
        """Test log likelihood matches precomputed value"""
        self.likelihood.log_likelihood()
        self.assertAlmostEqual(self.likelihood.log_likelihood(),
                               -4051.754299050375, 3)

    def test_log_likelihood_ratio(self):
        """Test log likelihood ratio returns the correct value"""
        self.assertAlmostEqual(
            self.likelihood.log_likelihood()
            - self.likelihood.noise_log_likelihood(),
            self.likelihood.log_likelihood_ratio(), 3)

    def test_phase_marginalisation(self):
        """Test phase marginalised likelihood matches brute force version"""
        like = []
        phases = np.linspace(0, 2 * np.pi, 1000)
        for phase in phases:
            self.waveform_generator.parameters['phase'] = phase
            like.append(np.exp(self.likelihood.log_likelihood_ratio()))

        marg_like = np.log(np.trapz(like, phases) / (2 * np.pi))
        self.waveform_generator.parameters = self.parameters.copy()
        self.assertAlmostEqual(marg_like, self.phase.log_likelihood_ratio(),
                               delta=0.5)

    def test_time_marginalisation(self):
        """Test time marginalised likelihood matches brute force version"""
        like = []
        times = np.linspace(self.prior['geocent_time'].minimum,
                            self.prior['geocent_time'].maximum, 4097)[:-1]
        for time in times:
            self.waveform_generator.parameters['geocent_time'] = time
            like.append(np.exp(self.likelihood.log_likelihood_ratio()))

        marg_like = np.log(np.trapz(like, times)
                           / self.waveform_generator.duration)
        self.waveform_generator.parameters = self.parameters.copy()
        self.assertAlmostEqual(marg_like, self.time.log_likelihood_ratio(),
                               delta=0.5)


if __name__ == '__main__':
    unittest.main()
