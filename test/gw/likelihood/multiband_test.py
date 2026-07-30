import os
import tempfile
import unittest
from copy import deepcopy

import array_api_compat as aac
import bilby
import numpy as np
import pytest
from parameterized import parameterized

from utils import BackendWaveformGenerator


@pytest.mark.array_backend
@pytest.mark.usefixtures("xp_class")
class TestMBLikelihood(unittest.TestCase):
    def setUp(self):
        self.duration = self.xp.asarray(16.0)
        self.fmin = self.xp.asarray(20.0)
        self.sampling_frequency = self.xp.asarray(2048.0)
        self.test_parameters = dict(
            chirp_mass=6.0,
            mass_ratio=0.5,
            a_1=0.0,
            a_2=0.0,
            tilt_1=0.0,
            tilt_2=0.0,
            phi_12=0.0,
            phi_jl=0.0,
            luminosity_distance=200.0,
            theta_jn=0.4,
            psi=self.xp.asarray(0.659),
            phase=1.3,
            geocent_time=self.xp.asarray(1187008882),
            ra=self.xp.asarray(1.3),
            dec=self.xp.asarray(-1.2)
        )  # Network SNR is ~50
        # torch needs sky parameters to be tensors

        self.ifos = bilby.gw.detector.InterferometerList(["H1", "L1", "V1"])
        bilby.core.utils.random.seed(70817)
        rng = bilby.core.utils.random.rng
        self.ifos.set_strain_data_from_power_spectral_densities(
            sampling_frequency=self.sampling_frequency, duration=self.duration,
            start_time=self.test_parameters['geocent_time'] - self.duration + 2.
        )
        for ifo in self.ifos:
            ifo.minimum_frequency = self.fmin
        self.ifos.set_array_backend(self.xp)

        spline_calibration_nodes = 10
        self.calibration_parameters = {}
        for ifo in self.ifos:
            ifo.calibration_model = bilby.gw.calibration.CubicSpline(
                prefix=f"recalib_{ifo.name}_",
                minimum_frequency=ifo.minimum_frequency,
                maximum_frequency=ifo.maximum_frequency,
                n_points=spline_calibration_nodes
            )
            for i in range(spline_calibration_nodes):
                self.test_parameters[f"recalib_{ifo.name}_amplitude_{i}"] = 0
                self.test_parameters[f"recalib_{ifo.name}_phase_{i}"] = 0
                # Calibration errors of 5% in amplitude and 5 degrees in phase
                self.calibration_parameters[f"recalib_{ifo.name}_amplitude_{i}"] = \
                    rng.normal(loc=0, scale=0.05)
                self.calibration_parameters[f"recalib_{ifo.name}_phase_{i}"] = \
                    rng.normal(loc=0, scale=5 * np.pi / 180)

        self.priors = bilby.gw.prior.BBHPriorDict()
        self.priors.pop("mass_1")
        self.priors.pop("mass_2")
        self.priors["chirp_mass"] = bilby.core.prior.Uniform(5.5, 6.5)
        self.priors["mass_ratio"] = bilby.core.prior.Uniform(0.125, 1)
        self.priors["geocent_time"] = bilby.core.prior.Uniform(
            self.test_parameters['geocent_time'] - 0.1,
            self.test_parameters['geocent_time'] + 0.1)

    def tearDown(self):
        del (
            self.ifos,
            self.priors
        )

    @parameterized.expand([
        ("IMRPhenomD", True, 2, False, 1.5e-2),
        ("IMRPhenomD", True, 2, True, 1.5e-2),
        ("IMRPhenomD", False, 2, False, 5e-3),
        ("IMRPhenomD", False, 2, True, 6e-3),
        ("IMRPhenomHM", False, 4, False, 8e-4),
        ("IMRPhenomHM", False, 4, True, 1e-3)
    ])
    def test_matches_original_likelihood(
        self, waveform_approximant, linear_interpolation, highest_mode, add_cal_errors, tolerance
    ):
        """
        Check if multi-band likelihood values match original likelihood values
        """
        wfg = bilby.gw.WaveformGenerator(
            duration=self.duration, sampling_frequency=self.sampling_frequency,
            frequency_domain_source_model=bilby.gw.source.lal_binary_black_hole,
            waveform_arguments=dict(
                reference_frequency=self.fmin, waveform_approximant=waveform_approximant
            )
        )
        wfg = BackendWaveformGenerator(wfg, self.xp)
        self.ifos.inject_signal(parameters=self.test_parameters, waveform_generator=wfg)

        wfg_mb = bilby.gw.WaveformGenerator(
            duration=self.duration, sampling_frequency=self.sampling_frequency,
            frequency_domain_source_model=bilby.gw.source.binary_black_hole_frequency_sequence,
            waveform_arguments=dict(
                reference_frequency=self.fmin, waveform_approximant=waveform_approximant
            )
        )
        wfg_mb = BackendWaveformGenerator(wfg_mb, self.xp)
        likelihood = bilby.gw.likelihood.GravitationalWaveTransient(
            interferometers=self.ifos, waveform_generator=wfg
        )
        likelihood_mb = bilby.gw.likelihood.MBGravitationalWaveTransient(
            interferometers=self.ifos, waveform_generator=wfg_mb,
            reference_chirp_mass=self.test_parameters['chirp_mass'],
            priors=self.priors.copy(), linear_interpolation=linear_interpolation,
            highest_mode=highest_mode
        )
        parameters = deepcopy(self.test_parameters)
        if add_cal_errors:
            parameters.update(self.calibration_parameters)
        ll = likelihood.log_likelihood_ratio(parameters)
        llmb = likelihood_mb.log_likelihood_ratio(parameters)
        self.assertLess(abs(ll - llmb), tolerance)
        self.assertEqual(aac.get_namespace(llmb), self.xp)

    def test_large_accuracy_factor(self):
        """
        Check if larger accuracy factor increases the accuracy.
        """
        waveform_approximant = "IMRPhenomD"
        wfg = bilby.gw.WaveformGenerator(
            duration=self.duration, sampling_frequency=self.sampling_frequency,
            frequency_domain_source_model=bilby.gw.source.lal_binary_black_hole,
            waveform_arguments=dict(
                reference_frequency=self.fmin, waveform_approximant=waveform_approximant
            )
        )
        wfg = BackendWaveformGenerator(wfg, self.xp)
        self.ifos.inject_signal(parameters=self.test_parameters, waveform_generator=wfg)

        wfg_mb = bilby.gw.WaveformGenerator(
            duration=self.duration, sampling_frequency=self.sampling_frequency,
            frequency_domain_source_model=bilby.gw.source.binary_black_hole_frequency_sequence,
            waveform_arguments=dict(
                reference_frequency=self.fmin, waveform_approximant=waveform_approximant
            )
        )
        wfg_mb = BackendWaveformGenerator(wfg_mb, self.xp)
        likelihood = bilby.gw.likelihood.GravitationalWaveTransient(
            interferometers=self.ifos, waveform_generator=wfg
        )
        likelihood_mb = bilby.gw.likelihood.MBGravitationalWaveTransient(
            interferometers=self.ifos, waveform_generator=wfg_mb,
            reference_chirp_mass=self.test_parameters['chirp_mass'],
            priors=self.priors.copy(), accuracy_factor=5
        )
        likelihood_mb_more_accurate = bilby.gw.likelihood.MBGravitationalWaveTransient(
            interferometers=self.ifos, waveform_generator=wfg_mb,
            reference_chirp_mass=self.test_parameters['chirp_mass'],
            priors=self.priors.copy(), accuracy_factor=50
        )
        self.assertLess(
            abs(
                likelihood.log_likelihood_ratio(self.test_parameters)
                - likelihood_mb_more_accurate.log_likelihood_ratio(self.test_parameters)
            ),
            abs(
                likelihood.log_likelihood_ratio(self.test_parameters)
                - likelihood_mb.log_likelihood_ratio(self.test_parameters)
            ) / 2
        )

    def test_reference_chirp_mass_from_prior(self):
        """
        Check if reference chirp mass is automatically determined from prior if no number has been passed
        """
        wfg_mb = bilby.gw.WaveformGenerator(
            duration=self.duration, sampling_frequency=self.sampling_frequency,
            frequency_domain_source_model=bilby.gw.source.binary_black_hole_frequency_sequence,
            waveform_arguments=dict(
                reference_frequency=self.fmin, waveform_approximant="IMRPhenomD"
            )
        )
        likelihood1 = bilby.gw.likelihood.MBGravitationalWaveTransient(
            interferometers=self.ifos, waveform_generator=wfg_mb,
            reference_chirp_mass=self.priors["chirp_mass"].minimum,
            priors=self.priors.copy()
        )
        likelihood2 = bilby.gw.likelihood.MBGravitationalWaveTransient(
            interferometers=self.ifos, waveform_generator=wfg_mb,
            priors=self.priors.copy()
        )
        self.assertAlmostEqual(likelihood1.reference_chirp_mass, likelihood2.reference_chirp_mass)

    def test_no_reference_chirp_mass(self):
        """
        Check if an error is raised if either reference_chirp_mass or priors is not specified.
        """
        wfg_mb = bilby.gw.WaveformGenerator(
            duration=self.duration, sampling_frequency=self.sampling_frequency,
            frequency_domain_source_model=bilby.gw.source.binary_black_hole_frequency_sequence,
            waveform_arguments=dict(
                reference_frequency=self.fmin, waveform_approximant="IMRPhenomD"
            )
        )
        with self.assertRaises(TypeError):
            bilby.gw.likelihood.MBGravitationalWaveTransient(
                interferometers=self.ifos, waveform_generator=wfg_mb
            )

    def test_cannot_determine_reference_chirp_mass(self):
        """
        Check if an error is raised if priors does not contain necessary information to determine reference chirp mass
        """
        wfg_mb = bilby.gw.WaveformGenerator(
            duration=self.duration, sampling_frequency=self.sampling_frequency,
            frequency_domain_source_model=bilby.gw.source.binary_black_hole_frequency_sequence,
            waveform_arguments=dict(
                reference_frequency=self.fmin, waveform_approximant="IMRPhenomD"
            )
        )
        for key in ["chirp_mass", "mass_1", "mass_2"]:
            if key in self.priors:
                self.priors.pop(key)
        with self.assertRaises(Exception):
            bilby.gw.likelihood.MBGravitationalWaveTransient(
                interferometers=self.ifos, waveform_generator=wfg_mb, priors=self.priors
            )

    @parameterized.expand([(True, ), (False, )])
    def test_inout_weights(self, linear_interpolation):
        """
        Check if multiband weights can be saved as a file, and a likelihood object constructed from the weights file
        produces the same likelihood value.
        """
        waveform_approximant = "IMRPhenomD"
        wfg = bilby.gw.WaveformGenerator(
            duration=self.duration, sampling_frequency=self.sampling_frequency,
            frequency_domain_source_model=bilby.gw.source.lal_binary_black_hole,
            waveform_arguments=dict(
                reference_frequency=self.fmin, waveform_approximant=waveform_approximant
            )
        )
        wfg = BackendWaveformGenerator(wfg, self.xp)
        self.ifos.inject_signal(
            parameters=self.test_parameters, waveform_generator=wfg
        )

        wfg_mb = bilby.gw.WaveformGenerator(
            duration=self.duration, sampling_frequency=self.sampling_frequency,
            frequency_domain_source_model=bilby.gw.source.binary_black_hole_frequency_sequence,
            waveform_arguments=dict(
                reference_frequency=self.fmin, waveform_approximant=waveform_approximant
            )
        )
        wfg_mb = BackendWaveformGenerator(wfg_mb, self.xp)
        likelihood_mb = bilby.gw.likelihood.MBGravitationalWaveTransient(
            interferometers=self.ifos, waveform_generator=wfg_mb,
            reference_chirp_mass=self.test_parameters['chirp_mass'],
            linear_interpolation=linear_interpolation,
        )
        llr = likelihood_mb.log_likelihood_ratio(self.test_parameters)

        with tempfile.TemporaryDirectory() as tmpdirname:
            # check if weights can be saved as a file
            filepath = os.path.join(tmpdirname, "weights.hdf5")
            likelihood_mb.save_weights(filepath)
            self.assertTrue(os.path.exists(filepath))

            # reset waveform generator to check if likelihood recovered from the weights file properly adds banded
            # frequency points to waveform arguments
            wfg_mb = bilby.gw.WaveformGenerator(
                duration=self.duration, sampling_frequency=self.sampling_frequency,
                frequency_domain_source_model=bilby.gw.source.binary_black_hole_frequency_sequence,
                waveform_arguments=dict(
                    reference_frequency=self.fmin, waveform_approximant=waveform_approximant
                )
            )
            wfg_mb = BackendWaveformGenerator(wfg_mb, self.xp)
            likelihood_mb_from_weights = bilby.gw.likelihood.MBGravitationalWaveTransient(
                interferometers=self.ifos, waveform_generator=wfg_mb, weights=filepath
            )

        llr_from_weights = likelihood_mb_from_weights.log_likelihood_ratio(self.test_parameters)
        self.assertEqual(aac.get_namespace(llr_from_weights), self.xp)

        self.assertAlmostEqual(llr, llr_from_weights)

    @parameterized.expand([(True, ), (False, )])
    def test_from_dict_weights(self, linear_interpolation):
        """
        Check if a likelihood object constructed from dictionary-like weights produce the same likelihood value
        """
        waveform_approximant = "IMRPhenomD"
        wfg = bilby.gw.WaveformGenerator(
            duration=self.duration, sampling_frequency=self.sampling_frequency,
            frequency_domain_source_model=bilby.gw.source.lal_binary_black_hole,
            waveform_arguments=dict(
                reference_frequency=self.fmin, waveform_approximant=waveform_approximant
            )
        )
        wfg = BackendWaveformGenerator(wfg, self.xp)
        self.ifos.inject_signal(
            parameters=self.test_parameters, waveform_generator=wfg
        )

        wfg_mb = bilby.gw.WaveformGenerator(
            duration=self.duration, sampling_frequency=self.sampling_frequency,
            frequency_domain_source_model=bilby.gw.source.binary_black_hole_frequency_sequence,
            waveform_arguments=dict(
                reference_frequency=self.fmin, waveform_approximant=waveform_approximant
            )
        )
        wfg_mb = BackendWaveformGenerator(wfg_mb, self.xp)
        likelihood_mb = bilby.gw.likelihood.MBGravitationalWaveTransient(
            interferometers=self.ifos, waveform_generator=wfg_mb,
            reference_chirp_mass=self.test_parameters['chirp_mass'],
            linear_interpolation=linear_interpolation,
        )
        llr = likelihood_mb.log_likelihood_ratio(self.test_parameters)
        self.assertEqual(aac.get_namespace(llr), self.xp)

        # reset waveform generator to check if likelihood recovered from the weights properly adds banded
        # frequency points to waveform arguments
        wfg_mb = bilby.gw.WaveformGenerator(
            duration=self.duration, sampling_frequency=self.sampling_frequency,
            frequency_domain_source_model=bilby.gw.source.binary_black_hole_frequency_sequence,
            waveform_arguments=dict(
                reference_frequency=self.fmin, waveform_approximant=waveform_approximant
            )
        )
        wfg_mb = BackendWaveformGenerator(wfg_mb, self.xp)
        weights = likelihood_mb.weights
        likelihood_mb_from_weights = bilby.gw.likelihood.MBGravitationalWaveTransient(
            interferometers=self.ifos, waveform_generator=wfg_mb, weights=weights
        )
        llr_from_weights = likelihood_mb_from_weights.log_likelihood_ratio(self.test_parameters)
        self.assertEqual(aac.get_namespace(llr_from_weights), self.xp)

        self.assertAlmostEqual(llr, llr_from_weights)

    @parameterized.expand([
        ("IMRPhenomD", True, 2, False, 1e-2),
        ("IMRPhenomD", True, 2, True, 1e-2),
        ("IMRPhenomHM", False, 4, False, 5e-3),
    ])
    def test_matches_original_likelihood_low_maximum_frequency(
        self, waveform_approximant, linear_interpolation, highest_mode, add_cal_errors, tolerance
    ):
        """
        Test for maximum frequency < sampling frequency / 2
        """
        for ifo in self.ifos:
            ifo.maximum_frequency = self.sampling_frequency / 8

        wfg = bilby.gw.WaveformGenerator(
            duration=self.duration, sampling_frequency=self.sampling_frequency,
            frequency_domain_source_model=bilby.gw.source.lal_binary_black_hole,
            waveform_arguments=dict(
                reference_frequency=self.fmin, waveform_approximant=waveform_approximant
            )
        )
        wfg = BackendWaveformGenerator(wfg, self.xp)
        self.ifos.inject_signal(parameters=self.test_parameters, waveform_generator=wfg)

        wfg_mb = bilby.gw.WaveformGenerator(
            duration=self.duration, sampling_frequency=self.sampling_frequency,
            frequency_domain_source_model=bilby.gw.source.binary_black_hole_frequency_sequence,
            waveform_arguments=dict(
                reference_frequency=self.fmin, waveform_approximant=waveform_approximant
            )
        )
        wfg_mb = BackendWaveformGenerator(wfg_mb, self.xp)
        likelihood = bilby.gw.likelihood.GravitationalWaveTransient(
            interferometers=self.ifos, waveform_generator=wfg
        )
        likelihood_mb = bilby.gw.likelihood.MBGravitationalWaveTransient(
            interferometers=self.ifos, waveform_generator=wfg_mb,
            reference_chirp_mass=self.test_parameters['chirp_mass'],
            priors=self.priors.copy(), linear_interpolation=linear_interpolation,
            highest_mode=highest_mode
        )
        parameters = deepcopy(self.test_parameters)
        if add_cal_errors:
            parameters.update(self.calibration_parameters)
        llrmb = likelihood_mb.log_likelihood_ratio(parameters)
        self.assertLess(
            abs(likelihood.log_likelihood_ratio(parameters) - llrmb),
            tolerance
        )
        self.assertEqual(aac.get_namespace(llrmb), self.xp)

