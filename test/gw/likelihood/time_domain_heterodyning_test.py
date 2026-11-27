import os
import pickle
import shutil
import tempfile
import unittest
from unittest import mock

import dill
import lalsimulation
import numpy as np
import pandas as pd
import pytest

import bilby
from bilby.gw.time_domain_utils import gohberg_semencul_product

GPS_START = 1126259640.0
DURATION = 8.0
SAMPLING_FREQUENCY = 1024.0


@pytest.mark.skipif(
    not hasattr(lalsimulation, "SimIMRPhenomTModelTimesTruncated"),
    reason="lalsimulation without the IMRPhenomT model-time functions",
)
class TestTimeDomainHeterodyningLikelihood(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """A zero-noise IMRPhenomTHM injection, and a heterodyned likelihood with its fiducial at the injection"""
        cls.outdir = tempfile.mkdtemp()
        cls.parameters = cls._parameters()
        ifos = cls._interferometers()
        ifos.inject_signal(
            parameters=cls.parameters,
            domain="time",
            placement="model_time",
            waveform_generator=cls._waveform_generator(
                bilby.gw.source.lal_binary_black_hole_model_time
            ),
        )
        cls.likelihood = (
            bilby.gw.likelihood.TimeDomainHeterodyningGravitationalWaveTransient(
                ifos, cls._waveform_generator(), fiducial_parameters=cls.parameters
            )
        )
        cls.exact = cls.likelihood.exact_likelihood()
        cls.nearby = [
            dict(
                cls.parameters,
                mass_1=30.0 * (1 + dm),
                phase=1.3 + dphi,
                geocent_time=cls.parameters["geocent_time"] + dt,
            )
            for dm, dphi, dt in [
                (0.001, 0.1, 0.0),
                (-0.0005, -0.1, 2e-5),
                (0.002, 0.3, 0.0),
            ]
        ]
        cls.far = [
            dict(
                cls.parameters,
                mass_1=30.0 * (1 + dm),
                phase=1.3 + dphi,
                geocent_time=cls.parameters["geocent_time"] + dt,
            )
            for dm, dphi, dt in [
                (0.005, 0.1, 1e-4),
                (-0.005, -0.2, -2e-4),
                (0.5, 0.0, 0.0),
            ]
        ]

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.outdir)

    @staticmethod
    def _interferometers():
        ifos = bilby.gw.detector.InterferometerList(["H1", "L1"])
        ifos.set_analysis_windows(duration=2, start_time=GPS_START + 1.5)
        ifos.set_strain_data_from_zero_noise(
            sampling_frequency=SAMPLING_FREQUENCY,
            duration=DURATION,
            start_time=GPS_START,
        )
        ifos.set_autocovariance_functions_from_power_spectral_densities()
        return ifos

    @staticmethod
    def _waveform_generator(
        source=bilby.gw.source.lal_binary_black_hole_time_domain_heterodyning,
        approximant="IMRPhenomTHM",
    ):
        return bilby.gw.WaveformGenerator(
            duration=DURATION,
            sampling_frequency=SAMPLING_FREQUENCY,
            start_time=GPS_START,
            time_domain_source_model=source,
            parameter_conversion=bilby.gw.conversion.convert_to_lal_binary_black_hole_parameters,
            waveform_arguments=dict(
                waveform_approximant=approximant,
                reference_frequency=20.0,
                minimum_frequency=10.0,
            ),
        )

    @staticmethod
    def _parameters(mass_2=15.0):
        return dict(
            mass_1=30.0,
            mass_2=mass_2,
            a_1=0.0,
            a_2=0.0,
            tilt_1=0.0,
            tilt_2=0.0,
            phi_12=0.0,
            phi_jl=0.0,
            luminosity_distance=1000.0,
            theta_jn=0.4,
            psi=2.659,
            phase=1.3,
            geocent_time=GPS_START + 3.0,
            ra=1.375,
            dec=-1.2108,
        )

    def _distance_priors(self):
        priors = bilby.gw.prior.BBHPriorDict()
        priors["luminosity_distance"] = bilby.core.prior.PowerLaw(
            alpha=2, minimum=50, maximum=1000, name="luminosity_distance"
        )
        return priors

    def _lookup_table_filename(self):
        return os.path.join(self.outdir, "distance_lookup.npz")

    def _chirp_mass_priors(self):
        chirp_mass = bilby.gw.conversion.component_masses_to_chirp_mass(30.0, 15.0)
        priors = bilby.core.prior.PriorDict(
            {
                key: value
                for key, value in self.parameters.items()
                if key not in ("mass_1", "mass_2")
            }
        )
        priors["chirp_mass"] = bilby.core.prior.Uniform(
            0.99 * chirp_mass, 1.01 * chirp_mass, name="chirp_mass"
        )
        priors["mass_ratio"] = bilby.core.prior.Uniform(0.45, 0.55, name="mass_ratio")
        return priors, chirp_mass

    def _snrs(self, parameters, likelihood=None):
        likelihood = self.likelihood if likelihood is None else likelihood
        parameters = dict(parameters)
        parameters.update(likelihood.get_sky_frame_parameters(parameters))
        signal = likelihood.waveform_polarizations(parameters)
        return (
            {
                ifo.name: likelihood.calculate_snrs(signal, ifo, parameters=parameters)
                for ifo in likelihood.interferometers
            },
            signal,
            parameters,
        )

    def _check_summary_data(self, likelihood, samples):
        """a.v and v^T S v equal <d, s> and <s, s> of the heterodyned template built at every sample"""
        for parameters in samples:
            snrs, signal, parameters = self._snrs(parameters, likelihood)
            for ifo in likelihood.interferometers:
                template = likelihood._compute_full_waveform(signal, ifo, parameters)
                d_inner_s = (
                    likelihood._data_times_inverse_covariance[ifo.name] @ template
                )
                s_inner_s = template @ gohberg_semencul_product(
                    *ifo.gohberg_semencul_vectors, template
                )
                self.assertAlmostEqual(
                    snrs[ifo.name].d_inner_h / d_inner_s, 1, places=8
                )
                self.assertAlmostEqual(
                    snrs[ifo.name].optimal_snr_squared / s_inner_s, 1, places=8
                )

    def _likelihood_with_logs(self, mass_2):
        with self.assertLogs("bilby", level="INFO") as logs:
            likelihood = (
                bilby.gw.likelihood.TimeDomainHeterodyningGravitationalWaveTransient(
                    self._interferometers(),
                    self._waveform_generator(),
                    fiducial_parameters=self._parameters(mass_2),
                )
            )
        return likelihood, "".join(logs.output)

    def test_meta_data(self):
        meta_data = self.likelihood.meta_data
        self.assertEqual(meta_data["likelihood_domain"], "time")
        self.assertEqual(meta_data["placement"], "model_time")
        self.assertEqual(meta_data["mode_keys"], self.likelihood.mode_keys)
        self.assertIsNone(meta_data["parameters_to_be_updated"])
        for ifo in self.likelihood.interferometers:
            self.assertEqual(
                len(meta_data["bin_edge_times"][ifo.name]),
                meta_data["number_of_bins"][ifo.name] + 1,
            )

    def test_summary_data_match_the_heterodyned_template(self):
        self._check_summary_data(self.likelihood, self.nearby + self.far)

    def test_exact_at_the_fiducial_for_aligned_spins(self):
        heterodyned = self.likelihood.log_likelihood_ratio(dict(self.parameters))
        exact = self.exact.log_likelihood_ratio(dict(self.parameters))
        self.assertAlmostEqual(heterodyned, exact, delta=1e-8 * abs(exact))

    def test_accurate_near_the_peak(self):
        for parameters in self.nearby:
            self.assertAlmostEqual(
                self.likelihood.log_likelihood_ratio(dict(parameters)),
                self.exact.log_likelihood_ratio(dict(parameters)),
                delta=0.05,
            )

    def test_bounded_by_the_data(self):
        """<d, s> - <s, s> / 2 <= d^T C^-1 d / 2 in each interferometer, with positive eigenvalues"""
        for parameters in self.nearby + self.far:
            snrs, _, _ = self._snrs(parameters)
            for name, snr in snrs.items():
                self.assertLessEqual(
                    snr.d_inner_h - snr.optimal_snr_squared / 2,
                    self.likelihood._data_inverse_covariance_data[name] / 2,
                )
        for summary in self.likelihood.summary_data.values():
            self.assertTrue(np.all(summary["S_eigenvalues"] > 0))

    def test_distance_scaling(self):
        near, _, _ = self._snrs(self.parameters)
        far, _, _ = self._snrs(
            dict(
                self.parameters,
                luminosity_distance=2 * self.parameters["luminosity_distance"],
            )
        )
        for name in near:
            self.assertAlmostEqual(
                far[name].d_inner_h / near[name].d_inner_h, 1 / 2, places=10
            )
            self.assertAlmostEqual(
                far[name].optimal_snr_squared / near[name].optimal_snr_squared,
                1 / 4,
                places=10,
            )

    def test_per_detector_log_likelihoods_add_up(self):
        parameters = self.nearby[0]
        per_detector = self.likelihood.compute_per_detector_log_likelihood(
            dict(parameters)
        )
        total = sum(
            per_detector[f"{ifo.name}_log_likelihood"]
            for ifo in self.likelihood.interferometers
        )
        self.assertAlmostEqual(
            total, self.likelihood.log_likelihood_ratio(dict(parameters)), places=8
        )

    def test_bin_edges(self):
        times = np.arange(-2048, 300) / 1024
        edges = bilby.gw.likelihood.TimeDomainHeterodyningGravitationalWaveTransient.time_domain_bin_edges(
            times
        )
        merger = np.flatnonzero(times < 0)[-1]
        np.testing.assert_array_equal(edges, np.unique(edges))
        self.assertEqual((edges[0], edges[-1]), (0, len(times) - 1))
        self.assertTrue({merger, merger + 2, merger + 4}.issubset(edges))

    def test_can_be_pickled_after_an_evaluation(self):
        value = self.likelihood.log_likelihood_ratio(dict(self.nearby[0]))
        for module in [pickle, dill]:
            restored = module.loads(module.dumps(self.likelihood))
            self.assertEqual(restored.log_likelihood_ratio(dict(self.nearby[0])), value)

    def test_failed_waveform_gives_minus_infinity(self):
        parameters = dict(self.nearby[0])
        parameters.update(self.likelihood.get_sky_frame_parameters(parameters))
        failed = dict(heterodyning_terms_at=lambda times: None)
        ifo = self.likelihood.interferometers[0]
        self.assertIsNone(
            self.likelihood.calculate_snrs(failed, ifo, parameters=parameters)
        )
        self.assertIsNone(
            self.likelihood._compute_full_waveform(failed, ifo, parameters)
        )
        for signal in [None, failed]:
            with mock.patch.object(
                self.likelihood, "waveform_polarizations", return_value=signal
            ):
                self.assertEqual(
                    self.likelihood.log_likelihood_ratio(dict(parameters)),
                    np.nan_to_num(-np.inf),
                )

    def test_equal_mass_fiducial_gets_all_modes(self):
        likelihood, logs = self._likelihood_with_logs(30.0)
        self.assertEqual(likelihood.mode_keys, [(2, 1), (2, 2), (3, 3), (4, 4), (5, 5)])
        self.assertEqual(likelihood.fiducial_parameters["mass_ratio"], 0.99)
        self.assertIn(
            "The fiducial waveform has no [(2, 1), (3, 3), (5, 5)] modes", logs
        )

    def test_unequal_mass_fiducial_is_unchanged(self):
        likelihood, logs = self._likelihood_with_logs(15.0)
        self.assertEqual(likelihood.mode_keys, [(2, 1), (2, 2), (3, 3), (4, 4), (5, 5)])
        self.assertEqual(likelihood.fiducial_parameters["mass_2"], 15.0)
        self.assertIn(
            "Modes in the heterodyned likelihood: [(2, 1), (2, 2), (3, 3), (4, 4), (5, 5)]",
            logs,
        )

    def test_precessing_waveform_model(self):
        """IMRPhenomTPHM: co-precessing modes with m >= 0 and time-dependent harmonic factors"""
        parameters = dict(
            self.parameters,
            a_1=0.6,
            a_2=0.4,
            tilt_1=1.0,
            tilt_2=2.0,
            phi_12=1.7,
            phi_jl=0.3,
        )
        ifos = self._interferometers()
        ifos.inject_signal(
            parameters=parameters,
            domain="time",
            placement="model_time",
            waveform_generator=self._waveform_generator(
                bilby.gw.source.lal_binary_black_hole_model_time, "IMRPhenomTPHM"
            ),
        )
        with self.assertLogs("bilby", level="INFO") as logs:
            likelihood = (
                bilby.gw.likelihood.TimeDomainHeterodyningGravitationalWaveTransient(
                    ifos,
                    self._waveform_generator(approximant="IMRPhenomTPHM"),
                    fiducial_parameters=parameters,
                )
            )
        self.assertIn(
            "Co-precessing modes in the heterodyned likelihood: [(2, 1), (2, 2), (3, 3), (4, 4), (5, 5)]",
            "".join(logs.output),
        )
        exact = likelihood.exact_likelihood()
        nearby = [
            dict(
                parameters, mass_1=30.0 * (1 + dm), phase=1.3 + dphi, tilt_1=1.0 + dtilt
            )
            for dm, dphi, dtilt in [(0.001, 0.1, 0.0), (-0.0005, -0.1, 0.02)]
        ]
        self._check_summary_data(likelihood, nearby)
        for sample in [parameters] + nearby:
            self.assertAlmostEqual(
                likelihood.log_likelihood_ratio(dict(sample)),
                exact.log_likelihood_ratio(dict(sample)),
                delta=0.05,
            )

    def test_update_fiducial_parameters(self):
        priors, chirp_mass = self._chirp_mass_priors()
        start = dict(priors.sample(), chirp_mass=1.005 * chirp_mass, mass_ratio=0.5)
        likelihood = (
            bilby.gw.likelihood.TimeDomainHeterodyningGravitationalWaveTransient(
                self.likelihood.interferometers,
                self._waveform_generator(),
                fiducial_parameters=start,
                priors=priors,
                update_fiducial_parameters=True,
                maximization_kwargs=dict(maxiter=2, popsize=2, polish=False, seed=1),
            )
        )
        self.assertEqual(
            likelihood.meta_data["parameters_to_be_updated"],
            ["chirp_mass", "mass_ratio"],
        )
        updated = likelihood.fiducial_parameters
        self.assertTrue(
            priors["chirp_mass"].minimum
            <= updated["chirp_mass"]
            <= priors["chirp_mass"].maximum
        )
        self.assertGreaterEqual(
            self.exact.log_likelihood_ratio(dict(updated)),
            self.exact.log_likelihood_ratio(dict(start)),
        )

    def test_fiducial_parameters_drawn_from_the_prior(self):
        priors, _ = self._chirp_mass_priors()
        with self.assertLogs("bilby", level="INFO") as logs:
            likelihood = (
                bilby.gw.likelihood.TimeDomainHeterodyningGravitationalWaveTransient(
                    self.likelihood.interferometers,
                    self._waveform_generator(),
                    priors=priors,
                )
            )
        self.assertIn("Drawing fiducial parameters from prior.", "".join(logs.output))
        for key in ["chirp_mass", "mass_ratio"]:
            self.assertTrue(
                priors[key].minimum
                <= likelihood.fiducial_parameters[key]
                <= priors[key].maximum
            )

    def test_distance_marginalization(self):
        priors = self._distance_priors()
        distance = priors["luminosity_distance"]
        likelihood = (
            bilby.gw.likelihood.TimeDomainHeterodyningGravitationalWaveTransient(
                self.likelihood.interferometers,
                self._waveform_generator(),
                fiducial_parameters=self.parameters,
                priors=priors,
                distance_marginalization=True,
                distance_marginalization_lookup_table=self._lookup_table_filename(),
            )
        )
        exact = likelihood.exact_likelihood()
        for parameters in self.nearby:
            self.assertAlmostEqual(
                likelihood.log_likelihood_ratio(dict(parameters)),
                exact.log_likelihood_ratio(dict(parameters)),
                delta=0.05,
            )
        parameters = dict(self.nearby[0], luminosity_distance=likelihood._ref_dist)
        parameters.update(likelihood.get_sky_frame_parameters(parameters))
        signal = likelihood.waveform_polarizations(dict(parameters))
        new_distance = likelihood.generate_distance_sample_from_marginalized_likelihood(
            signal, parameters=dict(parameters)
        )
        self.assertTrue(distance.minimum <= new_distance <= distance.maximum)
        expected = likelihood.waveform_polarizations(
            dict(parameters, luminosity_distance=new_distance)
        )
        for ifo in likelihood.interferometers:
            rescaled = likelihood.calculate_snrs(signal, ifo, parameters=parameters)
            direct = likelihood.calculate_snrs(expected, ifo, parameters=parameters)
            self.assertAlmostEqual(rescaled.d_inner_h / direct.d_inner_h, 1, places=10)
            self.assertAlmostEqual(
                rescaled.optimal_snr_squared / direct.optimal_snr_squared, 1, places=10
            )

    def test_exact_likelihood_keeps_the_priors(self):
        likelihood = (
            bilby.gw.likelihood.TimeDomainHeterodyningGravitationalWaveTransient(
                self._interferometers(),
                self._waveform_generator(),
                fiducial_parameters=self._parameters(),
                priors=self._distance_priors(),
                distance_marginalization=True,
                distance_marginalization_lookup_table=self._lookup_table_filename(),
            )
        )
        likelihood.exact_likelihood()
        likelihood.exact_likelihood()
        self.assertIsInstance(
            likelihood.priors["luminosity_distance"], bilby.core.prior.PowerLaw
        )

    def test_compare_with_exact_likelihood(self):
        samples = pd.DataFrame(
            [
                dict(parameters, theta_jn=0.4 + 0.01 * i)
                for i, parameters in enumerate(self.nearby)
            ]
        )
        output = self.likelihood.compare_with_exact_likelihood(
            samples, outdir=self.outdir, label="test", n_samples=2
        )
        self.assertEqual(len(output["samples"]), 2)
        np.testing.assert_allclose(output["heterodyned"], output["exact"], atol=0.05)
        for name in [
            "test_heterodyning_vs_exact.png",
            "test_heterodyning_differences.png",
        ]:
            self.assertTrue(os.path.exists(os.path.join(self.outdir, name)))

    def test_wrong_source_model_raises(self):
        with self.assertRaises(ValueError):
            bilby.gw.likelihood.TimeDomainHeterodyningGravitationalWaveTransient(
                self._interferometers(),
                self._waveform_generator(
                    bilby.gw.source.lal_binary_black_hole_model_time
                ),
                fiducial_parameters=self._parameters(),
            )

    def test_no_fiducial_parameters_and_no_priors_raises(self):
        with self.assertRaises(ValueError):
            bilby.gw.likelihood.TimeDomainHeterodyningGravitationalWaveTransient(
                self._interferometers(), self._waveform_generator()
            )

    def test_fiducial_minimum_frequency_above_the_default_raises(self):
        with self.assertRaises(ValueError):
            bilby.gw.likelihood.TimeDomainHeterodyningGravitationalWaveTransient(
                self._interferometers(),
                self._waveform_generator(),
                fiducial_parameters=self._parameters(),
                fiducial_minimum_frequency=100.0,
            )


if __name__ == "__main__":
    unittest.main()
