import unittest
from copy import deepcopy

import bilby
import numpy as np
import pytest
from parameterized import parameterized


class TestRelativeBinningLikelihood(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(2)
        duration = 16
        fmin = 20
        sampling_frequency = 8192
        chirp_mass = 13
        mass_ratio = 0.5
        mass_1, mass_2 = (
            bilby.gw.conversion.chirp_mass_and_mass_ratio_to_component_masses(
                chirp_mass=chirp_mass, mass_ratio=mass_ratio
            )
        )
        self.test_parameters = dict(
            chirp_mass=13,
            mass_ratio=0.5,
            a_1=0.3,
            a_2=0.4,
            tilt_1=1.0,
            tilt_2=0.2,
            phi_12=1.0,
            phi_jl=2.0,
            luminosity_distance=2000.0,
            theta_jn=0.4,
            psi=0.659,
            phase=1.3,
            geocent_time=1187008882,
            ra=1.3,
            dec=-1.2,
            fiducial=0,
        )
        self.fiducial_parameters = self.test_parameters.copy()
        del (
            self.fiducial_parameters["chirp_mass"],
            self.fiducial_parameters["mass_ratio"],
        )
        self.fiducial_parameters["mass_1"] = mass_1
        self.fiducial_parameters["mass_2"] = mass_2

        ifos = bilby.gw.detector.InterferometerList(["H1", "L1", "V1"])
        ifos.set_strain_data_from_power_spectral_densities(
            sampling_frequency=sampling_frequency,
            duration=duration,
            start_time=self.test_parameters["geocent_time"] - duration + 2.0,
            random_state=self.rng,
        )
        for ifo in ifos:
            ifo.minimum_frequency = fmin

        spline_calibration_nodes = 10
        self.calibration_parameters = {}
        for ifo in ifos:
            ifo.calibration_model = bilby.gw.calibration.CubicSpline(
                prefix=f"recalib_{ifo.name}_",
                minimum_frequency=ifo.minimum_frequency,
                maximum_frequency=ifo.maximum_frequency,
                n_points=spline_calibration_nodes,
            )
            for i in range(spline_calibration_nodes):
                self.test_parameters[f"recalib_{ifo.name}_amplitude_{i}"] = 0
                self.test_parameters[f"recalib_{ifo.name}_phase_{i}"] = 0
                # Calibration errors of 5% in amplitude and 5 degrees in phase
                self.calibration_parameters[f"recalib_{ifo.name}_amplitude_{i}"] = (
                    self.rng.normal(loc=0, scale=0.05)
                )
                self.calibration_parameters[f"recalib_{ifo.name}_phase_{i}"] = (
                    self.rng.normal(loc=0, scale=5 * np.pi / 180)
                )

        priors = bilby.gw.prior.BBHPriorDict()
        priors.pop("mass_1")
        priors.pop("mass_2")
        priors["chirp_mass"] = bilby.core.prior.Uniform(12, 14)
        priors["mass_ratio"] = bilby.core.prior.Uniform(0.125, 1)
        priors["geocent_time"] = self.test_parameters["geocent_time"]
        priors["fiducial"] = 0
        self.priors = priors

        approximant = "IMRPhenomXP"
        non_bin_wfg = bilby.gw.WaveformGenerator(
            duration=duration,
            sampling_frequency=sampling_frequency,
            frequency_domain_source_model=bilby.gw.source.lal_binary_black_hole,
            waveform_arguments=dict(
                reference_frequency=fmin,
                minimum_frequency=fmin,
                waveform_approximant=approximant,
            ),
        )
        bin_wfg = bilby.gw.waveform_generator.WaveformGenerator(
            duration=duration,
            sampling_frequency=sampling_frequency,
            frequency_domain_source_model=bilby.gw.source.lal_binary_black_hole_relative_binning,
            waveform_arguments=dict(
                reference_frequency=fmin,
                waveform_approximant=approximant,
                minimum_frequency=fmin,
            ),
        )
        ifos.inject_signal(
            parameters=self.test_parameters,
            waveform_generator=non_bin_wfg,
            raise_error=False,
        )
        self.ifos = ifos

        self.non_bin = bilby.gw.likelihood.GravitationalWaveTransient(
            interferometers=ifos,
            waveform_generator=deepcopy(non_bin_wfg),
            priors=priors.copy(),
        )
        self.binned = bilby.gw.likelihood.RelativeBinningGravitationalWaveTransient(
            interferometers=ifos,
            waveform_generator=deepcopy(bin_wfg),
            fiducial_parameters=self.fiducial_parameters,
            priors=priors.copy(),
            epsilon=0.05,
        )
        self.reference_ln_l = self.non_bin.log_likelihood_ratio(self.test_parameters)
        self.bin_wfg = bin_wfg
        self.priors = priors

    def tearDown(self):
        del (
            self.non_bin,
            self.binned,
        )

    @pytest.mark.flaky(reruns=3, only_rerun=["AssertionError"])
    def test_matches_non_binned_many(self):
        for _ in range(100):
            parameters = self.priors.sample(random_state=self.rng)
            regular_ln_l = self.non_bin.log_likelihood_ratio(parameters)
            binned_ln_l = self.binned.log_likelihood_ratio(parameters)
            self.assertLess(
                abs(regular_ln_l - binned_ln_l)
                / abs(self.reference_ln_l - regular_ln_l),
                0.1,
            )

    @parameterized.expand([(False,), (True,)])
    @pytest.mark.flaky(reruns=3, only_rerun=["AssertionError"])
    def test_matches_non_binned(self, add_cal_errors):
        parameters = deepcopy(self.test_parameters)
        if add_cal_errors:
            parameters.update(self.calibration_parameters)
        regular_ln_l = self.non_bin.log_likelihood_ratio(parameters)
        binned_ln_l = self.binned.log_likelihood_ratio(parameters)
        self.assertLess(abs(regular_ln_l - binned_ln_l), 1e-3)

    @pytest.mark.flaky(reruns=3, only_rerun=["AssertionError"])
    def test_optimization_gives_good_match(self):
        fiducial_parameters = self.test_parameters.copy()
        fiducial_parameters["chirp_mass"] *= 0.99
        priors = self.priors.copy()
        for key in [
            "ra",
            "dec",
            "geocent_time",
            "phase",
            "psi",
            "theta_jn",
            "luminosity_distance",
            "a_1",
            "a_2",
            "tilt_1",
            "tilt_2",
            "phi_12",
            "phi_jl",
        ]:
            priors[key] = self.test_parameters[key]
        binned = bilby.gw.likelihood.RelativeBinningGravitationalWaveTransient(
            interferometers=self.ifos,
            waveform_generator=deepcopy(self.bin_wfg),
            priors=priors,
            fiducial_parameters=fiducial_parameters,
            epsilon=0.05,
            update_fiducial_parameters=True,
        )
        regular_ln_l = self.non_bin.log_likelihood_ratio(self.test_parameters)
        binned_ln_l = binned.log_likelihood_ratio(self.test_parameters)
        self.assertLess(abs(regular_ln_l - binned_ln_l), 1e-3)

    def test_very_small_epsilon_returns_good_value(self):
        """
        If the frequency bins cover less than one bin, the likeilhood is nan,
        test that we avoid this.
        """
        binned = bilby.gw.likelihood.RelativeBinningGravitationalWaveTransient(
            interferometers=self.ifos,
            waveform_generator=deepcopy(self.bin_wfg),
            fiducial_parameters=self.fiducial_parameters,
            priors=self.priors.copy(),
            epsilon=0.001,
        )
        self.assertFalse(np.isnan(binned.log_likelihood_ratio(self.test_parameters)))

    def test_likelihood_when_waveform_extends_beyond_maximum_frequency(self):
        """
        Test that we can setup the relative binning likelihood & that it yields
        accurate results (in zero-noise) for signals that extend in frequency
        beyond the maximum frequency, such as in sub-solar mass mergers.
        """
        duration = 980
        fmin = 20
        sampling_frequency = 8192

        test_parameters = dict(
            chirp_mass=0.435,
            mass_ratio=1.0,
            a_1=0.3,
            a_2=0.4,
            tilt_1=1.0,
            tilt_2=0.2,
            phi_12=1.0,
            phi_jl=2.0,
            luminosity_distance=40,
            theta_jn=0.4,
            psi=0.659,
            phase=1.3,
            geocent_time=1187008882,
            ra=1.3,
            dec=-1.2,
            fiducial=0,
        )

        fiducial_parameters = test_parameters.copy()

        ifos = bilby.gw.detector.InterferometerList(["H1", "L1", "V1"])
        ifos.set_strain_data_from_zero_noise(
            sampling_frequency=sampling_frequency,
            duration=duration,
            start_time=test_parameters["geocent_time"] - duration + 2.0,
        )
        for ifo in ifos:
            ifo.minimum_frequency = fmin

        priors = bilby.gw.prior.BBHPriorDict()
        priors["chirp_mass"] = bilby.core.prior.Uniform(0.434, 0.436)
        priors["mass_ratio"] = bilby.core.prior.Uniform(0.125, 1)
        priors["geocent_time"] = test_parameters["geocent_time"]
        priors["fiducial"] = 0

        approximant = "IMRPhenomXP"
        non_bin_wfg = bilby.gw.WaveformGenerator(
            duration=duration,
            sampling_frequency=sampling_frequency,
            frequency_domain_source_model=bilby.gw.source.lal_binary_black_hole,
            waveform_arguments=dict(
                reference_frequency=fmin,
                minimum_frequency=fmin,
                waveform_approximant=approximant,
            ),
        )
        bin_wfg = bilby.gw.waveform_generator.WaveformGenerator(
            duration=duration,
            sampling_frequency=sampling_frequency,
            frequency_domain_source_model=bilby.gw.source.lal_binary_black_hole_relative_binning,
            waveform_arguments=dict(
                reference_frequency=fmin,
                waveform_approximant=approximant,
                minimum_frequency=fmin,
            ),
        )
        ifos.inject_signal(
            parameters=test_parameters,
            waveform_generator=non_bin_wfg,
            raise_error=False,
        )

        non_bin = bilby.gw.likelihood.GravitationalWaveTransient(
            interferometers=ifos,
            waveform_generator=deepcopy(non_bin_wfg),
            priors=priors.copy(),
        )
        binned = bilby.gw.likelihood.RelativeBinningGravitationalWaveTransient(
            interferometers=ifos,
            waveform_generator=deepcopy(bin_wfg),
            fiducial_parameters=fiducial_parameters,
            priors=priors.copy(),
            epsilon=0.05,
        )

        regular_ln_l = non_bin.log_likelihood_ratio(test_parameters)
        binned_ln_l = binned.log_likelihood_ratio(test_parameters)
        self.assertLess(abs(regular_ln_l - binned_ln_l), 1e-3)


class _InjectionMixin:
    """
    Zero-noise H1/L1 injection for comparing the relative-binning likelihood
    with the full likelihood. Subclasses set binary ("black_hole" or
    "neutron_star"), approximant, injection, minimum_frequency,
    sampling_frequency, epsilon, chirp_mass_scatter and seed.
    """

    duration = 4.0

    def setUp(self):
        self.injection = dict(self.injection)
        self.waveform_kwargs = dict(
            duration=self.duration,
            sampling_frequency=self.sampling_frequency,
            parameter_conversion=getattr(
                bilby.gw.conversion, f"convert_to_lal_binary_{self.binary}_parameters"
            ),
            waveform_arguments=dict(
                waveform_approximant=self.approximant,
                reference_frequency=self.minimum_frequency,
                minimum_frequency=self.minimum_frequency,
            ),
        )
        full_wfg = bilby.gw.WaveformGenerator(
            frequency_domain_source_model=getattr(
                bilby.gw.source, f"lal_binary_{self.binary}"
            ),
            **deepcopy(self.waveform_kwargs),
        )
        self.ifos = bilby.gw.detector.InterferometerList(["H1", "L1"])
        self.ifos.set_strain_data_from_zero_noise(
            sampling_frequency=self.sampling_frequency,
            duration=self.duration,
            start_time=self.injection["geocent_time"] + 2 - self.duration,
        )
        for ifo in self.ifos:
            ifo.minimum_frequency = self.minimum_frequency
        self.ifos.inject_signal(
            parameters=self.injection, waveform_generator=full_wfg, raise_error=False
        )
        self.full = bilby.gw.likelihood.GravitationalWaveTransient(self.ifos, full_wfg)

    def _binned(self, **kwargs):
        kwargs.setdefault("epsilon", self.epsilon)
        wfg = bilby.gw.WaveformGenerator(
            frequency_domain_source_model=getattr(
                bilby.gw.source, f"lal_binary_{self.binary}_relative_binning"
            ),
            **deepcopy(self.waveform_kwargs),
        )
        return bilby.gw.likelihood.RelativeBinningGravitationalWaveTransient(
            self.ifos, wfg, fiducial_parameters=dict(self.injection), **kwargs
        )

    def _max_error(self, binned):
        """Largest |Delta ln L| over 10 points near the injection."""
        rng = np.random.default_rng(self.seed)
        errors = list()
        for _ in range(10):
            parameters = dict(self.injection)
            parameters.update(
                chirp_mass=self.injection["chirp_mass"]
                * (1 + self.chirp_mass_scatter * rng.normal()),
                mass_ratio=self.injection["mass_ratio"] * (1 + 0.03 * rng.normal()),
                phase=self.injection["phase"] + 0.3 * rng.normal(),
                geocent_time=self.injection["geocent_time"] + 2e-4 * rng.normal(),
            )
            errors.append(
                abs(
                    binned.log_likelihood_ratio(parameters)
                    - self.full.log_likelihood_ratio(parameters)
                )
            )
        return max(errors)


class TestRelativeBinningFiducialFade(_InjectionMixin, unittest.TestCase):
    """
    NRTidal waveforms taper to zero at a frequency that depends on the
    parameters. Without the fade, the fiducial waveform is close to zero at
    the last bin edges, so that nearby waveforms have a large ratio to it
    there and the relative-binning likelihood is badly wrong.
    """

    binary = "neutron_star"
    approximant = "IMRPhenomPv2_NRTidalv2"
    injection = dict(
        chirp_mass=bilby.gw.conversion.component_masses_to_chirp_mass(1.6, 1.4),
        mass_ratio=1.4 / 1.6,
        a_1=0.05,
        a_2=0.03,
        tilt_1=0.0,
        tilt_2=np.pi,
        phi_12=0.0,
        phi_jl=0.0,
        lambda_1=400.0,
        lambda_2=450.0,
        luminosity_distance=40.0,
        theta_jn=0.8,
        psi=0.4,
        phase=0.3,
        geocent_time=1126259642.4,
        ra=1.3,
        dec=-0.4,
    )
    minimum_frequency = 100.0
    sampling_frequency = 4096.0
    epsilon = 0.05
    chirp_mass_scatter = 2e-4
    seed = 3

    def test_tapered_fiducial_without_fade_is_inaccurate(self):
        binned = self._binned(fade_fiducial_waveform=False)
        self.assertGreater(self._max_error(binned), 1.0)

    def test_fade_keeps_likelihood_accurate_for_tapered_fiducial(self):
        binned = self._binned(fade_fiducial_waveform=True)
        self.assertLess(self._max_error(binned), 1e-2)

    def test_fade_only_changes_summary_fiducial_near_the_end(self):
        faded = self._binned(fade_fiducial_waveform=True)
        unfaded = self._binned(fade_fiducial_waveform=False)
        frequencies = self.ifos[0].frequency_array
        for ifo in self.ifos:
            name = ifo.name
            np.testing.assert_array_equal(
                faded.per_detector_fiducial_waveforms[name],
                unfaded.per_detector_fiducial_waveforms[name],
            )
            difference = (
                faded._summary_fiducial_waveforms[name]
                != unfaded._summary_fiducial_waveforms[name]
            )
            self.assertTrue(np.any(difference))
            self.assertGreater(frequencies[difference].min(), 1500)


class TestRelativeBinningSplineDegree(_InjectionMixin, unittest.TestCase):
    binary = "black_hole"
    approximant = "IMRPhenomXAS"
    injection = dict(
        chirp_mass=17.0,
        mass_ratio=0.5,
        chi_1=0.3,
        chi_2=-0.2,
        luminosity_distance=800.0,
        theta_jn=0.8,
        psi=0.4,
        phase=0.3,
        geocent_time=1126259642.4,
        ra=1.3,
        dec=-0.4,
    )
    minimum_frequency = 20.0
    sampling_frequency = 2048.0
    epsilon = 0.1
    chirp_mass_scatter = 2e-3
    seed = 5

    @parameterized.expand([(1,), (2,), (3,), (5,)])
    def test_spline_degree_matches_non_binned(self, spline_degree):
        binned = self._binned(spline_degree=spline_degree)
        self.assertLess(self._max_error(binned), 2e-2)

    def test_matches_non_binned_at_fiducial_without_fade(self):
        binned = self._binned(spline_degree=3, fade_fiducial_waveform=False)
        self.assertAlmostEqual(
            binned.log_likelihood_ratio(self.injection),
            self.full.log_likelihood_ratio(self.injection),
            places=6,
        )

    @parameterized.expand([(0,), (6,), (1.5,)])
    def test_invalid_spline_degree_raises(self, spline_degree):
        with self.assertRaises(ValueError):
            self._binned(spline_degree=spline_degree)


if __name__ == "__main__":
    unittest.main()
