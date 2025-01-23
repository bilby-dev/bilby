import unittest
from copy import deepcopy

import bilby
import numpy as np
from parameterized import parameterized


class TestRelativeBinningLikelihood(unittest.TestCase):
    def setUp(self):
        duration = 16
        fmin = 20
        sampling_frequency = 8192
        chirp_mass = 13
        mass_ratio = 0.5
        mass_1, mass_2 = bilby.gw.conversion.chirp_mass_and_mass_ratio_to_component_masses(
            chirp_mass=chirp_mass, mass_ratio=mass_ratio
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
        )
        self.fiducial_parameters = self.test_parameters.copy()
        del self.fiducial_parameters["chirp_mass"], self.fiducial_parameters["mass_ratio"]
        self.fiducial_parameters["mass_1"] = mass_1
        self.fiducial_parameters["mass_2"] = mass_2

        ifos = bilby.gw.detector.InterferometerList(["H1", "L1", "V1"])
        ifos.set_strain_data_from_power_spectral_densities(
            sampling_frequency=sampling_frequency, duration=duration,
            start_time=self.test_parameters['geocent_time'] - duration + 2.
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
                n_points=spline_calibration_nodes
            )
            for i in range(spline_calibration_nodes):
                self.test_parameters[f"recalib_{ifo.name}_amplitude_{i}"] = 0
                self.test_parameters[f"recalib_{ifo.name}_phase_{i}"] = 0
                # Calibration errors of 5% in amplitude and 5 degrees in phase
                self.calibration_parameters[f"recalib_{ifo.name}_amplitude_{i}"] = \
                    np.random.normal(loc=0, scale=0.05)
                self.calibration_parameters[f"recalib_{ifo.name}_phase_{i}"] = \
                    np.random.normal(loc=0, scale=5 * np.pi / 180)

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
            duration=duration, sampling_frequency=sampling_frequency,
            frequency_domain_source_model=bilby.gw.source.lal_binary_black_hole,
            waveform_arguments=dict(
                reference_frequency=fmin, minimum_frequency=fmin, waveform_approximant=approximant)
        )
        bin_wfg = bilby.gw.waveform_generator.WaveformGenerator(
            duration=duration, sampling_frequency=sampling_frequency,
            frequency_domain_source_model=bilby.gw.source.lal_binary_black_hole_relative_binning,
            waveform_arguments=dict(
                reference_frequency=fmin, waveform_approximant=approximant, minimum_frequency=fmin)
        )
        ifos.inject_signal(
            parameters=self.test_parameters,
            waveform_generator=non_bin_wfg,
            raise_error=False,
        )
        self.ifos = ifos

        self.non_bin = bilby.gw.likelihood.GravitationalWaveTransient(
            interferometers=ifos, waveform_generator=deepcopy(non_bin_wfg),
            priors=priors.copy()
        )
        self.binned = bilby.gw.likelihood.RelativeBinningGravitationalWaveTransient(
            interferometers=ifos, waveform_generator=deepcopy(bin_wfg),
            fiducial_parameters=self.fiducial_parameters,
            priors=priors.copy(),
            epsilon=0.05,
        )
        self.non_bin.parameters.update(self.test_parameters)
        self.reference_ln_l = self.non_bin.log_likelihood_ratio()
        self.bin_wfg = bin_wfg
        self.priors = priors

    def tearDown(self):
        del (
            self.non_bin,
            self.binned,
        )

    def test_matches_non_binned_many(self):
        for _ in range(100):
            parameters = self.priors.sample()
            self.non_bin.parameters.update(parameters)
            self.binned.parameters.update(parameters)
            regular_ln_l = self.non_bin.log_likelihood_ratio()
            binned_ln_l = self.binned.log_likelihood_ratio()
            self.assertLess(
                abs(regular_ln_l - binned_ln_l)
                / abs(self.reference_ln_l - regular_ln_l),
                0.1
            )

    @parameterized.expand([(False, ), (True, )])
    def test_matches_non_binned(self, add_cal_errors):
        self.non_bin.parameters.update(self.test_parameters)
        self.binned.parameters.update(self.test_parameters)
        if add_cal_errors:
            self.non_bin.parameters.update(self.calibration_parameters)
            self.binned.parameters.update(self.calibration_parameters)
        regular_ln_l = self.non_bin.log_likelihood_ratio()
        binned_ln_l = self.binned.log_likelihood_ratio()
        self.assertLess(abs(regular_ln_l - binned_ln_l), 1e-3)

    def test_optimization_gives_good_match(self):
        fiducial_parameters = self.test_parameters.copy()
        fiducial_parameters["chirp_mass"] *= 0.99
        priors = self.priors.copy()
        for key in [
            "ra", "dec", "geocent_time", "phase", "psi", "theta_jn", "luminosity_distance",
            "a_1", "a_2", "tilt_1", "tilt_2", "phi_12", "phi_jl",
        ]:
            priors[key] = self.test_parameters[key]
        binned = bilby.gw.likelihood.RelativeBinningGravitationalWaveTransient(
            interferometers=self.ifos, waveform_generator=deepcopy(self.bin_wfg),
            priors=priors,
            fiducial_parameters=fiducial_parameters,
            epsilon=0.05,
            update_fiducial_parameters=True,
        )
        self.non_bin.parameters.update(self.test_parameters)
        binned.parameters.update(self.test_parameters)
        regular_ln_l = self.non_bin.log_likelihood_ratio()
        binned_ln_l = binned.log_likelihood_ratio()
        self.assertLess(abs(regular_ln_l - binned_ln_l), 1e-3)

    def test_very_small_epsilon_returns_good_value(self):
        """
        If the frequency bins cover less than one bin, the likeilhood is nan,
        test that we avoid this.
        """
        binned = bilby.gw.likelihood.RelativeBinningGravitationalWaveTransient(
            interferometers=self.ifos, waveform_generator=deepcopy(self.bin_wfg),
            fiducial_parameters=self.fiducial_parameters,
            priors=self.priors.copy(),
            epsilon=0.001,
        )
        binned.parameters.update(self.test_parameters)
        self.assertFalse(np.isnan(binned.log_likelihood_ratio()))

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
        )

        fiducial_parameters = test_parameters.copy()

        ifos = bilby.gw.detector.InterferometerList(["H1", "L1", "V1"])
        ifos.set_strain_data_from_zero_noise(
            sampling_frequency=sampling_frequency,
            duration=duration,
            start_time=test_parameters['geocent_time'] - duration + 2.
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
                waveform_approximant=approximant
            )
        )
        bin_wfg = bilby.gw.waveform_generator.WaveformGenerator(
            duration=duration,
            sampling_frequency=sampling_frequency,
            frequency_domain_source_model=bilby.gw.source.lal_binary_black_hole_relative_binning,
            waveform_arguments=dict(
                reference_frequency=fmin,
                waveform_approximant=approximant,
                minimum_frequency=fmin
            )
        )
        ifos.inject_signal(
            parameters=test_parameters,
            waveform_generator=non_bin_wfg,
            raise_error=False,
        )

        non_bin = bilby.gw.likelihood.GravitationalWaveTransient(
            interferometers=ifos,
            waveform_generator=deepcopy(non_bin_wfg),
            priors=priors.copy()
        )
        binned = bilby.gw.likelihood.RelativeBinningGravitationalWaveTransient(
            interferometers=ifos,
            waveform_generator=deepcopy(bin_wfg),
            fiducial_parameters=fiducial_parameters,
            priors=priors.copy(),
            epsilon=0.05,
        )

        non_bin.parameters.update(test_parameters)
        binned.parameters.update(test_parameters)

        regular_ln_l = non_bin.log_likelihood_ratio()
        binned_ln_l = binned.log_likelihood_ratio()
        self.assertLess(abs(regular_ln_l - binned_ln_l), 1e-3)


if __name__ == "__main__":
    unittest.main()
