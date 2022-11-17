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

        ifos = bilby.gw.detector.InterferometerList(["H1", "L1", "V1"])
        # np.random.seed(170817)
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
        # priors["chirp_mass"] = bilby.core.prior.Uniform(1.29, 1.31)
        priors["chirp_mass"] = bilby.core.prior.Uniform(12, 14)
        priors["mass_ratio"] = bilby.core.prior.Uniform(0.125, 1)
        priors["geocent_time"] = bilby.core.prior.Uniform(
            self.test_parameters['geocent_time'] - 0.1,
            self.test_parameters['geocent_time'] + 0.1)
        priors["fiducial"] = 0
        self.priors = priors

        approximant = "IMRPhenomXP"
        non_bin_wfg = bilby.gw.WaveformGenerator(
            duration=duration, sampling_frequency=sampling_frequency,
            frequency_domain_source_model=bilby.gw.source.lal_binary_black_hole,
            waveform_arguments=dict(
                reference_frequency=fmin, minimum_frequency=fmin, approximant=approximant)
        )
        bin_wfg = bilby.gw.waveform_generator.WaveformGenerator(
            duration=duration, sampling_frequency=sampling_frequency,
            frequency_domain_source_model=bilby.gw.source.lal_binary_black_hole_relative_binning,
            waveform_arguments=dict(
                reference_frequency=fmin, approximant=approximant, minimum_frequency=fmin)
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
            fiducial_parameters=self.test_parameters,
            priors=priors.copy(),
            epsilon=0.05,
            # chi=0.2,
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
            self.non_bin.parameters.update(self.priors.sample())
            self.binned.parameters.update(self.priors.sample())
            regular_ln_l = self.non_bin.log_likelihood_ratio()
            binned_ln_l = self.binned.log_likelihood_ratio()
            if regular_ln_l - self.reference_ln_l < -20:
                continue
            self.assertLess(
                abs(regular_ln_l - binned_ln_l)
                / self.reference_ln_l,
                1.5e-2
            )

    @parameterized.expand([(False, ), (True, )])
    def test_matches_non_binned(self, add_cal_errors):
        self.non_bin.parameters.update(self.test_parameters)
        self.binned.parameters.update(self.test_parameters)
        if add_cal_errors:
            self.non_bin.parameters.update(self.calibration_parameters)
            self.binned.parameters.update(self.calibration_parameters)
        self.assertLess(
            abs(self.non_bin.log_likelihood_ratio() - self.binned.log_likelihood_ratio())
            / self.reference_ln_l,
            1.5e-2
        )

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
        self.assertLess(
            abs(self.non_bin.log_likelihood_ratio() - binned.log_likelihood_ratio())
            / self.reference_ln_l,
            1.5e-2
        )


if __name__ == "__main__":
    unittest.main()
