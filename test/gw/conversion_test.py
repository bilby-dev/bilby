import unittest
from copy import deepcopy

import numpy as np
import pandas as pd

import bilby
from bilby.gw import conversion


class TestBasicConversions(unittest.TestCase):
    def setUp(self):
        self.mass_1 = 1.4
        self.mass_2 = 1.3
        self.mass_ratio = 13 / 14
        self.total_mass = 2.7
        self.chirp_mass = (1.4 * 1.3) ** 0.6 / 2.7 ** 0.2
        self.symmetric_mass_ratio = (1.4 * 1.3) / 2.7 ** 2
        self.cos_angle = -1
        self.angle = np.pi
        self.lambda_1 = 300
        self.lambda_2 = 300 * (14 / 13) ** 5
        self.lambda_tilde = (
            8
            / 13
            * (
                (
                    1
                    + 7 * self.symmetric_mass_ratio
                    - 31 * self.symmetric_mass_ratio ** 2
                )
                * (self.lambda_1 + self.lambda_2)
                + (1 - 4 * self.symmetric_mass_ratio) ** 0.5
                * (
                    1
                    + 9 * self.symmetric_mass_ratio
                    - 11 * self.symmetric_mass_ratio ** 2
                )
                * (self.lambda_1 - self.lambda_2)
            )
        )
        self.delta_lambda_tilde = (
            1
            / 2
            * (
                (1 - 4 * self.symmetric_mass_ratio) ** 0.5
                * (
                    1
                    - 13272 / 1319 * self.symmetric_mass_ratio
                    + 8944 / 1319 * self.symmetric_mass_ratio ** 2
                )
                * (self.lambda_1 + self.lambda_2)
                + (
                    1
                    - 15910 / 1319 * self.symmetric_mass_ratio
                    + 32850 / 1319 * self.symmetric_mass_ratio ** 2
                    + 3380 / 1319 * self.symmetric_mass_ratio ** 3
                )
                * (self.lambda_1 - self.lambda_2)
            )
        )

    def tearDown(self):
        del self.mass_1
        del self.mass_2
        del self.mass_ratio
        del self.total_mass
        del self.chirp_mass
        del self.symmetric_mass_ratio

    def test_total_mass_and_mass_ratio_to_component_masses(self):
        mass_1, mass_2 = conversion.total_mass_and_mass_ratio_to_component_masses(
            self.mass_ratio, self.total_mass
        )
        self.assertTrue(
            all([abs(mass_1 - self.mass_1) < 1e-5, abs(mass_2 - self.mass_2) < 1e-5])
        )

    def test_chirp_mass_and_primary_mass_to_mass_ratio(self):
        mass_ratio = conversion.chirp_mass_and_primary_mass_to_mass_ratio(
            self.chirp_mass, self.mass_1
        )
        self.assertAlmostEqual(self.mass_ratio, mass_ratio)

    def test_symmetric_mass_ratio_to_mass_ratio(self):
        mass_ratio = conversion.symmetric_mass_ratio_to_mass_ratio(
            self.symmetric_mass_ratio
        )
        self.assertAlmostEqual(self.mass_ratio, mass_ratio)

    def test_chirp_mass_and_total_mass_to_symmetric_mass_ratio(self):
        symmetric_mass_ratio = conversion.chirp_mass_and_total_mass_to_symmetric_mass_ratio(
            self.chirp_mass, self.total_mass
        )
        self.assertAlmostEqual(self.symmetric_mass_ratio, symmetric_mass_ratio)

    def test_chirp_mass_and_mass_ratio_to_total_mass(self):
        total_mass = conversion.chirp_mass_and_mass_ratio_to_total_mass(
            self.chirp_mass, self.mass_ratio
        )
        self.assertAlmostEqual(self.total_mass, total_mass)

    def test_chirp_mass_and_mass_ratio_to_component_masses(self):
        mass_1, mass_2 = \
            conversion.chirp_mass_and_mass_ratio_to_component_masses(
                self.chirp_mass, self.mass_ratio)
        self.assertAlmostEqual(self.mass_1, mass_1)
        self.assertAlmostEqual(self.mass_2, mass_2)

    def test_component_masses_to_chirp_mass(self):
        chirp_mass = conversion.component_masses_to_chirp_mass(self.mass_1, self.mass_2)
        self.assertAlmostEqual(self.chirp_mass, chirp_mass)

    def test_component_masses_to_total_mass(self):
        total_mass = conversion.component_masses_to_total_mass(self.mass_1, self.mass_2)
        self.assertAlmostEqual(self.total_mass, total_mass)

    def test_component_masses_to_symmetric_mass_ratio(self):
        symmetric_mass_ratio = conversion.component_masses_to_symmetric_mass_ratio(
            self.mass_1, self.mass_2
        )
        self.assertAlmostEqual(self.symmetric_mass_ratio, symmetric_mass_ratio)

    def test_component_masses_to_mass_ratio(self):
        mass_ratio = conversion.component_masses_to_mass_ratio(self.mass_1, self.mass_2)
        self.assertAlmostEqual(self.mass_ratio, mass_ratio)

    def test_mass_1_and_chirp_mass_to_mass_ratio(self):
        mass_ratio = conversion.mass_1_and_chirp_mass_to_mass_ratio(
            self.mass_1, self.chirp_mass
        )
        self.assertAlmostEqual(self.mass_ratio, mass_ratio)

    def test_lambda_tilde_to_lambda_1_lambda_2(self):
        lambda_1, lambda_2 = conversion.lambda_tilde_to_lambda_1_lambda_2(
            self.lambda_tilde, self.mass_1, self.mass_2
        )
        self.assertTrue(
            all(
                [
                    abs(self.lambda_1 - lambda_1) < 1e-5,
                    abs(self.lambda_2 - lambda_2) < 1e-5,
                ]
            )
        )

    def test_lambda_tilde_delta_lambda_tilde_to_lambda_1_lambda_2(self):
        (
            lambda_1,
            lambda_2,
        ) = conversion.lambda_tilde_delta_lambda_tilde_to_lambda_1_lambda_2(
            self.lambda_tilde, self.delta_lambda_tilde, self.mass_1, self.mass_2
        )
        self.assertTrue(
            all(
                [
                    abs(self.lambda_1 - lambda_1) < 1e-5,
                    abs(self.lambda_2 - lambda_2) < 1e-5,
                ]
            )
        )

    def test_lambda_1_lambda_2_to_lambda_tilde(self):
        lambda_tilde = conversion.lambda_1_lambda_2_to_lambda_tilde(
            self.lambda_1, self.lambda_2, self.mass_1, self.mass_2
        )
        self.assertTrue((self.lambda_tilde - lambda_tilde) < 1e-5)

    def test_lambda_1_lambda_2_to_delta_lambda_tilde(self):
        delta_lambda_tilde = conversion.lambda_1_lambda_2_to_delta_lambda_tilde(
            self.lambda_1, self.lambda_2, self.mass_1, self.mass_2
        )
        self.assertTrue((self.delta_lambda_tilde - delta_lambda_tilde) < 1e-5)

    def test_identity_conversion(self):
        original_samples = dict(
            mass_1=self.mass_1,
            mass_2=self.mass_2,
            mass_ratio=self.mass_ratio,
            total_mass=self.total_mass,
            chirp_mass=self.chirp_mass,
            symmetric_mass_ratio=self.symmetric_mass_ratio,
            cos_angle=self.cos_angle,
            angle=self.angle,
            lambda_1=self.lambda_1,
            lambda_2=self.lambda_2,
            lambda_tilde=self.lambda_tilde,
            delta_lambda_tilde=self.delta_lambda_tilde
        )
        identity_samples, blank_list = conversion.identity_map_conversion(original_samples)
        assert blank_list == []
        for key, val in identity_samples.items():
            assert val == self.__dict__[key]


class TestConvertToLALParams(unittest.TestCase):
    def setUp(self):
        self.search_keys = []
        self.parameters = dict()
        self.component_mass_pars = dict(mass_1=1.4, mass_2=1.4)
        self.mass_parameters = self.component_mass_pars.copy()
        self.mass_parameters["mass_ratio"] = conversion.component_masses_to_mass_ratio(
            **self.component_mass_pars
        )
        self.mass_parameters[
            "symmetric_mass_ratio"
        ] = conversion.component_masses_to_symmetric_mass_ratio(
            **self.component_mass_pars
        )
        self.mass_parameters["chirp_mass"] = conversion.component_masses_to_chirp_mass(
            **self.component_mass_pars
        )
        self.mass_parameters["total_mass"] = conversion.component_masses_to_total_mass(
            **self.component_mass_pars
        )
        self.component_tidal_parameters = dict(lambda_1=300, lambda_2=300)
        self.all_component_pars = self.component_tidal_parameters.copy()
        self.all_component_pars.update(self.component_mass_pars)
        self.tidal_parameters = self.component_tidal_parameters.copy()
        self.tidal_parameters[
            "lambda_tilde"
        ] = conversion.lambda_1_lambda_2_to_lambda_tilde(**self.all_component_pars)
        self.tidal_parameters[
            "delta_lambda_tilde"
        ] = conversion.lambda_1_lambda_2_to_delta_lambda_tilde(
            **self.all_component_pars
        )

    def tearDown(self):
        del self.search_keys
        del self.parameters

    def bbh_convert(self):
        (
            self.parameters,
            self.added_keys,
        ) = conversion.convert_to_lal_binary_black_hole_parameters(self.parameters)

    def bns_convert(self):
        (
            self.parameters,
            self.added_keys,
        ) = conversion.convert_to_lal_binary_neutron_star_parameters(self.parameters)

    def test_redshift_to_luminosity_distance(self):
        self.parameters["redshift"] = 1
        dl = conversion.redshift_to_luminosity_distance(self.parameters["redshift"])
        self.bbh_convert()
        self.assertEqual(self.parameters["luminosity_distance"], dl)

    def test_comoving_to_luminosity_distance(self):
        self.parameters["comoving_distance"] = 1
        dl = conversion.comoving_distance_to_luminosity_distance(
            self.parameters["comoving_distance"]
        )
        self.bbh_convert()
        self.assertEqual(self.parameters["luminosity_distance"], dl)

    def test_source_to_lab_frame(self):
        self.parameters["test_source"] = 1
        self.parameters["redshift"] = 1
        lab = self.parameters["test_source"] * (1 + self.parameters["redshift"])
        self.bbh_convert()
        self.assertEqual(self.parameters["test"], lab)

    def _conversion_to_component_mass(self, keys):
        for key in keys:
            self.parameters[key] = self.mass_parameters[key]
        self.bbh_convert()
        self.assertAlmostEqual(
            max(
                [
                    abs(self.parameters[key] - self.component_mass_pars[key])
                    for key in ["mass_1", "mass_2"]
                ]
            ),
            0,
        )

    def test_chirp_mass_total_mass(self):
        self._conversion_to_component_mass(["chirp_mass", "total_mass"])

    def test_chirp_mass_sym_mass_ratio(self):
        self._conversion_to_component_mass(["chirp_mass", "symmetric_mass_ratio"])

    def test_chirp_mass_mass_ratio(self):
        self._conversion_to_component_mass(["chirp_mass", "mass_ratio"])

    def test_total_mass_sym_mass_ratio(self):
        self._conversion_to_component_mass(["total_mass", "symmetric_mass_ratio"])

    def test_total_mass_mass_ratio(self):
        self._conversion_to_component_mass(["total_mass", "mass_ratio"])

    def test_total_mass_mass_1(self):
        self._conversion_to_component_mass(["total_mass", "mass_1"])

    def test_total_mass_mass_2(self):
        self._conversion_to_component_mass(["total_mass", "mass_2"])

    def test_sym_mass_ratio_mass_1(self):
        self._conversion_to_component_mass(["symmetric_mass_ratio", "mass_1"])

    def test_sym_mass_ratio_mass_2(self):
        self._conversion_to_component_mass(["symmetric_mass_ratio", "mass_2"])

    def test_mass_ratio_mass_1(self):
        self._conversion_to_component_mass(["mass_ratio", "mass_1"])

    def test_mass_ratio_mass_2(self):
        self._conversion_to_component_mass(["mass_ratio", "mass_2"])

    def test_bbh_aligned_spin_to_spherical(self):
        self.parameters["chi_1"] = -0.5
        a_1 = abs(self.parameters["chi_1"])
        tilt_1 = np.arccos(np.sign(self.parameters["chi_1"]))
        phi_jl = 0.0
        phi_12 = 0.0
        self.bbh_convert()
        self.assertDictEqual(
            {
                key: self.parameters[key]
                for key in ["a_1", "tilt_1", "phi_12", "phi_jl"]
            },
            dict(a_1=a_1, tilt_1=tilt_1, phi_jl=phi_jl, phi_12=phi_12),
        )

    def test_bbh_zero_aligned_spin_to_spherical_with_magnitude(self):
        """
        Test the the conversion returns the correct tilt angles when zero
        aligned spin is passed if the magnitude is also pass.

        If the magnitude is zero this returns tilt = 0.
        If the magnitude is non-zero this returns tilt = pi.
        """
        self.parameters["chi_1"] = 0
        self.parameters["chi_2"] = 0
        a_1 = 0
        self.parameters["a_1"] = a_1
        a_2 = 1
        self.parameters["a_2"] = a_2
        tilt_1 = 0
        tilt_2 = np.pi / 2
        phi_jl = 0
        phi_12 = 0
        self.bbh_convert()
        self.assertDictEqual(
            {
                key: self.parameters[key]
                for key in ["a_1", "a_2", "tilt_1", "tilt_2", "phi_12", "phi_jl"]
            },
            dict(a_1=a_1, a_2=a_2, tilt_1=tilt_1, tilt_2=tilt_2, phi_jl=phi_jl, phi_12=phi_12),
        )

    def test_bbh_zero_aligned_spin_to_spherical_without_magnitude(self):
        """
        Test the the conversion returns the correct tilt angles when zero
        aligned spin is passed if the magnitude is also pass.

        If the magnitude is zero this returns tilt = 0.
        If the magnitude is non-zero this returns tilt = pi.
        """
        self.parameters["chi_1"] = 0
        a_1 = 0
        tilt_1 = np.pi / 2
        phi_jl = 0
        phi_12 = 0
        self.bbh_convert()
        self.assertDictEqual(
            {
                key: self.parameters[key]
                for key in ["a_1", "tilt_1", "phi_12", "phi_jl"]
            },
            dict(a_1=a_1, tilt_1=tilt_1, phi_jl=phi_jl, phi_12=phi_12),
        )

    def test_bbh_cos_angle_to_angle_conversion(self):
        self.parameters["cos_tilt_1"] = 1
        t1 = np.arccos(self.parameters["cos_tilt_1"])
        self.bbh_convert()
        self.assertEqual(self.parameters["tilt_1"], t1)

    def _conversion_to_component_tidal(self, keys):
        for key in keys:
            self.parameters[key] = self.tidal_parameters[key]
        for key in ["mass_1", "mass_2"]:
            self.parameters[key] = self.mass_parameters[key]
        self.bns_convert()
        component_dict = {key: self.parameters[key] for key in ["lambda_1", "lambda_2"]}
        self.assertDictEqual(component_dict, self.component_tidal_parameters)

    def test_lambda_tilde_delta_lambda_tilde(self):
        self._conversion_to_component_tidal(["lambda_tilde", "delta_lambda_tilde"])

    def test_lambda_tilde(self):
        self._conversion_to_component_tidal(["lambda_tilde"])

    def test_lambda_1(self):
        self._conversion_to_component_tidal(["lambda_1"])


class TestGenerateAllParameters(unittest.TestCase):
    def setUp(self):
        self.parameters = dict(
            mass_1=36.0,
            mass_2=29.0,
            a_1=0.4,
            a_2=0.3,
            tilt_1=0.5,
            tilt_2=1.0,
            phi_12=1.7,
            phi_jl=0.3,
            luminosity_distance=2000.0,
            theta_jn=0.4,
            psi=2.659,
            phase=1.3,
            geocent_time=1126259642.413,
            ra=1.375,
            dec=-1.2108,
            lambda_tilde=1000,
            delta_lambda_tilde=0,
        )
        self.expected_bbh_keys = [
            "mass_1",
            "mass_2",
            "a_1",
            "a_2",
            "tilt_1",
            "tilt_2",
            "phi_12",
            "phi_jl",
            "luminosity_distance",
            "theta_jn",
            "psi",
            "phase",
            "geocent_time",
            "ra",
            "dec",
            "reference_frequency",
            "waveform_approximant",
            "minimum_frequency",
            "chirp_mass",
            "total_mass",
            "symmetric_mass_ratio",
            "mass_ratio",
            "iota",
            "spin_1x",
            "spin_1y",
            "spin_1z",
            "spin_2x",
            "spin_2y",
            "spin_2z",
            "phi_1",
            "phi_2",
            "chi_eff",
            "chi_1_in_plane",
            "chi_2_in_plane",
            "chi_p",
            "cos_tilt_1",
            "cos_tilt_2",
            "redshift",
            "comoving_distance",
            "mass_1_source",
            "mass_2_source",
            "chirp_mass_source",
            "total_mass_source",
        ]
        self.expected_tidal_keys = [
            "lambda_1",
            "lambda_2",
            "lambda_tilde",
            "delta_lambda_tilde",
        ]
        self.data_frame = pd.DataFrame({
            key: [value] * 100 for key, value in self.parameters.items()
        })

    def test_generate_all_bbh_parameters(self):
        self._generate(
            bilby.gw.conversion.generate_all_bbh_parameters,
            self.expected_bbh_keys,
        )

    def test_generate_all_bns_parameters(self):
        self._generate(
            bilby.gw.conversion.generate_all_bns_parameters,
            self.expected_bbh_keys + self.expected_tidal_keys,
        )

    def _generate(self, func, expected):
        for values in [self.parameters, self.data_frame]:
            new_parameters = func(values)
            for key in expected:
                self.assertIn(key, new_parameters)

    def test_generate_bbh_parameters_with_likelihood(self):
        priors = bilby.gw.prior.BBHPriorDict()
        priors["geocent_time"] = bilby.core.prior.Uniform(0.4, 0.6)
        ifos = bilby.gw.detector.InterferometerList(["H1", "L1"])
        ifos.set_strain_data_from_power_spectral_densities(duration=1, sampling_frequency=256)
        wfg = bilby.gw.waveform_generator.WaveformGenerator(
            frequency_domain_source_model=bilby.gw.source.lal_binary_black_hole
        )
        likelihood = bilby.gw.likelihood.GravitationalWaveTransient(
            interferometers=ifos,
            waveform_generator=wfg,
            priors=priors,
            phase_marginalization=True,
            time_marginalization=True,
            reference_frame="H1L1",
        )
        self.parameters["zenith"] = 0.0
        self.parameters["azimuth"] = 0.0
        self.parameters["time_jitter"] = 0.0
        del self.parameters["ra"], self.parameters["dec"]
        self.parameters = pd.DataFrame(self.parameters, index=range(1))
        likelihood.parameters["mass_1"] = -1.0
        initial_likelihood_parameters = deepcopy(likelihood.parameters)
        converted = bilby.gw.conversion.generate_all_bbh_parameters(
            sample=self.parameters, likelihood=likelihood, priors=priors
        )
        extra_expected = [
            "geocent_time",
            "phase",
            "H1_optimal_snr",
            "H1_matched_filter_snr",
            "L1_optimal_snr",
            "L1_matched_filter_snr",
            "ra",
            "dec",
        ]
        for key in extra_expected:
            self.assertIn(key, converted)
        # make sure conversion didn't clobber likelihood state
        self.assertEqual(initial_likelihood_parameters, likelihood.parameters)
        self.assertNotEqual(converted["mass_1"].values[0], likelihood.parameters["mass_1"])

    def test_identity_generation_no_likelihood(self):
        test_fixed_prior = bilby.core.prior.PriorDict({
            "test_param_a": bilby.core.prior.DeltaFunction(0, name="test_param_a"),
            "test_param_b": bilby.core.prior.DeltaFunction(1, name="test_param_b")
        }
        )
        output_sample = conversion.identity_map_generation(self.parameters, priors=test_fixed_prior)
        assert output_sample.pop("test_param_a") == 0
        assert output_sample.pop("test_param_b") == 1
        for key, val in self.parameters.items():
            assert output_sample.pop(key) == val
        assert output_sample == {}

    def test_identity_generation_with_likelihood(self):
        priors = bilby.gw.prior.BBHPriorDict()
        priors["geocent_time"] = bilby.core.prior.Uniform(0.4, 0.6)
        self.parameters["time_jitter"] = 0.0
        # Note we do *not* switch to azimuth/zenith, because the identity generation function
        # is not intended to be capable of that conversion
        ifos = bilby.gw.detector.InterferometerList(["H1"])
        ifos.set_strain_data_from_power_spectral_densities(duration=1, sampling_frequency=256)
        wfg = bilby.gw.waveform_generator.WaveformGenerator(
            frequency_domain_source_model=bilby.gw.source.lal_binary_black_hole
        )
        likelihood = bilby.gw.likelihood.GravitationalWaveTransient(
            interferometers=ifos,
            waveform_generator=wfg,
            priors=priors,
            phase_marginalization=True,
            time_marginalization=True,
            reference_frame="sky",
        )
        output_sample = conversion.identity_map_generation(self.parameters, priors=priors, likelihood=likelihood)
        extra_expected = [
            "phase",
            "geocent_time",
            "H1_optimal_snr",
            "H1_matched_filter_snr",
        ]
        for key in extra_expected:
            self.assertIn(key, output_sample)
        for key, val in self.parameters.items():
            self.assertTrue(output_sample[key] == val)


class TestDistanceTransformations(unittest.TestCase):
    def setUp(self):
        self.distances = np.linspace(1, 1000, 100)

    def test_luminosity_redshift_with_cosmology(self):
        z = conversion.luminosity_distance_to_redshift(
            self.distances, cosmology="WMAP9"
        )
        dl = conversion.redshift_to_luminosity_distance(z, cosmology="WMAP9")
        self.assertAlmostEqual(max(abs(dl - self.distances)), 0, 4)

    def test_comoving_redshift_with_cosmology(self):
        z = conversion.comoving_distance_to_redshift(self.distances, cosmology="WMAP9")
        dc = conversion.redshift_to_comoving_distance(z, cosmology="WMAP9")
        self.assertAlmostEqual(max(abs(dc - self.distances)), 0, 4)

    def test_comoving_luminosity_with_cosmology(self):
        dc = conversion.comoving_distance_to_luminosity_distance(
            self.distances, cosmology="WMAP9"
        )
        dl = conversion.luminosity_distance_to_comoving_distance(dc, cosmology="WMAP9")
        self.assertAlmostEqual(max(abs(dl - self.distances)), 0, 4)


class TestGenerateMassParameters(unittest.TestCase):
    def setUp(self):
        self.expected_values = {'mass_1': 2.0,
                                'mass_2': 1.0,
                                'chirp_mass': 1.2167286837864113,
                                'total_mass': 3.0,
                                'mass_1_source': 4.0,
                                'mass_2_source': 2.0,
                                'chirp_mass_source': 2.433457367572823,
                                'total_mass_source': 6,
                                'symmetric_mass_ratio': 0.2222222222222222,
                                'mass_ratio': 0.5}

    def helper_generation_from_keys(self, keys, expected_values, source=False):
        # Explicitly test the helper generate_component_masses
        local_test_vars = \
            {key: expected_values[key] for key in keys}
        local_test_vars_with_component_masses = \
            conversion.generate_component_masses(local_test_vars, source=source)
        if source:
            self.assertTrue("mass_1_source" in local_test_vars_with_component_masses.keys())
            self.assertTrue("mass_2_source" in local_test_vars_with_component_masses.keys())
        else:
            self.assertTrue("mass_1" in local_test_vars_with_component_masses.keys())
            self.assertTrue("mass_2" in local_test_vars_with_component_masses.keys())
        for key in local_test_vars_with_component_masses.keys():
            self.assertAlmostEqual(
                local_test_vars_with_component_masses[key],
                self.expected_values[key])

        # Test the function more generally
        local_all_mass_parameters = \
            conversion.generate_mass_parameters(local_test_vars, source=source)
        if source:
            self.assertEqual(
                set(local_all_mass_parameters.keys()),
                set(["mass_1_source",
                     "mass_2_source",
                     "chirp_mass_source",
                     "total_mass_source",
                     "symmetric_mass_ratio",
                     "mass_ratio",
                     ]
                    )
            )
        else:
            self.assertEqual(
                set(local_all_mass_parameters.keys()),
                set(["mass_1",
                     "mass_2",
                     "chirp_mass",
                     "total_mass",
                     "symmetric_mass_ratio",
                     "mass_ratio",
                     ]
                    )
            )
        for key in local_all_mass_parameters.keys():
            self.assertAlmostEqual(expected_values[key], local_all_mass_parameters[key])

    def test_from_mass_1_and_mass_2(self):
        self.helper_generation_from_keys(["mass_1", "mass_2"],
                                         self.expected_values)

    def test_from_mass_1_and_mass_ratio(self):
        self.helper_generation_from_keys(["mass_1", "mass_ratio"],
                                         self.expected_values)

    def test_from_mass_2_and_mass_ratio(self):
        self.helper_generation_from_keys(["mass_2", "mass_ratio"],
                                         self.expected_values)

    def test_from_mass_1_and_total_mass(self):
        self.helper_generation_from_keys(["mass_2", "total_mass"],
                                         self.expected_values)

    def test_from_chirp_mass_and_mass_ratio(self):
        self.helper_generation_from_keys(["chirp_mass", "mass_ratio"],
                                         self.expected_values)

    def test_from_chirp_mass_and_symmetric_mass_ratio(self):
        self.helper_generation_from_keys(["chirp_mass", "symmetric_mass_ratio"],
                                         self.expected_values)

    def test_from_chirp_mass_and_symmetric_mass_1(self):
        self.helper_generation_from_keys(["chirp_mass", "mass_1"],
                                         self.expected_values)

    def test_from_chirp_mass_and_symmetric_mass_2(self):
        self.helper_generation_from_keys(["chirp_mass", "mass_2"],
                                         self.expected_values)

    def test_from_mass_1_source_and_mass_2_source(self):
        self.helper_generation_from_keys(["mass_1_source", "mass_2_source"],
                                         self.expected_values, source=True)

    def test_from_mass_1_source_and_mass_ratio(self):
        self.helper_generation_from_keys(["mass_1_source", "mass_ratio"],
                                         self.expected_values, source=True)

    def test_from_mass_2_source_and_mass_ratio(self):
        self.helper_generation_from_keys(["mass_2_source", "mass_ratio"],
                                         self.expected_values, source=True)

    def test_from_mass_1_source_and_total_mass(self):
        self.helper_generation_from_keys(["mass_2_source", "total_mass_source"],
                                         self.expected_values, source=True)

    def test_from_chirp_mass_source_and_mass_ratio(self):
        self.helper_generation_from_keys(["chirp_mass_source", "mass_ratio"],
                                         self.expected_values, source=True)

    def test_from_chirp_mass_source_and_symmetric_mass_ratio(self):
        self.helper_generation_from_keys(["chirp_mass_source", "symmetric_mass_ratio"],
                                         self.expected_values, source=True)

    def test_from_chirp_mass_source_and_symmetric_mass_1(self):
        self.helper_generation_from_keys(["chirp_mass_source", "mass_1_source"],
                                         self.expected_values, source=True)

    def test_from_chirp_mass_source_and_symmetric_mass_2(self):
        self.helper_generation_from_keys(["chirp_mass_source", "mass_2_source"],
                                         self.expected_values, source=True)


class TestEquationOfStateConversions(unittest.TestCase):
    '''
    Class to test equation of state conversions.
    The test points were generated from a simulation independent of bilby using the original lalsimulation calls.
    Specific cases tested are described within each function.

    '''
    def setUp(self):
        self.mass_1_source_spectral = [
            4.922542724434885,
            4.350626907771598,
            4.206155335439082,
            1.7822696459661311,
            1.3091740103047926
        ]
        self.mass_2_source_spectral = [
            3.459974694590303,
            1.2276461777181447,
            3.7287707089639976,
            0.3724016563531846,
            1.055042934805801
        ]
        self.spectral_pca_gamma_0 = [
            0.7074873121348357,
            0.05855931126849878,
            0.7795329261793462,
            1.467907561566463,
            2.9066488405635624
        ]
        self.spectral_pca_gamma_1 = [
            -0.29807111670823816,
            2.027708558522935,
            -1.4415775226512115,
            -0.7104870098896858,
            -0.4913817181089619
        ]
        self.spectral_pca_gamma_2 = [
            0.25625095371021156,
            -0.19574096643220049,
            -0.2710238103460012,
            0.22815820981582358,
            -0.1543413205016374
        ]
        self.spectral_pca_gamma_3 = [
            -0.04030365100175101,
            0.05698030777919032,
            -0.045595911403040264,
            -0.023480394227900117,
            -0.07114492992285618
        ]
        self.spectral_gamma_0 = [
            1.1259406796075457,
            0.3191335618787259,
            1.3651245109783452,
            1.3540140238735314,
            1.4551949842961993
        ]
        self.spectral_gamma_1 = [
            0.26791504475282835,
            0.3930374252139248,
            0.11438399886108475,
            0.14181113477953,
            -0.11989033256620368
        ]
        self.spectral_gamma_2 = [
            -0.06810849354463173,
            -0.038250139296677754,
            -0.0801540229444505,
            -0.05230330841791625,
            -0.005197303281460286
        ]
        self.spectral_gamma_3 = [
            0.002848121360389597,
            0.000872447754855139,
            0.005528747386660879,
            0.0024325946344566484,
            0.00043890906202786106
        ]
        self.mass_1_source_polytrope = [
            2.2466565877822573,
            2.869741556013239,
            4.123897187899834,
            2.014160764697004,
            1.414796714032148,
            2.0919349759766614
        ]
        self.mass_2_source_polytrope = [
            0.36696047254774256,
            0.8580637120326807,
            1.650477659961306,
            1.310399737462001,
            0.5470843356210495,
            1.2311162283818198
        ]
        self.polytrope_log10_pressure_1 = [
            34.05849276958394,
            33.06962096113144,
            33.07579629429792,
            33.93412833210738,
            34.24096323517809,
            35.293288373856534
        ]
        self.polytrope_log10_pressure_2 = [
            33.82891829901602,
            35.14230456819543,
            34.940095188881976,
            34.72710820593933,
            35.42780071717415,
            35.648689969687915
        ]
        self.polytrope_gamma_0 = [
            2.359580734009537,
            2.3111471709796945,
            4.784129809424835,
            1.4900432021657437,
            1.0037220431922798,
            4.183994058757201
        ]
        self.polytrope_gamma_1 = [
            1.9497583698697314,
            1.0141111305083874,
            2.8228335336587826,
            4.032519623275465,
            1.10894361284508,
            3.168076721819637
        ]
        self.polytrope_gamma_2 = [
            4.6001755196585385,
            4.424090418206996,
            4.429607300132092,
            1.8176338276795763,
            2.9938859949129797,
            1.300271383168368
        ]
        self.lambda_1_spectral = [0., 0., 0., 0., 1275.7253186286332]
        self.lambda_2_spectral = [0., 0., 0., 0., 4504.897675043909]
        self.lambda_1_polytrope = [0., 0., 0., 0., 0., 234.66424898184766]
        self.lambda_2_polytrope = [0., 0., 0., 0., 0., 3710.931378294547]
        self.eos_check_spectral = [0, 0, 0, 0, 1]
        self.eos_check_polytrope = [0, 0, 0, 0, 0, 1]

    def test_spectral_pca_to_spectral(self):
        for i in range(len(self.mass_1_source_spectral)):
            spectral_gamma_0, spectral_gamma_1, spectral_gamma_2, spectral_gamma_3 = \
                conversion.spectral_pca_to_spectral(
                    self.spectral_pca_gamma_0[i],
                    self.spectral_pca_gamma_1[i],
                    self.spectral_pca_gamma_2[i],
                    self.spectral_pca_gamma_3[i]
                )
            self.assertAlmostEqual(spectral_gamma_0, self.spectral_gamma_0[i], places=5)
            self.assertAlmostEqual(spectral_gamma_1, self.spectral_gamma_1[i], places=5)
            self.assertAlmostEqual(spectral_gamma_2, self.spectral_gamma_2[i], places=5)
            self.assertAlmostEqual(spectral_gamma_3, self.spectral_gamma_3[i], places=5)

    def test_spectral_params_to_lambda_1_lambda_2(self):
        '''
        The points cover 5 test cases:
            - Fail SimNeutronStarEOS4ParamSDGammaCheck()
            - Fail max_speed_of_sound_ <=1.1
            - Fail mass_1_source <= max_mass
            - Fail mass_2_source >= min_mass
            - Passes all and produces accurate lambda_1, lambda_2, eos_check values
        '''
        for i in range(len(self.mass_1_source_spectral)):
            spectral_gamma_0, spectral_gamma_1, spectral_gamma_2, spectral_gamma_3 = \
                conversion.spectral_pca_to_spectral(
                    self.spectral_pca_gamma_0[i],
                    self.spectral_pca_gamma_1[i],
                    self.spectral_pca_gamma_2[i],
                    self.spectral_pca_gamma_3[i]
                )
            lambda_1, lambda_2, eos_check = \
                conversion.spectral_params_to_lambda_1_lambda_2(
                    spectral_gamma_0,
                    spectral_gamma_1,
                    spectral_gamma_2,
                    spectral_gamma_3,
                    self.mass_1_source_spectral[i],
                    self.mass_2_source_spectral[i]
                )
            self.assertAlmostEqual(self.lambda_1_spectral[i], lambda_1, places=0)
            self.assertAlmostEqual(self.lambda_2_spectral[i], lambda_2, places=0)
            self.assertAlmostEqual(self.eos_check_spectral[i], eos_check)

    def test_polytrope_or_causal_params_to_lambda_1_lambda_2_causal(self):
        '''
        The points cover 6 test cases:
            - Fail log10_pressure1 >= log10_pressure2
            - Fail SimNeutronStarEOS3PDViableFamilyCheck()
            - Fail max_speed_of_sound_ <= 1.1
            - Fail mass_1_source <= max_mass
            - Fail mass_2_source >= min_mass
            - Passes all and produces accurate lambda_1, lambda_2, eos_check values
        '''
        for i in range(len(self.mass_1_source_polytrope)):
            lambda_1, lambda_2, eos_check = \
                conversion.polytrope_or_causal_params_to_lambda_1_lambda_2(
                    self.polytrope_gamma_0[i],
                    self.polytrope_log10_pressure_1[i],
                    self.polytrope_gamma_1[i],
                    self.polytrope_log10_pressure_2[i],
                    self.polytrope_gamma_2[i],
                    self.mass_1_source_polytrope[i],
                    self.mass_2_source_polytrope[i],
                    0
                )
            self.assertAlmostEqual(self.lambda_1_polytrope[i], lambda_1, places=2)
            self.assertAlmostEqual(self.lambda_2_polytrope[i], lambda_2, places=1)
            self.assertAlmostEqual(self.eos_check_polytrope[i], eos_check)


if __name__ == "__main__":
    unittest.main()
