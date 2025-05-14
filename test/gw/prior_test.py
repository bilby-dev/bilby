from collections import OrderedDict
import unittest
import glob
import os
import sys
import pickle

import numpy as np
from astropy import cosmology
from scipy.stats import ks_2samp
import matplotlib.pyplot as plt
import pandas as pd

import bilby
from bilby.core.prior import Uniform, Constraint
from bilby.gw.prior import BBHPriorDict
from bilby.gw import conversion


class TestBBHPriorDict(unittest.TestCase):
    def setUp(self):
        self.prior_dict = dict()
        self.base_directory = "/".join(
            os.path.dirname(os.path.abspath(sys.argv[0])).split("/")[:-1]
        )
        self.filename = os.path.join(
            os.path.dirname(os.path.realpath(__file__)),
            "prior_files/precessing_spins_bbh.prior",
        )
        self.bbh_prior_dict = bilby.gw.prior.BBHPriorDict(filename=self.filename)
        for key, value in self.bbh_prior_dict.items():
            self.prior_dict[key] = value

    def tearDown(self):
        del self.prior_dict
        del self.filename
        del self.bbh_prior_dict
        del self.base_directory

    def test_read_write_default_prior(self):
        filename = "test_prior.prior"
        self.bbh_prior_dict.to_file(outdir=".", label="test_prior")
        new_prior = bilby.gw.prior.BBHPriorDict(filename=filename)
        for key in self.bbh_prior_dict:
            self.assertEqual(self.bbh_prior_dict[key], new_prior[key])
        os.remove(filename)

    def test_create_default_prior(self):
        default = bilby.gw.prior.BBHPriorDict()
        minima = all(
            [
                self.bbh_prior_dict[key].minimum == default[key].minimum
                for key in default.keys()
            ]
        )
        maxima = all(
            [
                self.bbh_prior_dict[key].maximum == default[key].maximum
                for key in default.keys()
            ]
        )
        names = all(
            [
                self.bbh_prior_dict[key].name == default[key].name
                for key in default.keys()
            ]
        )
        boundaries = all(
            [
                self.bbh_prior_dict[key].boundary == default[key].boundary
                for key in default.keys()
            ]
        )

        self.assertTrue(all([minima, maxima, names, boundaries]))

    def test_create_from_dict(self):
        new_dict = bilby.gw.prior.BBHPriorDict(dictionary=self.prior_dict)
        for key in self.bbh_prior_dict:
            self.assertEqual(self.bbh_prior_dict[key], new_dict[key])

    def test_redundant_priors_not_in_dict_before(self):
        for prior in [
            "chirp_mass",
            "total_mass",
            "mass_ratio",
            "symmetric_mass_ratio",
            "cos_tilt_1",
            "cos_tilt_2",
            "phi_1",
            "phi_2",
            "cos_theta_jn",
            "comoving_distance",
            "redshift",
        ]:
            self.assertTrue(self.bbh_prior_dict.test_redundancy(prior))

    def test_redundant_priors_already_in_dict(self):
        for prior in [
            "mass_1",
            "mass_2",
            "tilt_1",
            "tilt_2",
            "phi_1",
            "phi_2",
            "theta_jn",
            "luminosity_distance",
        ]:
            self.assertTrue(self.bbh_prior_dict.test_redundancy(prior))

    def test_correct_not_redundant_priors_masses(self):
        del self.bbh_prior_dict["chirp_mass"]
        for prior in ["chirp_mass", "total_mass", "symmetric_mass_ratio"]:
            self.assertFalse(self.bbh_prior_dict.test_redundancy(prior))

    def test_correct_not_redundant_priors_spin_magnitudes(self):
        del self.bbh_prior_dict["a_2"]
        self.assertFalse(self.bbh_prior_dict.test_redundancy("a_2"))

    def test_correct_not_redundant_priors_spin_tilt_1(self):
        del self.bbh_prior_dict["tilt_1"]
        for prior in ["tilt_1", "cos_tilt_1"]:
            self.assertFalse(self.bbh_prior_dict.test_redundancy(prior))

    def test_correct_not_redundant_priors_spin_tilt_2(self):
        del self.bbh_prior_dict["tilt_2"]
        for prior in ["tilt_2", "cos_tilt_2"]:
            self.assertFalse(self.bbh_prior_dict.test_redundancy(prior))

    def test_correct_not_redundant_priors_spin_azimuth(self):
        del self.bbh_prior_dict["phi_12"]
        for prior in ["phi_1", "phi_2", "phi_12"]:
            self.assertFalse(self.bbh_prior_dict.test_redundancy(prior))

    def test_correct_not_redundant_priors_inclination(self):
        del self.bbh_prior_dict["theta_jn"]
        for prior in ["theta_jn", "cos_theta_jn"]:
            self.assertFalse(self.bbh_prior_dict.test_redundancy(prior))

    def test_correct_not_redundant_priors_distance(self):
        del self.bbh_prior_dict["luminosity_distance"]
        for prior in ["luminosity_distance", "comoving_distance", "redshift"]:
            self.assertFalse(self.bbh_prior_dict.test_redundancy(prior))

    def test_add_unrelated_prior(self):
        self.assertFalse(self.bbh_prior_dict.test_redundancy("abc"))

    def test_test_has_redundant_priors(self):
        self.assertFalse(self.bbh_prior_dict.test_has_redundant_keys())
        for prior in [
            "mass_1",
            "mass_2",
            "total_mass",
            "symmetric_mass_ratio",
            "cos_tilt_1",
            "cos_tilt_2",
            "phi_1",
            "phi_2",
            "cos_theta_jn",
            "comoving_distance",
            "redshift",
        ]:
            self.bbh_prior_dict[prior] = 0
            self.assertTrue(self.bbh_prior_dict.test_has_redundant_keys())
            del self.bbh_prior_dict[prior]

    def test_add_constraint_prior_not_redundant(self):
        self.bbh_prior_dict["chirp_mass"] = bilby.prior.Constraint(
            minimum=20, maximum=40, name="chirp_mass"
        )
        self.assertFalse(self.bbh_prior_dict.test_has_redundant_keys())

    def test_is_cosmological_true(self):
        self.bbh_prior_dict["luminosity_distance"] = bilby.gw.prior.UniformComovingVolume(
            minimum=10, maximum=10000, name="luminosity_distance"
        )
        self.assertTrue(self.bbh_prior_dict.is_cosmological)

    def test_is_cosmological_false(self):
        del self.bbh_prior_dict["luminosity_distance"]
        self.assertFalse(self.bbh_prior_dict.is_cosmological)

    def test_check_valid_cosmology(self):
        self.bbh_prior_dict["luminosity_distance"] = bilby.gw.prior.UniformComovingVolume(
            minimum=10, maximum=10000, name="luminosity_distance"
        )
        self.assertTrue(self.bbh_prior_dict.check_valid_cosmology())

    def test_check_valid_cosmology_raises_error(self):
        self.bbh_prior_dict["luminosity_distance"] = bilby.gw.prior.UniformComovingVolume(
            minimum=10, maximum=10000, name="luminosity_distance", cosmology="Planck15",
        )
        self.bbh_prior_dict["redshift"] = bilby.gw.prior.UniformComovingVolume(
            minimum=0.1, maximum=1, name="redshift", cosmology="Planck15_LAL",
        )
        self.assertEqual(
            self.bbh_prior_dict._cosmological_priors,
            ["luminosity_distance", "redshift"],
        )
        self.assertRaises(ValueError, self.bbh_prior_dict.check_valid_cosmology)

    def test_cosmology(self):
        self.bbh_prior_dict["luminosity_distance"] = bilby.gw.prior.UniformComovingVolume(
            minimum=10, maximum=10000, name="luminosity_distance"
        )
        self.assertEqual(
            self.bbh_prior_dict.cosmology,
            self.bbh_prior_dict["luminosity_distance"].cosmology,
        )

    def test_pickle_prior(self):
        priors = dict(
            chirp_mass=bilby.core.prior.Uniform(10, 20),
            mass_ratio=bilby.core.prior.Uniform(0.125, 1),
        )
        priors = bilby.gw.prior.BBHPriorDict(priors)
        with open("test.pickle", "wb") as file:
            pickle.dump(priors, file)
        with open("test.pickle", "rb") as file:
            priors_loaded = pickle.load(file)
        self.assertEqual(priors, priors_loaded)


class TestPriorConversion(unittest.TestCase):
    def test_bilby_to_lalinference(self):
        mass_1 = [1, 20]
        mass_2 = [1, 20]
        chirp_mass = [1, 5]
        mass_ratio = [0.125, 1]

        bilby_prior = BBHPriorDict(
            dictionary=dict(
                chirp_mass=Uniform(
                    name="chirp_mass", minimum=chirp_mass[0], maximum=chirp_mass[1]
                ),
                mass_ratio=Uniform(
                    name="mass_ratio", minimum=mass_ratio[0], maximum=mass_ratio[1]
                ),
                mass_2=Constraint(name="mass_2", minimum=mass_1[0], maximum=mass_1[1]),
                mass_1=Constraint(name="mass_1", minimum=mass_2[0], maximum=mass_2[1]),
            )
        )

        lalinf_prior = BBHPriorDict(
            dictionary=dict(
                mass_ratio=Constraint(
                    name="mass_ratio", minimum=mass_ratio[0], maximum=mass_ratio[1]
                ),
                chirp_mass=Constraint(
                    name="chirp_mass", minimum=chirp_mass[0], maximum=chirp_mass[1]
                ),
                mass_2=Uniform(name="mass_2", minimum=mass_1[0], maximum=mass_1[1]),
                mass_1=Uniform(name="mass_1", minimum=mass_2[0], maximum=mass_2[1]),
            )
        )

        nsamples = 5000
        bilby_samples = bilby_prior.sample(nsamples)
        bilby_samples, _ = conversion.convert_to_lal_binary_black_hole_parameters(
            bilby_samples
        )

        # Quicker way to generate LA prior samples (rather than specifying Constraint)
        lalinf_samples = []
        while len(lalinf_samples) < nsamples:
            s = lalinf_prior.sample()
            if s["mass_1"] < s["mass_2"]:
                s["mass_1"], s["mass_2"] = s["mass_2"], s["mass_1"]
            if s["mass_2"] / s["mass_1"] > 0.125:
                lalinf_samples.append(s)
        lalinf_samples = pd.DataFrame(lalinf_samples)
        lalinf_samples["mass_ratio"] = (
            lalinf_samples["mass_2"] / lalinf_samples["mass_1"]
        )

        # Construct fake result object
        result = bilby.core.result.Result()
        result.search_parameter_keys = ["mass_ratio", "chirp_mass"]
        result.meta_data = dict()
        result.priors = bilby_prior
        result.posterior = pd.DataFrame(bilby_samples)
        result_converted = bilby.gw.prior.convert_to_flat_in_component_mass_prior(
            result
        )

        if "plot" in sys.argv:
            # Useful for debugging
            plt.hist(bilby_samples["mass_ratio"], bins=50, density=True, alpha=0.5)
            plt.hist(
                result_converted.posterior["mass_ratio"],
                bins=50,
                density=True,
                alpha=0.5,
            )
            plt.hist(lalinf_samples["mass_ratio"], bins=50, alpha=0.5, density=True)
            plt.show()

        # Check that the non-reweighted posteriors fail a KS test
        ks = ks_2samp(bilby_samples["mass_ratio"], lalinf_samples["mass_ratio"])
        print("Non-reweighted KS test = ", ks)
        self.assertFalse(ks.pvalue > 0.05)

        # Check that the non-reweighted posteriors pass a KS test
        ks = ks_2samp(
            result_converted.posterior["mass_ratio"], lalinf_samples["mass_ratio"]
        )
        print("Reweighted KS test = ", ks)
        self.assertTrue(ks.pvalue > 0.001)


class TestPackagedPriors(unittest.TestCase):
    """ Test that the prepackaged priors load """

    def test_aligned(self):
        filename = "aligned_spins_bbh.prior"
        prior_dict = bilby.gw.prior.BBHPriorDict(filename=filename)
        self.assertTrue("chi_1" in prior_dict)
        self.assertTrue("chi_2" in prior_dict)

    def test_binary_black_holes(self):
        filename = "precessing_spins_bbh.prior"
        prior_dict = bilby.gw.prior.BBHPriorDict(filename=filename)
        self.assertTrue("a_1" in prior_dict)

    def test_all(self):
        prior_files = glob.glob(bilby.gw.prior.DEFAULT_PRIOR_DIR + "/*prior")
        for ff in prior_files:
            print("Checking prior file {}".format(ff))
            prior_dict = bilby.gw.prior.BBHPriorDict(filename=ff)
            self.assertTrue("chirp_mass" in prior_dict)
            self.assertTrue("mass_ratio" in prior_dict)
            if "precessing" in ff:
                self.assertTrue("a_1" in prior_dict)
            elif "aligned" in ff:
                self.assertTrue("chi_1" in prior_dict)


class TestBNSPriorDict(unittest.TestCase):
    def setUp(self):
        self.prior_dict = OrderedDict()
        self.base_directory = "/".join(
            os.path.dirname(os.path.abspath(sys.argv[0])).split("/")[:-1]
        )
        self.filename = os.path.join(
            os.path.dirname(os.path.realpath(__file__)),
            "prior_files/aligned_spins_bns_tides_on.prior",
        )
        self.bns_prior_dict = bilby.gw.prior.BNSPriorDict(filename=self.filename)
        for key, value in self.bns_prior_dict.items():
            self.prior_dict[key] = value

    def tearDown(self):
        del self.prior_dict
        del self.filename
        del self.bns_prior_dict
        del self.base_directory

    def test_create_default_prior(self):
        default = bilby.gw.prior.BNSPriorDict()
        minima = all(
            [
                self.bns_prior_dict[key].minimum == default[key].minimum
                for key in default.keys()
            ]
        )
        maxima = all(
            [
                self.bns_prior_dict[key].maximum == default[key].maximum
                for key in default.keys()
            ]
        )
        names = all(
            [
                self.bns_prior_dict[key].name == default[key].name
                for key in default.keys()
            ]
        )
        boundaries = all(
            [
                self.bns_prior_dict[key].boundary == default[key].boundary
                for key in default.keys()
            ]
        )

        self.assertTrue(all([minima, maxima, names, boundaries]))

    def test_create_from_dict(self):
        new_dict = bilby.gw.prior.BNSPriorDict(dictionary=self.prior_dict)
        self.assertDictEqual(self.bns_prior_dict, new_dict)

    def test_redundant_priors_not_in_dict_before(self):
        for prior in [
            "chirp_mass",
            "total_mass",
            "mass_ratio",
            "symmetric_mass_ratio",
            "cos_theta_jn",
            "comoving_distance",
            "redshift",
            "lambda_tilde",
            "delta_lambda_tilde",
        ]:
            self.assertTrue(self.bns_prior_dict.test_redundancy(prior))

    def test_redundant_priors_already_in_dict(self):
        for prior in [
            "mass_1",
            "mass_2",
            "chi_1",
            "chi_2",
            "theta_jn",
            "luminosity_distance",
            "lambda_1",
            "lambda_2",
        ]:
            self.assertTrue(self.bns_prior_dict.test_redundancy(prior))

    def test_correct_not_redundant_priors_masses(self):
        del self.bns_prior_dict["chirp_mass"]
        for prior in ["chirp_mass", "total_mass", "symmetric_mass_ratio"]:
            self.assertFalse(self.bns_prior_dict.test_redundancy(prior))

    def test_correct_not_redundant_priors_spin_magnitudes(self):
        del self.bns_prior_dict["chi_2"]
        self.assertFalse(self.bns_prior_dict.test_redundancy("chi_2"))

    def test_correct_not_redundant_priors_inclination(self):
        del self.bns_prior_dict["theta_jn"]
        for prior in ["theta_jn", "cos_theta_jn"]:
            self.assertFalse(self.bns_prior_dict.test_redundancy(prior))

    def test_correct_not_redundant_priors_distance(self):
        del self.bns_prior_dict["luminosity_distance"]
        for prior in ["luminosity_distance", "comoving_distance", "redshift"]:
            self.assertFalse(self.bns_prior_dict.test_redundancy(prior))

    def test_correct_not_redundant_priors_tidal(self):
        del self.bns_prior_dict["lambda_1"]
        for prior in ["lambda_1", "lambda_tilde", "delta_lambda_tilde"]:
            self.assertFalse(self.bns_prior_dict.test_redundancy(prior))

    def test_add_unrelated_prior(self):
        self.assertFalse(self.bns_prior_dict.test_redundancy("abc"))

    def test_test_has_redundant_priors(self):
        self.assertFalse(self.bns_prior_dict.test_has_redundant_keys())
        for prior in [
            "mass_1",
            "mass_2",
            "total_mass",
            "symmetric_mass_ratio",
            "cos_theta_jn",
            "comoving_distance",
            "redshift",
        ]:
            self.bns_prior_dict[prior] = 0
            self.assertTrue(self.bns_prior_dict.test_has_redundant_keys())
            del self.bns_prior_dict[prior]

    def test_add_constraint_prior_not_redundant(self):
        self.bns_prior_dict["chirp_mass"] = bilby.prior.Constraint(
            minimum=1, maximum=2, name="chirp_mass"
        )
        self.assertFalse(self.bns_prior_dict.test_has_redundant_keys())


class TestCalibrationPrior(unittest.TestCase):
    def setUp(self):
        self.minimum_frequency = 20
        self.maximum_frequency = 1024

    def test_create_constant_uncertainty_spline_prior(self):
        "Test that generated spline prior has the correct number of elements."
        amplitude_sigma = 0.1
        phase_sigma = 0.1
        n_nodes = 9
        label = "test"
        test = bilby.gw.prior.CalibrationPriorDict.constant_uncertainty_spline(
            amplitude_sigma,
            phase_sigma,
            self.minimum_frequency,
            self.maximum_frequency,
            n_nodes,
            label,
        )

        self.assertEqual(len(test), n_nodes * 3)


class TestUniformComovingVolumePrior(unittest.TestCase):
    def setUp(self):
        pass

    def test_minimum(self):
        prior = bilby.gw.prior.UniformComovingVolume(
            minimum=10, maximum=10000, name="luminosity_distance"
        )
        self.assertEqual(prior.minimum, 10)

    def test_maximum(self):
        prior = bilby.gw.prior.UniformComovingVolume(
            minimum=10, maximum=10000, name="luminosity_distance"
        )
        self.assertEqual(prior.maximum, 10000)

    def test_increase_maximum(self):
        prior = bilby.gw.prior.UniformComovingVolume(
            minimum=10, maximum=10000, name="luminosity_distance"
        )
        prior.maximum = 20000
        prior_sample = prior.sample(5000)
        self.assertGreater(np.mean(prior_sample), 10000)

    def test_zero_minimum_works(self):
        prior = bilby.gw.prior.UniformComovingVolume(
            minimum=0, maximum=10000, name="luminosity_distance"
        )
        self.assertEqual(prior.minimum, 0)

    def test_specify_cosmology(self):
        prior = bilby.gw.prior.UniformComovingVolume(
            minimum=10, maximum=10000, name="luminosity_distance", cosmology="Planck13"
        )
        self.assertEqual(repr(prior.cosmology), repr(cosmology.Planck13))

    def test_comoving_prior_creation(self):
        prior = bilby.gw.prior.UniformComovingVolume(
            minimum=10, maximum=1000, name="comoving_distance"
        )
        self.assertEqual(prior.latex_label, "$d_C$")

    def test_redshift_prior_creation(self):
        prior = bilby.gw.prior.UniformComovingVolume(
            minimum=0.1, maximum=1, name="redshift"
        )
        self.assertEqual(prior.latex_label, "$z$")

    def test_redshift_to_luminosity_distance(self):
        prior = bilby.gw.prior.UniformComovingVolume(
            minimum=0.1, maximum=1, name="redshift"
        )
        new_prior = prior.get_corresponding_prior("luminosity_distance")
        self.assertEqual(new_prior.name, "luminosity_distance")

    def test_luminosity_distance_to_redshift(self):
        prior = bilby.gw.prior.UniformComovingVolume(
            minimum=10, maximum=10000, name="luminosity_distance"
        )
        new_prior = prior.get_corresponding_prior("redshift")
        self.assertEqual(new_prior.name, "redshift")

    def test_luminosity_distance_to_comoving_distance(self):
        prior = bilby.gw.prior.UniformComovingVolume(
            minimum=10, maximum=10000, name="luminosity_distance"
        )
        new_prior = prior.get_corresponding_prior("comoving_distance")
        self.assertEqual(new_prior.name, "comoving_distance")


class TestAlignedSpin(unittest.TestCase):
    def setUp(self):
        pass

    def test_default_prior_matches_analytic(self):
        prior = bilby.gw.prior.AlignedSpin()
        chis = np.linspace(-1, 1, 20)
        analytic = -np.log(np.abs(chis)) / 2
        max_difference = max(abs(analytic - prior.prob(chis)))
        self.assertAlmostEqual(max_difference, 0, 2)

    def test_non_analytic_form_has_correct_statistics(self):
        a_prior = bilby.core.prior.TruncatedGaussian(mu=0, sigma=0.1, minimum=0, maximum=1)
        z_prior = bilby.core.prior.TruncatedGaussian(mu=0.4, sigma=0.2, minimum=-1, maximum=1)
        chi_prior = bilby.gw.prior.AlignedSpin(a_prior, z_prior)
        chis = chi_prior.sample(100000)
        alts = a_prior.sample(100000) * z_prior.sample(100000)
        self.assertAlmostEqual(np.mean(chis), np.mean(alts), 2)
        self.assertAlmostEqual(np.std(chis), np.std(alts), 2)


class TestConditionalChiUniformSpinMagnitude(unittest.TestCase):

    def setUp(self):
        pass

    def test_marginalized_prior_is_uniform(self):
        priors = bilby.gw.prior.BBHPriorDict(aligned_spin=True)
        priors["a_1"] = bilby.gw.prior.ConditionalChiUniformSpinMagnitude(
            minimum=0.1, maximum=priors["chi_1"].maximum, name="a_1"
        )
        samples = priors.sample(100000)["a_1"]
        ks = ks_2samp(samples, np.random.uniform(0, priors["chi_1"].maximum, 100000))
        self.assertTrue(ks.pvalue > 0.001)


if __name__ == "__main__":
    unittest.main()
