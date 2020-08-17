import os
import unittest

import numpy as np

import bilby


class TestPriorDict(unittest.TestCase):
    def setUp(self):
        self.first_prior = bilby.core.prior.Uniform(
            name="a", minimum=0, maximum=1, unit="kg", boundary=None
        )
        self.second_prior = bilby.core.prior.PowerLaw(
            name="b", alpha=3, minimum=1, maximum=2, unit="m/s", boundary=None
        )
        self.third_prior = bilby.core.prior.DeltaFunction(name="c", peak=42, unit="m")
        self.priors = dict(
            mass=self.first_prior, speed=self.second_prior, length=self.third_prior
        )
        self.prior_set_from_dict = bilby.core.prior.PriorDict(dictionary=self.priors)
        self.default_prior_file = os.path.join(
            os.path.dirname(os.path.realpath(__file__)),
            "prior_files/precessing_spins_bbh.prior",
        )
        self.prior_set_from_file = bilby.core.prior.PriorDict(
            filename=self.default_prior_file
        )

    def tearDown(self):
        del self.first_prior
        del self.second_prior
        del self.third_prior
        del self.priors
        del self.prior_set_from_dict
        del self.default_prior_file
        del self.prior_set_from_file

    def test_copy(self):
        priors = bilby.core.prior.PriorDict(self.priors)
        self.assertEqual(priors, priors.copy())

    def test_prior_set(self):
        priors_dict = bilby.core.prior.PriorDict(self.priors)
        priors_set = bilby.core.prior.PriorSet(self.priors)
        self.assertEqual(priors_dict, priors_set)

    def test_prior_set_is_dict(self):
        self.assertIsInstance(self.prior_set_from_dict, dict)

    def test_prior_set_has_correct_length(self):
        self.assertEqual(3, len(self.prior_set_from_dict))

    def test_prior_set_has_expected_priors(self):
        self.assertDictEqual(self.priors, dict(self.prior_set_from_dict))

    def test_read_from_file(self):
        expected = dict(
            mass_1=bilby.core.prior.Constraint(
                name="mass_1",
                minimum=5,
                maximum=100,
            ),
            mass_2=bilby.core.prior.Constraint(
                name="mass_2",
                minimum=5,
                maximum=100,
            ),
            chirp_mass=bilby.core.prior.Uniform(
                name="chirp_mass",
                minimum=25,
                maximum=100,
                latex_label="$\mathcal{M}$",
            ),
            mass_ratio=bilby.core.prior.Uniform(
                name="mass_ratio",
                minimum=0.125,
                maximum=1,
                latex_label="$q$",
                unit=None,
            ),
            a_1=bilby.core.prior.Uniform(
                name="a_1", minimum=0, maximum=0.99
            ),
            a_2=bilby.core.prior.Uniform(
                name="a_2", minimum=0, maximum=0.99
            ),
            tilt_1=bilby.core.prior.Sine(name="tilt_1"),
            tilt_2=bilby.core.prior.Sine(name="tilt_2"),
            phi_12=bilby.core.prior.Uniform(
                name="phi_12", minimum=0, maximum=2 * np.pi, boundary="periodic"
            ),
            phi_jl=bilby.core.prior.Uniform(
                name="phi_jl", minimum=0, maximum=2 * np.pi, boundary="periodic"
            ),
            luminosity_distance=bilby.gw.prior.UniformSourceFrame(
                name="luminosity_distance",
                minimum=1e2,
                maximum=5e3,
                unit="Mpc",
                boundary=None,
            ),
            dec=bilby.core.prior.Cosine(name="dec"),
            ra=bilby.core.prior.Uniform(
                name="ra", minimum=0, maximum=2 * np.pi, boundary="periodic"
            ),
            theta_jn=bilby.core.prior.Sine(name="theta_jn"),
            psi=bilby.core.prior.Uniform(
                name="psi", minimum=0, maximum=np.pi, boundary="periodic"
            ),
            phase=bilby.core.prior.Uniform(
                name="phase", minimum=0, maximum=2 * np.pi, boundary="periodic"
            ),
        )
        self.assertDictEqual(expected, self.prior_set_from_file)

    def test_to_file(self):
        """
        We compare that the strings contain all of the same characters in not
        necessarily the same order as python2 doesn't conserve the order of the
        arguments.
        """
        expected = [
            "length = DeltaFunction(peak=42, name='c', latex_label='c', unit='m')\n",
            "speed = PowerLaw(alpha=3, minimum=1, maximum=2, name='b', latex_label='b', "
            "unit='m/s', boundary=None)\n",
            "mass = Uniform(minimum=0, maximum=1, name='a', latex_label='a', "
            "unit='kg', boundary=None)\n",
        ]
        self.prior_set_from_dict.to_file(outdir="prior_files", label="to_file_test")
        with open("prior_files/to_file_test.prior") as f:
            for i, line in enumerate(f.readlines()):
                self.assertTrue(
                    any([sorted(line) == sorted(expect) for expect in expected])
                )

    def test_from_dict_with_string(self):
        string_prior = (
            "PowerLaw(name='b', alpha=3, minimum=1, maximum=2, unit='m/s', "
            "boundary=None)"
        )
        self.priors["speed"] = string_prior
        from_dict = bilby.core.prior.PriorDict(dictionary=self.priors)
        self.assertDictEqual(self.prior_set_from_dict, from_dict)

    def test_convert_floats_to_delta_functions(self):
        self.prior_set_from_dict["d"] = 5
        self.prior_set_from_dict["e"] = 7.3
        self.prior_set_from_dict["f"] = "unconvertable"
        self.prior_set_from_dict.convert_floats_to_delta_functions()
        expected = dict(
            mass=bilby.core.prior.Uniform(
                name="a", minimum=0, maximum=1, unit="kg", boundary=None
            ),
            speed=bilby.core.prior.PowerLaw(
                name="b", alpha=3, minimum=1, maximum=2, unit="m/s", boundary=None
            ),
            length=bilby.core.prior.DeltaFunction(name="c", peak=42, unit="m"),
            d=bilby.core.prior.DeltaFunction(peak=5),
            e=bilby.core.prior.DeltaFunction(peak=7.3),
            f="unconvertable",
        )
        self.assertDictEqual(expected, self.prior_set_from_dict)

    def test_prior_set_from_dict_but_using_a_string(self):
        prior_set = bilby.core.prior.PriorDict(dictionary=self.default_prior_file)
        expected = bilby.core.prior.PriorDict(
            dict(
                mass_1=bilby.core.prior.Constraint(
                    name="mass_1",
                    minimum=5,
                    maximum=100,
                ),
                mass_2=bilby.core.prior.Constraint(
                    name="mass_2",
                    minimum=5,
                    maximum=100,
                ),
                chirp_mass=bilby.core.prior.Uniform(
                    name="chirp_mass",
                    minimum=25,
                    maximum=100,
                    latex_label="$\mathcal{M}$",
                ),
                mass_ratio=bilby.core.prior.Uniform(
                    name="mass_ratio",
                    minimum=0.125,
                    maximum=1,
                    latex_label="$q$",
                    unit=None,
                ),
                a_1=bilby.core.prior.Uniform(
                    name="a_1", minimum=0, maximum=0.99,
                ),
                a_2=bilby.core.prior.Uniform(
                    name="a_2", minimum=0, maximum=0.99,
                ),
                tilt_1=bilby.core.prior.Sine(name="tilt_1"),
                tilt_2=bilby.core.prior.Sine(name="tilt_2"),
                phi_12=bilby.core.prior.Uniform(
                    name="phi_12", minimum=0, maximum=2 * np.pi, boundary="periodic"
                ),
                phi_jl=bilby.core.prior.Uniform(
                    name="phi_jl", minimum=0, maximum=2 * np.pi, boundary="periodic"
                ),
                luminosity_distance=bilby.gw.prior.UniformSourceFrame(
                    name="luminosity_distance",
                    minimum=1e2,
                    maximum=5e3,
                    unit="Mpc",
                    boundary=None,
                ),
                dec=bilby.core.prior.Cosine(name="dec"),
                ra=bilby.core.prior.Uniform(
                    name="ra", minimum=0, maximum=2 * np.pi, boundary="periodic"
                ),
                theta_jn=bilby.core.prior.Sine(name="theta_jn"),
                psi=bilby.core.prior.Uniform(
                    name="psi", minimum=0, maximum=np.pi, boundary="periodic"
                ),
                phase=bilby.core.prior.Uniform(
                    name="phase", minimum=0, maximum=2 * np.pi, boundary="periodic"
                ),
            )
        )
        all_keys = set(prior_set.keys()).union(set(expected.keys()))
        for key in all_keys:
            self.assertEqual(expected[key], prior_set[key])

    def test_dict_argument_is_not_string_or_dict(self):
        with self.assertRaises(ValueError):
            bilby.core.prior.PriorDict(dictionary=list())

    def test_sample_subset_correct_size(self):
        size = 7
        samples = self.prior_set_from_dict.sample_subset(
            keys=self.prior_set_from_dict.keys(), size=size
        )
        self.assertEqual(len(self.prior_set_from_dict), len(samples))
        for key in samples:
            self.assertEqual(size, len(samples[key]))

    def test_sample_subset_correct_size_when_non_priors_in_dict(self):
        self.prior_set_from_dict["asdf"] = "not_a_prior"
        samples = self.prior_set_from_dict.sample_subset(
            keys=self.prior_set_from_dict.keys()
        )
        self.assertEqual(len(self.prior_set_from_dict) - 1, len(samples))

    def test_sample_subset_with_actual_subset(self):
        size = 3
        samples = self.prior_set_from_dict.sample_subset(keys=["length"], size=size)
        expected = dict(length=np.array([42.0, 42.0, 42.0]))
        self.assertTrue(np.array_equal(expected["length"], samples["length"]))

    def test_sample_subset_constrained_as_array(self):
        size = 3
        keys = ["mass", "speed"]
        out = self.prior_set_from_dict.sample_subset_constrained_as_array(keys, size)
        self.assertTrue(isinstance(out, np.ndarray))
        self.assertTrue(out.shape == (len(keys), size))

    def test_sample(self):
        size = 7
        np.random.seed(42)
        samples1 = self.prior_set_from_dict.sample_subset(
            keys=self.prior_set_from_dict.keys(), size=size
        )
        np.random.seed(42)
        samples2 = self.prior_set_from_dict.sample(size=size)
        self.assertEqual(set(samples1.keys()), set(samples2.keys()))
        for key in samples1:
            self.assertTrue(np.array_equal(samples1[key], samples2[key]))

    def test_prob(self):
        samples = self.prior_set_from_dict.sample_subset(keys=["mass", "speed"])
        expected = self.first_prior.prob(samples["mass"]) * self.second_prior.prob(
            samples["speed"]
        )
        self.assertEqual(expected, self.prior_set_from_dict.prob(samples))

    def test_ln_prob(self):
        samples = self.prior_set_from_dict.sample_subset(keys=["mass", "speed"])
        expected = self.first_prior.ln_prob(
            samples["mass"]
        ) + self.second_prior.ln_prob(samples["speed"])
        self.assertEqual(expected, self.prior_set_from_dict.ln_prob(samples))

    def test_rescale(self):
        theta = [0.5, 0.5, 0.5]
        expected = [
            self.first_prior.rescale(0.5),
            self.second_prior.rescale(0.5),
            self.third_prior.rescale(0.5),
        ]
        self.assertListEqual(
            sorted(expected),
            sorted(
                self.prior_set_from_dict.rescale(
                    keys=self.prior_set_from_dict.keys(), theta=theta
                )
            ),
        )

    def test_redundancy(self):
        for key in self.prior_set_from_dict.keys():
            self.assertFalse(self.prior_set_from_dict.test_redundancy(key=key))


class TestJsonIO(unittest.TestCase):
    def setUp(self):
        mvg = bilby.core.prior.MultivariateGaussianDist(
            names=["testa", "testb"],
            mus=[1, 1],
            covs=np.array([[2.0, 0.5], [0.5, 2.0]]),
            weights=1.0,
        )
        mvn = bilby.core.prior.MultivariateGaussianDist(
            names=["testa", "testb"],
            mus=[1, 1],
            covs=np.array([[2.0, 0.5], [0.5, 2.0]]),
            weights=1.0,
        )
        hp_map_file = os.path.join(
            os.path.dirname(os.path.realpath(__file__)),
            "prior_files/GW150914_testing_skymap.fits",
        )
        hp_dist = bilby.gw.prior.HealPixMapPriorDist(
            hp_map_file, names=["testra", "testdec"]
        )
        hp_3d_dist = bilby.gw.prior.HealPixMapPriorDist(
            hp_map_file, names=["testra", "testdec", "testdistance"], distance=True
        )

        self.priors = bilby.core.prior.PriorDict(
            dict(
                aa=bilby.core.prior.DeltaFunction(name="test", unit="unit", peak=1),
                bb=bilby.core.prior.Gaussian(name="test", unit="unit", mu=0, sigma=1),
                cc=bilby.core.prior.Normal(name="test", unit="unit", mu=0, sigma=1),
                dd=bilby.core.prior.PowerLaw(
                    name="test", unit="unit", alpha=0, minimum=0, maximum=1
                ),
                ee=bilby.core.prior.PowerLaw(
                    name="test", unit="unit", alpha=-1, minimum=0.5, maximum=1
                ),
                ff=bilby.core.prior.PowerLaw(
                    name="test", unit="unit", alpha=2, minimum=1, maximum=1e2
                ),
                gg=bilby.core.prior.Uniform(
                    name="test", unit="unit", minimum=0, maximum=1
                ),
                hh=bilby.core.prior.LogUniform(
                    name="test", unit="unit", minimum=5e0, maximum=1e2
                ),
                ii=bilby.gw.prior.UniformComovingVolume(
                    name="redshift", minimum=0.1, maximum=1.0
                ),
                jj=bilby.gw.prior.UniformSourceFrame(
                    name="luminosity_distance", minimum=1.0, maximum=1000.0
                ),
                kk=bilby.core.prior.Sine(name="test", unit="unit"),
                ll=bilby.core.prior.Cosine(name="test", unit="unit"),
                m=bilby.core.prior.Interped(
                    name="test",
                    unit="unit",
                    xx=np.linspace(0, 10, 1000),
                    yy=np.linspace(0, 10, 1000) ** 4,
                    minimum=3,
                    maximum=5,
                ),
                nn=bilby.core.prior.TruncatedGaussian(
                    name="test", unit="unit", mu=1, sigma=0.4, minimum=-1, maximum=1
                ),
                oo=bilby.core.prior.TruncatedNormal(
                    name="test", unit="unit", mu=1, sigma=0.4, minimum=-1, maximum=1
                ),
                pp=bilby.core.prior.HalfGaussian(name="test", unit="unit", sigma=1),
                qq=bilby.core.prior.HalfNormal(name="test", unit="unit", sigma=1),
                rr=bilby.core.prior.LogGaussian(name="test", unit="unit", mu=0, sigma=1),
                ss=bilby.core.prior.LogNormal(name="test", unit="unit", mu=0, sigma=1),
                tt=bilby.core.prior.Exponential(name="test", unit="unit", mu=1),
                uu=bilby.core.prior.StudentT(
                    name="test", unit="unit", df=3, mu=0, scale=1
                ),
                vv=bilby.core.prior.Beta(name="test", unit="unit", alpha=2.0, beta=2.0),
                xx=bilby.core.prior.Logistic(name="test", unit="unit", mu=0, scale=1),
                yy=bilby.core.prior.Cauchy(name="test", unit="unit", alpha=0, beta=1),
                zz=bilby.core.prior.Lorentzian(
                    name="test", unit="unit", alpha=0, beta=1
                ),
                a_=bilby.core.prior.Gamma(name="test", unit="unit", k=1, theta=1),
                ab=bilby.core.prior.ChiSquared(name="test", unit="unit", nu=2),
                ac=bilby.gw.prior.AlignedSpin(name="test", unit="unit"),
                ad=bilby.core.prior.MultivariateGaussian(
                    dist=mvg, name="testa", unit="unit"
                ),
                ae=bilby.core.prior.MultivariateGaussian(
                    dist=mvg, name="testb", unit="unit"
                ),
                af=bilby.core.prior.MultivariateNormal(
                    dist=mvn, name="testa", unit="unit"
                ),
                ag=bilby.core.prior.MultivariateNormal(
                    dist=mvn, name="testb", unit="unit"
                ),
                ah=bilby.gw.prior.HealPixPrior(
                    dist=hp_dist, name="testra", unit="unit"
                ),
                ai=bilby.gw.prior.HealPixPrior(
                    dist=hp_dist, name="testdec", unit="unit"
                ),
                aj=bilby.gw.prior.HealPixPrior(
                    dist=hp_3d_dist, name="testra", unit="unit"
                ),
                ak=bilby.gw.prior.HealPixPrior(
                    dist=hp_3d_dist, name="testdec", unit="unit"
                ),
                al=bilby.gw.prior.HealPixPrior(
                    dist=hp_3d_dist, name="testdistance", unit="unit"
                ),
            )
        )

    def test_read_write_to_json(self):
        """ Interped prior is removed as there is numerical error in the recovered prior."""
        self.priors.to_json(outdir="prior_files", label="json_test")
        new_priors = bilby.core.prior.PriorDict.from_json(
            filename="prior_files/json_test_prior.json"
        )
        old_interped = self.priors.pop("m")
        new_interped = new_priors.pop("m")
        self.assertDictEqual(self.priors, new_priors)
        self.assertLess(max(abs(old_interped.xx - new_interped.xx)), 1e-15)
        self.assertLess(max(abs(old_interped.yy - new_interped.yy)), 1e-15)


class TestLoadPrior(unittest.TestCase):
    def test_load_prior_with_float(self):
        filename = os.path.join(
            os.path.dirname(os.path.realpath(__file__)),
            "prior_files/prior_with_floats.prior",
        )
        prior = bilby.core.prior.PriorDict(filename)
        self.assertTrue("mass_1" in prior)
        self.assertTrue("mass_2" in prior)
        self.assertTrue(prior["mass_2"].peak == 20)

    def test_load_prior_with_parentheses(self):
        filename = os.path.join(
            os.path.dirname(os.path.realpath(__file__)),
            "prior_files/prior_with_parentheses.prior",
        )
        prior = bilby.core.prior.PriorDict(filename)
        self.assertTrue(isinstance(prior["logA"], bilby.core.prior.Uniform))


if __name__ == "__main__":
    unittest.main()
