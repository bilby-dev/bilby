import unittest
import numpy as np
import os
import shutil
import peyote
import logging


class Test(unittest.TestCase):
    outdir = 'TestDir'
    dir_path = os.path.dirname(os.path.realpath(__file__))

    # Run a script to produce standard data
    msd = {}  # A dictionary of variables saved in make_standard_data.py
    execfile(dir_path + '/tutorials/make_standard_data.py', msd)

    @classmethod
    def setUpClass(self):
        if os.path.isdir(self.outdir):
            try:
                shutil.rmtree(self.outdir)
            except OSError:
                logging.warning(
                    "{} not removed prior to tests".format(self.outdir))

    @classmethod
    def tearDownClass(self):
        if os.path.isdir(self.outdir):
            try:
                shutil.rmtree(self.outdir)
            except OSError:
                logging.warning(
                    "{} not removed prior to tests".format(self.outdir))

    def test_make_standard_data(self):
        " Load in the saved standard data and compare with new data "

        # Load in the saved standard data
        frequencies_saved, hf_real_saved, hf_imag_saved = np.loadtxt(
            self.dir_path + '/tutorials/standard_data.txt').T
        hf_signal_and_noise_saved = hf_real_saved + 1j * hf_imag_saved

        self.assertTrue(
            all(self.msd['frequencies'] == frequencies_saved))
        self.assertAlmostEqual(all(self.msd['hf_signal_and_noise'] - hf_signal_and_noise_saved), 0.00000000, 5)

    def test_recover_luminosity_distance(self):
        likelihood = peyote.likelihood.Likelihood(
            [self.msd['IFO']], self.msd['source'])

        prior = self.msd['source'].copy()
        dL = self.msd['source'].luminosity_distance
        prior.luminosity_distance = peyote.parameter.Parameter(
            'luminosity_distance',
            prior=peyote.prior.Uniform(lower=dL - 10, upper=dL + 10))

        result = peyote.run_sampler(likelihood, prior, sampler='nestle',
                                    verbose=False)
        self.assertAlmostEqual(np.mean(result.samples), dL,
                               delta=np.std(result.samples))


if __name__ == '__main__':
    unittest.main()


class TestParameterInstantiationWithoutOptionalParameters(unittest.TestCase):

    def setUp(self):
        self.test_name = 'test_name'
        self.parameter = peyote.parameter.Parameter(self.test_name)

    def tearDown(self):
        del self.parameter

    def test_name(self):
        self.assertEqual(self.parameter.name, self.test_name)

    def test_prior(self):
        self.assertIsNone(self.parameter.prior)

    def test_value(self):
        self.assertTrue(np.isnan(self.parameter.value))

    def test_latex_label(self):
        self.assertEqual(self.parameter.latex_label, self.test_name)

    def test_is_fixed(self):
        self.assertFalse(self.parameter.is_fixed)


class TestParameterName(unittest.TestCase):

    def setUp(self):
        self.test_name = 'test_name'
        self.parameter = peyote.parameter.Parameter(self.test_name)

    def tearDown(self):
        del self.parameter

    def test_name_assignment(self):
        self.parameter.name = "other_name"
        self.assertEqual(self.parameter.name, "other_name")


class TestParameterPrior(unittest.TestCase):

    def setUp(self):
        self.test_name = 'test_name'
        self.parameter = peyote.parameter.Parameter(self.test_name)

    def tearDown(self):
        del self.parameter

    def test_prior_assignment(self):
        test_prior = peyote.prior.Uniform(0, 100)
        self.parameter.prior = test_prior
        self.assertDictEqual(test_prior.__dict__, self.parameter.prior.__dict__)

    def test_default_assignment(self):
        test_prior = peyote.prior.PowerLaw(alpha=0, bounds=(5, 100))
        self.parameter.name = 'mchirp'
        self.parameter.prior = None
        self.assertDictEqual(test_prior.__dict__, self.parameter.prior.__dict__)


class TestParameterValue(unittest.TestCase):
    def setUp(self):
        self.test_name = 'test_name'
        self.parameter = peyote.parameter.Parameter(self.test_name)

    def tearDown(self):
        del self.parameter

    def test_prior_assignment(self):
        test_value = 15
        self.parameter.value = test_value
        self.assertEqual(test_value, self.parameter.value)

    def test_default_value_assignment(self):
        self.parameter.name = 'a1'
        self.parameter.value = None
        self.assertEqual(self.parameter.value, 0)

    def test_default_value_assignment_default(self):
        self.parameter.value = None
        self.assertTrue(np.isnan(self.parameter.value))


class TestParameterLatexLabel(unittest.TestCase):
    def setUp(self):
        self.test_name = 'test_name'
        self.parameter = peyote.parameter.Parameter(self.test_name)

    def tearDown(self):
        del self.parameter

    def test_label_assignment(self):
        test_label = 'test_label'
        self.parameter.latex_label = 'test_label'
        self.assertEqual(test_label, self.parameter.latex_label)

    def test_default_label_assignment(self):
        self.parameter.name = 'mchirp'
        self.parameter.latex_label = None
        self.assertEqual(self.parameter.latex_label, '$\mathcal{M}$')

    def test_default_label_assignment_default(self):
        self.assertTrue(self.parameter.latex_label, self.parameter.name)


class TestParameterIsFixed(unittest.TestCase):
    def setUp(self):
        self.test_name = 'test_name'
        self.parameter = peyote.parameter.Parameter(self.test_name)

    def tearDown(self):
        del self.parameter

    def test_is_fixed_assignment(self):
        self.parameter.is_fixed = True
        self.assertTrue(self.parameter.is_fixed)

    def test_default_is_fixed_assignment(self):
        self.assertFalse(self.parameter.is_fixed)