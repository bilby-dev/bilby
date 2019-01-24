from __future__ import division, absolute_import
import unittest
import os
import sys

import numpy as np
from astropy import cosmology

import bilby


class TestBBHPriorDict(unittest.TestCase):

    def setUp(self):
        self.prior_dict = dict()
        self.base_directory =\
            '/'.join(os.path.dirname(
                os.path.abspath(sys.argv[0])).split('/')[:-1])
        self.filename = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'prior_files/binary_black_holes.prior')
        self.default_prior = bilby.gw.prior.BBHPriorDict(
            filename=self.filename)

    def tearDown(self):
        del self.prior_dict
        del self.filename

    def test_create_default_prior(self):
        default = bilby.gw.prior.BBHPriorDict()
        minima = all([self.default_prior[key].minimum == default[key].minimum
                      for key in default.keys()])
        maxima = all([self.default_prior[key].maximum == default[key].maximum
                      for key in default.keys()])
        names = all([self.default_prior[key].name == default[key].name
                     for key in default.keys()])

        self.assertTrue(all([minima, maxima, names]))

    def test_create_from_dict(self):
        bilby.gw.prior.BBHPriorDict(dictionary=self.prior_dict)

    def test_create_from_filename(self):
        bilby.gw.prior.BBHPriorDict(filename=self.filename)

    def test_key_in_prior_not_redundant(self):
        test = self.default_prior.test_redundancy('mass_1')
        self.assertFalse(test)

    def test_chirp_mass_redundant(self):
        test = self.default_prior.test_redundancy('chirp_mass')
        self.assertTrue(test)

    def test_comoving_distance_redundant(self):
        test = self.default_prior.test_redundancy('comoving_distance')
        self.assertTrue(test)


class TestCalibrationPrior(unittest.TestCase):

    def setUp(self):
        self.minimum_frequency = 20
        self.maximum_frequency = 1024

    def test_create_constant_uncertainty_spline_prior(self):
        "Test that generated spline prior has the correct number of elements."
        amplitude_sigma = 0.1
        phase_sigma = 0.1
        n_nodes = 9
        label = 'test'
        test = bilby.gw.prior.CalibrationPriorDict.constant_uncertainty_spline(
            amplitude_sigma, phase_sigma, self.minimum_frequency,
            self.maximum_frequency, n_nodes, label)

        self.assertEqual(len(test), n_nodes * 3)


class TestUniformComovingVolumePrior(unittest.TestCase):

    def setUp(self):
        pass

    def test_minimum(self):
        prior = bilby.gw.prior.UniformComovingVolume(minimum=10, maximum=10000)
        self.assertEqual(prior.minimum, 10)

    def test_maximum(self):
        prior = bilby.gw.prior.UniformComovingVolume(minimum=10, maximum=10000)
        self.assertEqual(prior.maximum, 10000)

    def test_zero_minimum_works(self):
        prior = bilby.gw.prior.UniformComovingVolume(minimum=0, maximum=10000)
        self.assertEqual(prior.minimum, 0)

    def test_specify_cosmology(self):
        prior = bilby.gw.prior.UniformComovingVolume(
            minimum=10, maximum=10000, cosmology='Planck13')
        self.assertEqual(prior.cosmology, cosmology.Planck13.name)

    def test_redshift_prior_creation(self):
        prior = bilby.gw.prior.UniformComovingVolume(
            minimum=0.1, maximum=1, name='redshift')
        self.assertEqual(prior.latex_label, '$z$')


class TestAlignedSpin(unittest.TestCase):

    def setUp(self):
        pass

    def test_default_prior_matches_analytic(self):
        prior = bilby.gw.prior.AlignedSpin()
        chis = np.linspace(-1, 1, 20)
        analytic = - np.log(np.abs(chis)) / 2
        max_difference = max(abs(analytic - prior.prob(chis)))
        self.assertAlmostEqual(max_difference, 0, 2)

if __name__ == '__main__':
    unittest.main()
