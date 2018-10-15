from __future__ import division, absolute_import
import unittest
import bilby
import os
import sys


class TestBBHPriorSet(unittest.TestCase):

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


if __name__ == '__main__':
    unittest.main()
