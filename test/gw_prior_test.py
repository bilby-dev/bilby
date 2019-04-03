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
        self.filename = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                     'prior_files/binary_black_holes.prior')
        self.bbh_prior_dict = bilby.gw.prior.BBHPriorDict(filename=self.filename)
        for key, value in self.bbh_prior_dict.items():
            self.prior_dict[key] = value

    def tearDown(self):
        del self.prior_dict
        del self.filename
        del self.bbh_prior_dict
        del self.base_directory

    def test_create_default_prior(self):
        default = bilby.gw.prior.BBHPriorDict()
        minima = all([self.bbh_prior_dict[key].minimum == default[key].minimum
                      for key in default.keys()])
        maxima = all([self.bbh_prior_dict[key].maximum == default[key].maximum
                      for key in default.keys()])
        names = all([self.bbh_prior_dict[key].name == default[key].name
                     for key in default.keys()])
        boundaries = all([self.bbh_prior_dict[key].periodic_boundary is default[key].periodic_boundary
                          for key in default.keys()])

        self.assertTrue(all([minima, maxima, names, boundaries]))

    def test_create_from_dict(self):
        new_dict = bilby.gw.prior.BBHPriorDict(dictionary=self.prior_dict)
        for key in self.bbh_prior_dict:
            self.assertEqual(self.bbh_prior_dict[key], new_dict[key])

    def test_redundant_priors_not_in_dict_before(self):
        for prior in ['chirp_mass', 'total_mass', 'mass_ratio', 'symmetric_mass_ratio',
                      'cos_tilt_1', 'cos_tilt_2', 'phi_1', 'phi_2', 'cos_theta_jn',
                      'comoving_distance', 'redshift']:
            self.assertTrue(self.bbh_prior_dict.test_redundancy(prior))

    def test_redundant_priors_already_in_dict(self):
        for prior in ['mass_1', 'mass_2', 'tilt_1', 'tilt_2',
                      'phi_1', 'phi_2', 'theta_jn', 'luminosity_distance']:
            self.assertTrue(self.bbh_prior_dict.test_redundancy(prior))

    def test_correct_not_redundant_priors_masses(self):
        del self.bbh_prior_dict['mass_2']
        for prior in ['mass_2', 'chirp_mass', 'total_mass',  'symmetric_mass_ratio']:
            self.assertFalse(self.bbh_prior_dict.test_redundancy(prior))

    def test_correct_not_redundant_priors_spin_magnitudes(self):
        del self.bbh_prior_dict['a_2']
        self.assertFalse(self.bbh_prior_dict.test_redundancy('a_2'))

    def test_correct_not_redundant_priors_spin_tilt_1(self):
        del self.bbh_prior_dict['tilt_1']
        for prior in ['tilt_1', 'cos_tilt_1']:
            self.assertFalse(self.bbh_prior_dict.test_redundancy(prior))

    def test_correct_not_redundant_priors_spin_tilt_2(self):
        del self.bbh_prior_dict['tilt_2']
        for prior in ['tilt_2', 'cos_tilt_2']:
            self.assertFalse(self.bbh_prior_dict.test_redundancy(prior))

    def test_correct_not_redundant_priors_spin_azimuth(self):
        del self.bbh_prior_dict['phi_12']
        for prior in ['phi_1', 'phi_2', 'phi_12']:
            self.assertFalse(self.bbh_prior_dict.test_redundancy(prior))

    def test_correct_not_redundant_priors_inclination(self):
        del self.bbh_prior_dict['theta_jn']
        for prior in ['theta_jn', 'cos_theta_jn']:
            self.assertFalse(self.bbh_prior_dict.test_redundancy(prior))

    def test_correct_not_redundant_priors_distance(self):
        del self.bbh_prior_dict['luminosity_distance']
        for prior in ['luminosity_distance', 'comoving_distance',
                      'redshift']:
            self.assertFalse(self.bbh_prior_dict.test_redundancy(prior))

    def test_add_unrelated_prior(self):
        self.assertFalse(self.bbh_prior_dict.test_redundancy('abc'))

    def test_test_has_redundant_priors(self):
        self.assertFalse(self.bbh_prior_dict.test_has_redundant_keys())
        for prior in ['chirp_mass', 'total_mass', 'mass_ratio', 'symmetric_mass_ratio',
                      'cos_tilt_1', 'cos_tilt_2', 'phi_1', 'phi_2', 'cos_theta_jn',
                      'comoving_distance', 'redshift']:
            self.bbh_prior_dict[prior] = 0
            self.assertTrue(self.bbh_prior_dict.test_has_redundant_keys())
            del self.bbh_prior_dict[prior]

    def test_add_constraint_prior_not_redundant(self):
        self.bbh_prior_dict['chirp_mass'] = bilby.prior.Constraint(
            minimum=20, maximum=40, name='chirp_mass')
        self.assertFalse(self.bbh_prior_dict.test_has_redundant_keys())


class TestBNSPriorDict(unittest.TestCase):

    def setUp(self):
        self.prior_dict = dict()
        self.base_directory =\
            '/'.join(os.path.dirname(
                os.path.abspath(sys.argv[0])).split('/')[:-1])
        self.filename = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                     'prior_files/binary_neutron_stars.prior')
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
        minima = all([self.bns_prior_dict[key].minimum == default[key].minimum
                      for key in default.keys()])
        maxima = all([self.bns_prior_dict[key].maximum == default[key].maximum
                      for key in default.keys()])
        names = all([self.bns_prior_dict[key].name == default[key].name
                     for key in default.keys()])
        boundaries = all([self.bns_prior_dict[key].periodic_boundary == default[key].periodic_boundary
                          for key in default.keys()])

        self.assertTrue(all([minima, maxima, names, boundaries]))

    def test_create_from_dict(self):
        new_dict = bilby.gw.prior.BNSPriorDict(dictionary=self.prior_dict)
        self.assertDictEqual(self.bns_prior_dict, new_dict)

    def test_redundant_priors_not_in_dict_before(self):
        for prior in ['chirp_mass', 'total_mass', 'mass_ratio',
                      'symmetric_mass_ratio', 'cos_theta_jn', 'comoving_distance',
                      'redshift', 'lambda_tilde', 'delta_lambda']:
            self.assertTrue(self.bns_prior_dict.test_redundancy(prior))

    def test_redundant_priors_already_in_dict(self):
        for prior in ['mass_1', 'mass_2', 'chi_1', 'chi_2',
                      'theta_jn', 'luminosity_distance',
                      'lambda_1', 'lambda_2']:
            self.assertTrue(self.bns_prior_dict.test_redundancy(prior))

    def test_correct_not_redundant_priors_masses(self):
        del self.bns_prior_dict['mass_2']
        for prior in ['mass_2', 'chirp_mass', 'total_mass',  'symmetric_mass_ratio']:
            self.assertFalse(self.bns_prior_dict.test_redundancy(prior))

    def test_correct_not_redundant_priors_spin_magnitudes(self):
        del self.bns_prior_dict['chi_2']
        self.assertFalse(self.bns_prior_dict.test_redundancy('chi_2'))

    def test_correct_not_redundant_priors_inclination(self):
        del self.bns_prior_dict['theta_jn']
        for prior in ['theta_jn', 'cos_theta_jn']:
            self.assertFalse(self.bns_prior_dict.test_redundancy(prior))

    def test_correct_not_redundant_priors_distance(self):
        del self.bns_prior_dict['luminosity_distance']
        for prior in ['luminosity_distance', 'comoving_distance',
                      'redshift']:
            self.assertFalse(self.bns_prior_dict.test_redundancy(prior))

    def test_correct_not_redundant_priors_tidal(self):
        del self.bns_prior_dict['lambda_1']
        for prior in['lambda_1', 'lambda_tilde', 'delta_lambda']:
            self.assertFalse(self.bns_prior_dict.test_redundancy(prior))

    def test_add_unrelated_prior(self):
        self.assertFalse(self.bns_prior_dict.test_redundancy('abc'))

    def test_test_has_redundant_priors(self):
        self.assertFalse(self.bns_prior_dict.test_has_redundant_keys())
        for prior in ['chirp_mass', 'total_mass', 'mass_ratio', 'symmetric_mass_ratio',
                      'cos_theta_jn', 'comoving_distance', 'redshift']:
            self.bns_prior_dict[prior] = 0
            self.assertTrue(self.bns_prior_dict.test_has_redundant_keys())
            del self.bns_prior_dict[prior]

    def test_add_constraint_prior_not_redundant(self):
        self.bns_prior_dict['chirp_mass'] = bilby.prior.Constraint(
            minimum=1, maximum=2, name='chirp_mass')
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
        label = 'test'
        test = bilby.gw.prior.CalibrationPriorDict.constant_uncertainty_spline(
            amplitude_sigma, phase_sigma, self.minimum_frequency,
            self.maximum_frequency, n_nodes, label)

        self.assertEqual(len(test), n_nodes * 3)


class TestUniformComovingVolumePrior(unittest.TestCase):

    def setUp(self):
        pass

    def test_minimum(self):
        prior = bilby.gw.prior.UniformComovingVolume(
            minimum=10, maximum=10000, name='luminosity_distance')
        self.assertEqual(prior.minimum, 10)

    def test_maximum(self):
        prior = bilby.gw.prior.UniformComovingVolume(
            minimum=10, maximum=10000, name='luminosity_distance')
        self.assertEqual(prior.maximum, 10000)

    def test_zero_minimum_works(self):
        prior = bilby.gw.prior.UniformComovingVolume(
            minimum=0, maximum=10000, name='luminosity_distance')
        self.assertEqual(prior.minimum, 0)

    def test_specify_cosmology(self):
        prior = bilby.gw.prior.UniformComovingVolume(
            minimum=10, maximum=10000, name='luminosity_distance',
            cosmology='Planck13')
        self.assertEqual(repr(prior.cosmology), repr(cosmology.Planck13))

    def test_comoving_prior_creation(self):
        prior = bilby.gw.prior.UniformComovingVolume(
            minimum=10, maximum=1000, name='comoving_distance')
        self.assertEqual(prior.latex_label, '$d_C$')

    def test_redshift_prior_creation(self):
        prior = bilby.gw.prior.UniformComovingVolume(
            minimum=0.1, maximum=1, name='redshift')
        self.assertEqual(prior.latex_label, '$z$')

    def test_redshift_to_luminosity_distance(self):
        prior = bilby.gw.prior.UniformComovingVolume(
            minimum=0.1, maximum=1, name='redshift')
        new_prior = prior.get_corresponding_prior('luminosity_distance')
        self.assertEqual(new_prior.name, 'luminosity_distance')

    def test_luminosity_distance_to_redshift(self):
        prior = bilby.gw.prior.UniformComovingVolume(
            minimum=10, maximum=10000, name='luminosity_distance')
        new_prior = prior.get_corresponding_prior('redshift')
        self.assertEqual(new_prior.name, 'redshift')

    def test_luminosity_distance_to_comoving_distance(self):
        prior = bilby.gw.prior.UniformComovingVolume(
            minimum=10, maximum=10000, name='luminosity_distance')
        new_prior = prior.get_corresponding_prior('comoving_distance')
        self.assertEqual(new_prior.name, 'comoving_distance')


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
