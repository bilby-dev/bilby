import unittest
import mock
import random

import numpy as np

import bilby.gw.sampler.proposal
from bilby.core import prior
from bilby.core.sampler import proposal


class TestSample(unittest.TestCase):

    def setUp(self):
        self.sample = proposal.Sample(dict(a=1, c=2))

    def tearDown(self):
        del self.sample

    def test_add_sample(self):
        other = proposal.Sample(dict(a=2, c=5))
        expected = proposal.Sample(dict(a=3, c=7))
        self.assertDictEqual(expected, self.sample + other)

    def test_subtract_sample(self):
        other = proposal.Sample(dict(a=2, c=5))
        expected = proposal.Sample(dict(a=-1, c=-3))
        self.assertDictEqual(expected, self.sample - other)

    def test_multiply_sample(self):
        other = 2
        expected = proposal.Sample(dict(a=2, c=4))
        self.assertDictEqual(expected, self.sample * other)


class TestJumpProposal(unittest.TestCase):

    def setUp(self):
        self.priors = prior.PriorDict(dict(reflecting=prior.Uniform(minimum=-0.5, maximum=1, periodic_boundary=False),
                                           periodic=prior.Uniform(minimum=-0.5, maximum=1, periodic_boundary=True),
                                           default=prior.Uniform(minimum=-0.5, maximum=1)))
        self.sample_above = dict(reflecting=1.1, periodic=1.1, default=1.1)
        self.sample_below = dict(reflecting=-0.6, periodic=-0.6, default=-0.6)
        self.sample_way_above_case1 = dict(reflecting=272, periodic=272, default=272)
        self.sample_way_above_case2 = dict(reflecting=270.1, periodic=270.1, default=270.1)
        self.sample_way_below_case1 = dict(reflecting=-274, periodic=-274.1, default=-274)
        self.sample_way_below_case2 = dict(reflecting=-273.1, periodic=-273.1, default=-273.1)
        self.jump_proposal = proposal.JumpProposal(priors=self.priors)

    def tearDown(self):
        del self.priors
        del self.sample_above
        del self.sample_below
        del self.sample_way_above_case1
        del self.sample_way_above_case2
        del self.sample_way_below_case1
        del self.sample_way_below_case2
        del self.jump_proposal

    def test_set_get_log_j(self):
        self.jump_proposal.log_j = 2.3
        self.assertEqual(2.3, self.jump_proposal.log_j)

    def test_boundary_above_reflecting(self):
        new_sample = self.jump_proposal(self.sample_above)
        self.assertAlmostEqual(0.9, new_sample['reflecting'])

    def test_boundary_above_periodic(self):
        new_sample = self.jump_proposal(self.sample_above)
        self.assertAlmostEqual(-0.4, new_sample['periodic'])

    def test_boundary_above_default(self):
        new_sample = self.jump_proposal(self.sample_above)
        self.assertAlmostEqual(0.9, new_sample['default'])

    def test_boundary_below_reflecting(self):
        new_sample = self.jump_proposal(self.sample_below)
        self.assertAlmostEqual(-0.4, new_sample['reflecting'])

    def test_boundary_below_periodic(self):
        new_sample = self.jump_proposal(self.sample_below)
        self.assertAlmostEqual(0.9, new_sample['periodic'])

    def test_boundary_below_default(self):
        new_sample = self.jump_proposal(self.sample_below)
        self.assertAlmostEqual(-0.4, new_sample['default'])

    def test_boundary_way_below_reflecting_case1(self):
        new_sample = self.jump_proposal(self.sample_way_below_case1)
        self.assertAlmostEqual(0.0, new_sample['reflecting'])

    def test_boundary_way_below_reflecting_case2(self):
        new_sample = self.jump_proposal(self.sample_way_below_case2)
        self.assertAlmostEqual(-0.1, new_sample['reflecting'])

    def test_boundary_way_below_periodic(self):
        new_sample = self.jump_proposal(self.sample_way_below_case2)
        self.assertAlmostEqual(-0.1, new_sample['periodic'])

    def test_boundary_way_above_reflecting_case1(self):
        new_sample = self.jump_proposal(self.sample_way_above_case1)
        self.assertAlmostEqual(0.0, new_sample['reflecting'])

    def test_boundary_way_above_reflecting_case2(self):
        new_sample = self.jump_proposal(self.sample_way_above_case2)
        self.assertAlmostEqual(0.1, new_sample['reflecting'])

    def test_boundary_way_above_periodic(self):
        new_sample = self.jump_proposal(self.sample_way_above_case2)
        self.assertAlmostEqual(0.1, new_sample['periodic'])

    def test_priors(self):
        self.assertEqual(self.priors, self.jump_proposal.priors)


class TestNormJump(unittest.TestCase):

    def setUp(self):
        self.priors = prior.PriorDict(dict(reflecting=prior.Uniform(minimum=-0.5, maximum=1, periodic_boundary=True),
                                           periodic=prior.Uniform(minimum=-0.5, maximum=1, periodic_boundary=False),
                                           default=prior.Uniform(minimum=-0.5, maximum=1)))
        self.jump_proposal = proposal.NormJump(step_size=3.0, priors=self.priors)

    def tearDown(self):
        del self.priors
        del self.jump_proposal

    def test_step_size_init(self):
        self.assertEqual(3.0, self.jump_proposal.step_size)

    def test_set_step_size(self):
        self.jump_proposal.step_size = 1.0
        self.assertEqual(1.0, self.jump_proposal.step_size)

    def test_jump_proposal_call(self):
        with mock.patch("numpy.random.normal") as m:
            m.return_value = 0.5
            sample = proposal.Sample(dict(reflecting=0.0, periodic=0.0, default=0.0))
            new_sample = self.jump_proposal(sample)
            expected = proposal.Sample(dict(reflecting=0.5, periodic=0.5, default=0.5))
            self.assertDictEqual(expected, new_sample)


class TestEnsembleWalk(unittest.TestCase):

    def setUp(self):
        self.priors = prior.PriorDict(dict(reflecting=prior.Uniform(minimum=-0.5, maximum=1, periodic_boundary=False),
                                           periodic=prior.Uniform(minimum=-0.5, maximum=1, periodic_boundary=True),
                                           default=prior.Uniform(minimum=-0.5, maximum=1)))
        self.jump_proposal = proposal.EnsembleWalk(random_number_generator=random.random,
                                                   n_points=4, priors=self.priors)

    def tearDown(self):
        del self.priors
        del self.jump_proposal

    def test_n_points_init(self):
        self.assertEqual(4, self.jump_proposal.n_points)

    def test_set_n_points(self):
        self.jump_proposal.n_points = 3
        self.assertEqual(3, self.jump_proposal.n_points)

    def test_random_number_generator_init(self):
        self.assertEqual(random.random, self.jump_proposal.random_number_generator)

    def test_get_center_of_mass(self):
        samples = [proposal.Sample(dict(reflecting=0.1*i, periodic=0.1*i, default=0.1*i)) for i in range(3)]
        expected = proposal.Sample(dict(reflecting=0.1, periodic=0.1, default=0.1))
        actual = self.jump_proposal.get_center_of_mass(samples)
        for key in samples[0].keys():
            self.assertAlmostEqual(expected[key], actual[key])

    def test_jump_proposal_call(self):
        with mock.patch('random.sample') as m:
            self.jump_proposal.random_number_generator = lambda: 2
            m.return_value = [proposal.Sample(dict(periodic=0.3, reflecting=0.3, default=0.3)),
                              proposal.Sample(dict(periodic=0.1, reflecting=0.1, default=0.1))]
            sample = proposal.Sample(dict(periodic=0.1, reflecting=0.1, default=0.1))
            new_sample = self.jump_proposal(sample, coordinates=None)
            expected = proposal.Sample(dict(periodic=0.1, reflecting=0.1, default=0.1))
            for key, value in new_sample.items():
                self.assertAlmostEqual(expected[key], value)


class TestEnsembleEnsembleStretch(unittest.TestCase):

    def setUp(self):
        self.priors = prior.PriorDict(dict(reflecting=prior.Uniform(minimum=-0.5, maximum=1, periodic_boundary=False),
                                           periodic=prior.Uniform(minimum=-0.5, maximum=1, periodic_boundary=True),
                                           default=prior.Uniform(minimum=-0.5, maximum=1)))
        self.jump_proposal = proposal.EnsembleStretch(scale=3.0, priors=self.priors)

    def tearDown(self):
        del self.priors
        del self.jump_proposal

    def test_scale_init(self):
        self.assertEqual(3.0, self.jump_proposal.scale)

    def test_set_get_scale(self):
        self.jump_proposal.scale = 5.0
        self.assertEqual(5.0, self.jump_proposal.scale)

    def test_jump_proposal_call(self):
        with mock.patch('random.choice') as m:
            with mock.patch('random.uniform') as n:
                second_sample = proposal.Sample(dict(periodic=0.3, reflecting=0.3, default=0.3))
                random_number = 0.5
                m.return_value = second_sample
                n.return_value = random_number
                sample = proposal.Sample(dict(periodic=0.1, reflecting=0.1, default=0.1))
                new_sample = self.jump_proposal(sample, coordinates=None)
                coords = 0.3 - 0.2 * np.exp(random_number * np.log(self.jump_proposal.scale))
                expected = proposal.Sample(dict(periodic=coords, reflecting=coords, default=coords))
                for key, value in new_sample.items():
                    self.assertAlmostEqual(expected[key], value)

    def test_log_j_after_call(self):
        with mock.patch('random.uniform') as m1:
            with mock.patch('numpy.log') as m2:
                with mock.patch('numpy.exp') as m3:
                    m1.return_value = 1
                    m2.return_value = 1
                    m3.return_value = 1
                    coordinates = [proposal.Sample(dict(periodic=0.3, reflecting=0.3, default=0.3)),
                                   proposal.Sample(dict(periodic=0.3, reflecting=0.3, default=0.3))]
                    sample = proposal.Sample(dict(periodic=0.2, reflecting=0.2, default=0.2))
                    self.jump_proposal(sample=sample,
                                       coordinates=coordinates)
                    self.assertEqual(3, self.jump_proposal.log_j)


class TestDifferentialEvolution(unittest.TestCase):

    def setUp(self):
        self.priors = prior.PriorDict(dict(reflecting=prior.Uniform(minimum=-0.5, maximum=1, periodic_boundary=False),
                                           periodic=prior.Uniform(minimum=-0.5, maximum=1, periodic_boundary=True),
                                           default=prior.Uniform(minimum=-0.5, maximum=1)))
        self.jump_proposal = proposal.DifferentialEvolution(sigma=1e-3, mu=0.5, priors=self.priors)

    def tearDown(self):
        del self.priors
        del self.jump_proposal

    def test_mu_init(self):
        self.assertEqual(0.5, self.jump_proposal.mu)

    def test_set_get_mu(self):
        self.jump_proposal.mu = 1
        self.assertEqual(1, self.jump_proposal.mu)

    def test_set_get_sigma(self):
        self.jump_proposal.sigma = 2
        self.assertEqual(2, self.jump_proposal.sigma)

    def test_jump_proposal_call(self):
        with mock.patch('random.sample') as m:
            with mock.patch('random.gauss') as n:
                m.return_value = proposal.Sample(dict(periodic=0.2, reflecting=0.2, default=0.2)),\
                                 proposal.Sample(dict(periodic=0.3, reflecting=0.3, default=0.3))
                n.return_value = 1
                sample = proposal.Sample(dict(periodic=0.1, reflecting=0.1, default=0.1))
                expected = proposal.Sample(dict(periodic=0.2, reflecting=0.2, default=0.2))
                new_sample = self.jump_proposal(sample, coordinates=None)
                for key, value in new_sample.items():
                    self.assertAlmostEqual(expected[key], value)


class TestEnsembleEigenVector(unittest.TestCase):

    def setUp(self):
        self.priors = prior.PriorDict(dict(reflecting=prior.Uniform(minimum=-0.5, maximum=1, periodic_boundary=False),
                                           periodic=prior.Uniform(minimum=-0.5, maximum=1, periodic_boundary=True),
                                           default=prior.Uniform(minimum=-0.5, maximum=1)))
        self.jump_proposal = proposal.EnsembleEigenVector(priors=self.priors)

    def tearDown(self):
        del self.priors
        del self.jump_proposal

    def test_init_eigen_values(self):
        self.assertIsNone(self.jump_proposal.eigen_values)

    def test_init_eigen_vectors(self):
        self.assertIsNone(self.jump_proposal.eigen_vectors)

    def test_init_covariance(self):
        self.assertIsNone(self.jump_proposal.covariance)

    def test_jump_proposal_update_eigenvectors_none(self):
        self.assertIsNone(self.jump_proposal.update_eigenvectors(coordinates=None))

    def test_jump_proposal_update_eigenvectors_1_d(self):
        coordinates = [proposal.Sample(dict(periodic=0.3)), proposal.Sample(dict(periodic=0.1))]
        with mock.patch('numpy.var') as m:
            m.return_value = 1
            self.jump_proposal.update_eigenvectors(coordinates)
            self.assertTrue(np.equal(np.array([1]), self.jump_proposal.eigen_values))
            self.assertTrue(np.equal(np.array([1]), self.jump_proposal.covariance))
            self.assertTrue(np.equal(np.array([[1.]]), self.jump_proposal.eigen_vectors))

    def test_jump_proposal_update_eigenvectors_n_d(self):
        coordinates = [proposal.Sample(dict(periodic=0.3, reflecting=0.3, default=0.3)),
                       proposal.Sample(dict(periodic=0.1, reflecting=0.1, default=0.1))]
        with mock.patch('numpy.cov') as m:
            with mock.patch('numpy.linalg.eigh') as n:
                m.side_effect = lambda x: x
                n.return_value = 1, 2
                self.jump_proposal.update_eigenvectors(coordinates)
                self.assertTrue(np.array_equal(np.array([[0.3, 0.1], [0.3, 0.1], [0.3, 0.1]]), self.jump_proposal.covariance))
                self.assertEqual(1, self.jump_proposal.eigen_values)
                self.assertEqual(2, self.jump_proposal.eigen_vectors)

    def test_jump_proposal_call(self):
        self.jump_proposal.update_eigenvectors = lambda x: None
        self.jump_proposal.eigen_values = np.array([1, np.nan, np.nan])
        self.jump_proposal.eigen_vectors = np.array([[0.1, np.nan, np.nan],
                                                    [0.4, np.nan, np.nan],
                                                    [0.7, np.nan, np.nan]])
        with mock.patch('random.randrange') as m:
            with mock.patch('random.gauss') as n:
                m.return_value = 0
                n.return_value = 1
                expected = proposal.Sample()
                expected['periodic'] = 0.2
                expected['reflecting'] = 0.5
                expected['default'] = 0.8
                sample = proposal.Sample()
                sample['periodic'] = 0.1
                sample['reflecting'] = 0.1
                sample['default'] = 0.1
                new_sample = self.jump_proposal(sample, coordinates=None)
                for key, value in new_sample.items():
                    self.assertAlmostEqual(expected[key], value)


class TestSkyLocationWanderJump(unittest.TestCase):

    def setUp(self):
        self.priors = prior.PriorDict(dict(ra=prior.Uniform(minimum=0.0, maximum=2*np.pi, periodic_boundary=True),
                                           dec=prior.Uniform(minimum=0.0, maximum=np.pi, periodic_boundary=False)))
        self.jump_proposal = bilby.gw.sampler.proposal.SkyLocationWanderJump(priors=self.priors)

    def tearDown(self):
        del self.priors
        del self.jump_proposal

    def test_jump_proposal_call_without_inverse_temperature(self):
        with mock.patch('random.gauss') as m:
            m.return_value = 1
            sample = proposal.Sample(dict(ra=0.2, dec=-0.5))
            expected = proposal.Sample(dict(ra=1.2, dec=0.5))
            new_sample = self.jump_proposal(sample)
            for key, value in new_sample.items():
                self.assertAlmostEqual(expected[key], value)
            m.assert_called_with(0, 1.0 / 2 / np.pi)

    def test_jump_proposal_call_with_inverse_temperature(self):
        with mock.patch('random.gauss') as m:
            m.return_value = 1
            sample = proposal.Sample(dict(ra=0.2, dec=-0.5))
            expected = proposal.Sample(dict(ra=1.2, dec=0.5))
            new_sample = self.jump_proposal(sample, inverse_temperature=2.0)
            for key, value in new_sample.items():
                self.assertAlmostEqual(expected[key], value)
            m.assert_called_with(0, np.sqrt(1 / 2.0) / 2 / np.pi)


class TestCorrelatedPolarisationPhaseJump(unittest.TestCase):

    def setUp(self):
        self.priors = prior.PriorDict(dict(phase=prior.Uniform(minimum=0.0, maximum=2*np.pi),
                                           psi=prior.Uniform(minimum=0.0, maximum=np.pi)))
        self.jump_proposal = bilby.gw.sampler.proposal.CorrelatedPolarisationPhaseJump(priors=self.priors)

    def tearDown(self):
        del self.priors
        del self.jump_proposal

    def test_jump_proposal_call_case_1(self):
        with mock.patch('random.random') as m:
            m.return_value = 0.3
            sample = proposal.Sample(dict(phase=0.2, psi=0.5))
            alpha = 3.0 * np.pi * 0.3
            beta = 0.3
            expected = proposal.Sample(dict(phase=0.5*(alpha-beta), psi=0.5*(alpha+beta)))
            self.assertEqual(expected, self.jump_proposal(sample, coordinates=None))

    def test_jump_proposal_call_case_2(self):
        with mock.patch('random.random') as m:
            m.return_value = 0.7
            sample = proposal.Sample(dict(phase=0.2, psi=0.5))
            alpha = 0.7
            beta = 3.0 * np.pi * 0.7 - 2 * np.pi
            expected = proposal.Sample(dict(phase=0.5*(alpha-beta), psi=0.5*(alpha+beta)))
            self.assertEqual(expected, self.jump_proposal(sample))


class TestPolarisationPhaseJump(unittest.TestCase):

    def setUp(self):
        self.priors = prior.PriorDict(dict(phase=prior.Uniform(minimum=0.0, maximum=2*np.pi),
                                           psi=prior.Uniform(minimum=0.0, maximum=np.pi)))
        self.jump_proposal = bilby.gw.sampler.proposal.PolarisationPhaseJump(priors=self.priors)

    def tearDown(self):
        del self.priors
        del self.jump_proposal

    def test_jump_proposal_call(self):
        sample = proposal.Sample(dict(phase=0.2, psi=0.5))
        expected = proposal.Sample(dict(phase=0.2+np.pi, psi=0.5+np.pi/2))
        self.assertEqual(expected, self.jump_proposal(sample))


class TestDrawFlatPrior(unittest.TestCase):

    def setUp(self):
        self.priors = prior.PriorDict(dict(phase=prior.Uniform(minimum=0.0, maximum=2*np.pi),
                                           psi=prior.Cosine(minimum=0.0, maximum=np.pi)))
        self.jump_proposal = proposal.DrawFlatPrior(priors=self.priors)

    def tearDown(self):
        del self.priors
        del self.jump_proposal

    def test_jump_proposal_call(self):
        with mock.patch('bilby.core.prior.Uniform.sample') as m:
            m.return_value = 0.3
            sample = proposal.Sample(dict(phase=0.2, psi=0.5))
            expected = proposal.Sample(dict(phase=0.3, psi=0.3))
            self.assertEqual(expected, self.jump_proposal(sample))
