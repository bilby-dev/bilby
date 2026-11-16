import unittest
from unittest import mock

import numpy as np

import bilby
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
        self.priors = prior.PriorDict(
            dict(
                periodic=prior.Uniform(minimum=-0.5, maximum=1, boundary="periodic"),
                default=prior.Uniform(minimum=-0.5, maximum=1),
            )
        )
        self.sample_above = dict(periodic=1.1, default=1.1)
        self.sample_below = dict(periodic=-0.6, default=-0.6)
        self.sample_way_above = dict(periodic=270.1, default=270.1)
        self.sample_way_below = dict(periodic=-273.1, default=-273.1)
        self.jump_proposal = proposal.JumpProposal(priors=self.priors)

    def tearDown(self):
        del self.priors
        del self.sample_above
        del self.sample_below
        del self.sample_way_above
        del self.sample_way_below
        del self.jump_proposal

    def test_set_get_log_j(self):
        self.jump_proposal.log_j = 2.3
        self.assertEqual(2.3, self.jump_proposal.log_j)

    def test_boundary_above_periodic(self):
        new_sample = self.jump_proposal(self.sample_above)
        self.assertAlmostEqual(-0.4, new_sample["periodic"])

    def test_boundary_above_default(self):
        new_sample = self.jump_proposal(self.sample_above)
        self.assertAlmostEqual(1.1, new_sample["default"])

    def test_boundary_below_periodic(self):
        new_sample = self.jump_proposal(self.sample_below)
        self.assertAlmostEqual(0.9, new_sample["periodic"])

    def test_boundary_below_default(self):
        new_sample = self.jump_proposal(self.sample_below)
        self.assertAlmostEqual(-0.6, new_sample["default"])

    def test_boundary_way_below_periodic(self):
        new_sample = self.jump_proposal(self.sample_way_below)
        self.assertAlmostEqual(-0.1, new_sample["periodic"])

    def test_boundary_way_above_periodic(self):
        new_sample = self.jump_proposal(self.sample_way_above)
        self.assertAlmostEqual(0.1, new_sample["periodic"])

    def test_priors(self):
        self.assertEqual(self.priors, self.jump_proposal.priors)


class TestNormJump(unittest.TestCase):
    def setUp(self):
        self.priors = prior.PriorDict(
            dict(
                periodic=prior.Uniform(minimum=-0.5, maximum=1, boundary="periodic"),
                default=prior.Uniform(minimum=-0.5, maximum=1),
            )
        )
        self.jump_proposal = proposal.NormJump(step_size=3.0, priors=self.priors)
        bilby.core.utils.random.seed(5)

    def tearDown(self):
        del self.priors
        del self.jump_proposal

    def test_step_size_init(self):
        self.assertEqual(3.0, self.jump_proposal.step_size)

    def test_set_step_size(self):
        self.jump_proposal.step_size = 1.0
        self.assertEqual(1.0, self.jump_proposal.step_size)

    def test_jump_proposal_call(self):
        sample = proposal.Sample(dict(periodic=0.0, default=0.0))
        new_sample = self.jump_proposal(sample)
        expected = proposal.Sample(
            dict(
                periodic=0.5942057242396577,
                default=-3.973076986884435,
            )
        )
        self.assertDictEqual(expected, new_sample)


class TestEnsembleWalk(unittest.TestCase):
    def setUp(self):
        self.priors = prior.PriorDict(
            dict(
                periodic=prior.Uniform(minimum=-0.5, maximum=1, boundary="periodic"),
                default=prior.Uniform(minimum=-0.5, maximum=1),
            )
        )
        bilby.core.utils.random.seed(5)
        self.jump_proposal = proposal.EnsembleWalk(
            random_number_generator=bilby.core.utils.random.rng.uniform,
            n_points=4,
            priors=self.priors,
        )
        self.coordinates = [proposal.Sample(self.priors.sample()) for _ in range(10)]

    def tearDown(self):
        del self.priors
        del self.jump_proposal

    def test_n_points_init(self):
        self.assertEqual(4, self.jump_proposal.n_points)

    def test_set_n_points(self):
        self.jump_proposal.n_points = 3
        self.assertEqual(3, self.jump_proposal.n_points)

    def test_random_number_generator_init(self):
        self.assertEqual(
            bilby.core.utils.random.rng.uniform,
            self.jump_proposal.random_number_generator,
        )

    def test_get_center_of_mass(self):
        samples = [
            proposal.Sample(dict(periodic=0.1 * i, default=0.1 * i)) for i in range(3)
        ]
        expected = proposal.Sample(dict(periodic=0.1, default=0.1))
        actual = self.jump_proposal.get_center_of_mass(samples)
        for key in samples[0].keys():
            self.assertAlmostEqual(expected[key], actual[key])

    def test_jump_proposal_call(self):
        sample = proposal.Sample(dict(periodic=0.1, default=0.1))
        new_sample = self.jump_proposal(sample, coordinates=self.coordinates)
        expected = proposal.Sample(
            dict(
                periodic=-0.01001485963458848,
                default=0.41669050016670317,
            )
        )
        for key, value in new_sample.items():
            self.assertAlmostEqual(expected[key], value)


class TestEnsembleEnsembleStretch(unittest.TestCase):
    def setUp(self):
        self.priors = prior.PriorDict(
            dict(
                periodic=prior.Uniform(minimum=-0.5, maximum=1, boundary="periodic"),
                default=prior.Uniform(minimum=-0.5, maximum=1),
            )
        )
        bilby.core.utils.random.seed(5)
        self.jump_proposal = proposal.EnsembleStretch(scale=3.0, priors=self.priors)
        self.coordinates = [proposal.Sample(self.priors.sample()) for _ in range(10)]

    def tearDown(self):
        del self.priors
        del self.jump_proposal

    def test_scale_init(self):
        self.assertEqual(3.0, self.jump_proposal.scale)

    def test_set_get_scale(self):
        self.jump_proposal.scale = 5.0
        self.assertEqual(5.0, self.jump_proposal.scale)

    def test_jump_proposal_call(self):
        sample = proposal.Sample(dict(periodic=0.1, default=0.1))
        new_sample = self.jump_proposal(sample, coordinates=self.coordinates)
        expected = proposal.Sample(
            dict(
                periodic=0.26384936926580865,
                default=-0.10086519835584029,
            )
        )
        for key, value in new_sample.items():
            self.assertAlmostEqual(expected[key], value)

    def test_log_j_after_call(self):
        sample = proposal.Sample(dict(periodic=0.2, default=0.2))
        self.jump_proposal(sample=sample, coordinates=self.coordinates)
        self.assertAlmostEqual(-1.0043443032744377, self.jump_proposal.log_j, 10)


class TestDifferentialEvolution(unittest.TestCase):
    def setUp(self):
        self.priors = prior.PriorDict(
            dict(
                periodic=prior.Uniform(minimum=-0.5, maximum=1, boundary="periodic"),
                default=prior.Uniform(minimum=-0.5, maximum=1),
            )
        )
        bilby.core.utils.random.seed(5)
        self.jump_proposal = proposal.DifferentialEvolution(
            sigma=1e-3, mu=0.5, priors=self.priors
        )
        self.coordinates = [proposal.Sample(self.priors.sample()) for _ in range(10)]

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
        sample = proposal.Sample(dict(periodic=0.1, default=0.1))
        expected = proposal.Sample(
            dict(
                periodic=0.29494873527517573,
                default=-0.09386014999795975,
            )
        )
        new_sample = self.jump_proposal(sample, coordinates=self.coordinates)
        for key, value in new_sample.items():
            self.assertAlmostEqual(expected[key], value)


class TestEnsembleEigenVector(unittest.TestCase):
    def setUp(self):
        self.priors = prior.PriorDict(
            dict(
                periodic=prior.Uniform(minimum=-0.5, maximum=1, boundary="periodic"),
                default=prior.Uniform(minimum=-0.5, maximum=1),
            )
        )
        bilby.core.utils.random.seed(5)
        self.jump_proposal = proposal.EnsembleEigenVector(priors=self.priors)
        self.coordinates = [proposal.Sample(self.priors.sample()) for _ in range(10)]

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
        coordinates = [
            proposal.Sample(dict(periodic=0.3)),
            proposal.Sample(dict(periodic=0.1)),
        ]
        with mock.patch("numpy.var") as m:
            m.return_value = 1
            self.jump_proposal.update_eigenvectors(coordinates)
            self.assertTrue(np.equal(np.array([1]), self.jump_proposal.eigen_values))
            self.assertTrue(np.equal(np.array([1]), self.jump_proposal.covariance))
            self.assertTrue(
                np.equal(np.array([[1.0]]), self.jump_proposal.eigen_vectors)
            )

    def test_jump_proposal_update_eigenvectors_n_d(self):
        coordinates = [
            proposal.Sample(dict(periodic=0.3, default=0.3)),
            proposal.Sample(dict(periodic=0.1, default=0.1)),
        ]
        with mock.patch("numpy.cov") as m:
            with mock.patch("numpy.linalg.eigh") as n:
                m.side_effect = lambda x: x
                n.return_value = 1, 2
                self.jump_proposal.update_eigenvectors(coordinates)
                self.assertTrue(
                    np.array_equal(
                        np.array([[0.3, 0.1], [0.3, 0.1]]),
                        self.jump_proposal.covariance,
                    )
                )
                self.assertEqual(1, self.jump_proposal.eigen_values)
                self.assertEqual(2, self.jump_proposal.eigen_vectors)

    def test_jump_proposal_call(self):
        expected = proposal.Sample()
        expected["periodic"] = 0.06846297922445672
        expected["default"] = 0.4040642588236736
        sample = proposal.Sample()
        sample["periodic"] = 0.1
        sample["default"] = 0.1
        new_sample = self.jump_proposal(sample, coordinates=self.coordinates)
        for key, value in new_sample.items():
            self.assertAlmostEqual(expected[key], value)


if __name__ == "__main__":
    unittest.main()
