import unittest
from unittest import mock

import numpy as np

import bilby.gw.sampler.proposal
from bilby.core import prior
from bilby.core.sampler import proposal


class TestSkyLocationWanderJump(unittest.TestCase):
    def setUp(self):
        self.priors = prior.PriorDict(
            dict(
                ra=prior.Uniform(minimum=0.0, maximum=2 * np.pi, boundary="periodic"),
                dec=prior.Uniform(minimum=0.0, maximum=np.pi, boundary="reflective"),
            )
        )
        self.jump_proposal = bilby.gw.sampler.proposal.SkyLocationWanderJump(
            priors=self.priors
        )

    def tearDown(self):
        del self.priors
        del self.jump_proposal

    def test_jump_proposal_call_without_inverse_temperature(self):
        with mock.patch("random.gauss") as m:
            m.return_value = 1
            sample = proposal.Sample(dict(ra=0.2, dec=-0.5))
            expected = proposal.Sample(dict(ra=1.2, dec=0.5))
            new_sample = self.jump_proposal(sample)
            for key, value in new_sample.items():
                self.assertAlmostEqual(expected[key], value)
            m.assert_called_with(0, 1.0 / 2 / np.pi)

    def test_jump_proposal_call_with_inverse_temperature(self):
        with mock.patch("random.gauss") as m:
            m.return_value = 1
            sample = proposal.Sample(dict(ra=0.2, dec=-0.5))
            expected = proposal.Sample(dict(ra=1.2, dec=0.5))
            new_sample = self.jump_proposal(sample, inverse_temperature=2.0)
            for key, value in new_sample.items():
                self.assertAlmostEqual(expected[key], value)
            m.assert_called_with(0, np.sqrt(1 / 2.0) / 2 / np.pi)


class TestCorrelatedPolarisationPhaseJump(unittest.TestCase):
    def setUp(self):
        self.priors = prior.PriorDict(
            dict(
                phase=prior.Uniform(minimum=0.0, maximum=2 * np.pi),
                psi=prior.Uniform(minimum=0.0, maximum=np.pi),
            )
        )
        self.jump_proposal = bilby.gw.sampler.proposal.CorrelatedPolarisationPhaseJump(
            priors=self.priors
        )

    def tearDown(self):
        del self.priors
        del self.jump_proposal

    def test_jump_proposal_call_case_1(self):
        with mock.patch("random.random") as m:
            m.return_value = 0.3
            sample = proposal.Sample(dict(phase=0.2, psi=0.5))
            alpha = 3.0 * np.pi * 0.3
            beta = 0.3
            expected = proposal.Sample(
                dict(phase=0.5 * (alpha - beta), psi=0.5 * (alpha + beta))
            )
            self.assertEqual(expected, self.jump_proposal(sample, coordinates=None))

    def test_jump_proposal_call_case_2(self):
        with mock.patch("random.random") as m:
            m.return_value = 0.7
            sample = proposal.Sample(dict(phase=0.2, psi=0.5))
            alpha = 0.7
            beta = 3.0 * np.pi * 0.7 - 2 * np.pi
            expected = proposal.Sample(
                dict(phase=0.5 * (alpha - beta), psi=0.5 * (alpha + beta))
            )
            self.assertEqual(expected, self.jump_proposal(sample))


class TestPolarisationPhaseJump(unittest.TestCase):
    def setUp(self):
        self.priors = prior.PriorDict(
            dict(
                phase=prior.Uniform(minimum=0.0, maximum=2 * np.pi),
                psi=prior.Uniform(minimum=0.0, maximum=np.pi),
            )
        )
        self.jump_proposal = bilby.gw.sampler.proposal.PolarisationPhaseJump(
            priors=self.priors
        )

    def tearDown(self):
        del self.priors
        del self.jump_proposal

    def test_jump_proposal_call(self):
        sample = proposal.Sample(dict(phase=0.2, psi=0.5))
        expected = proposal.Sample(dict(phase=0.2 + np.pi, psi=0.5 + np.pi / 2))
        self.assertEqual(expected, self.jump_proposal(sample))


class TestDrawFlatPrior(unittest.TestCase):
    def setUp(self):
        self.priors = prior.PriorDict(
            dict(
                phase=prior.Uniform(minimum=0.0, maximum=2 * np.pi),
                psi=prior.Cosine(minimum=0.0, maximum=np.pi),
            )
        )
        self.jump_proposal = proposal.DrawFlatPrior(priors=self.priors)

    def tearDown(self):
        del self.priors
        del self.jump_proposal

    def test_jump_proposal_call(self):
        with mock.patch("bilby.core.prior.Uniform.sample") as m:
            m.return_value = 0.3
            sample = proposal.Sample(dict(phase=0.2, psi=0.5))
            expected = proposal.Sample(dict(phase=0.3, psi=0.3))
            self.assertEqual(expected, self.jump_proposal(sample))


if __name__ == "__main__":
    unittest.main()
