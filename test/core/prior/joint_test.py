import unittest
import bilby

import numpy as np


class TestMultivariateGaussianDistFromRepr(unittest.TestCase):
    def test_mvg_from_repr(self):
        mvg = bilby.core.prior.MultivariateGaussianDist(
            names=["testa", "testb"],
            mus=[1, 1],
            covs=np.array([[2.0, 0.5], [0.5, 2.0]]),
            weights=1.0,
        )

        # string representation
        mvgstr = """\
MultivariateGaussianDist(
    names=['testa', 'testb'],
    nmodes=1,
    mus=[[1, 1]],
    corrcoefs=[[[1.0, 0.25], [0.25, 1.0]]],
    covs=[[[2.0, 0.5], [0.5, 2.0]]],
    weights=[1.0],
    bounds={'testa': (-inf, inf), 'testb': (-inf, inf)}
)"""

        fromstr = bilby.core.prior.MultivariateGaussianDist.from_repr(mvgstr)

        for key, item in mvg.__dict__.items():
            if isinstance(item, dict):
                self.assertTrue(item == fromstr.__getattribute__(key))
            elif key == "mvn":
                for d1, d2 in zip(fromstr.__getattribute__(key), item):
                    self.assertTrue(type(d1) == type(d2))  # noqa: E721
            elif isinstance(item, (list, tuple, np.ndarray)):
                self.assertTrue(
                    np.all(np.array(item) == np.array(fromstr.__getattribute__(key)))
                )


class TestMultivariateGaussianDistParameterScales(unittest.TestCase):
    def _test_mvg_ln_prob_diff_expected(self, mvg, weights, muss, sigmass, corrcoefss):
        all_test_points = []
        all_expected_probs = []

        # first test all modes individually and store the results
        for mode, (weight, mus, sigmas, corrcoefs) in enumerate(zip(weights, muss, sigmass, corrcoefss)):
            # the columns of the Cholesky decompsition give the directions along which
            # the multivariate Gaussian PDF will decrease by exact differences per unit
            # sigma; test that these are as expected
            ln_prob_mus = mvg.ln_prob(mus, mode=mode)
            d = np.linalg.cholesky(corrcoefs)
            test_points = []
            test_point_probs = []
            for i in np.ndindex(4, 4, 4):
                ln_prob_mus_sigmas_d_i = mvg.ln_prob(mus + sigmas * (d @ i), mode=mode)
                test_points.append(mus + sigmas * (d @ i))
                test_point_probs.append(ln_prob_mus_sigmas_d_i)
                diff_ln_prob = ln_prob_mus - ln_prob_mus_sigmas_d_i
                diff_ln_prob_expected = 0.5 * np.sum(np.array(i)**2)
                self.assertTrue(
                    np.allclose(diff_ln_prob, diff_ln_prob_expected)
                )
            test_point_probs_at_once = mvg.ln_prob(test_points, mode=mode)

            np.testing.assert_allclose(test_point_probs, test_point_probs_at_once)
            all_test_points.append(test_points)
            all_expected_probs.append(test_point_probs_at_once)

        # For the points associated with each mode, we have calculated the expected probabilities
        # above. We can test if the other modes are taken into account correctly by:
        # 1. For the points with known values for *one* mode, calculate the probabilties summed over all modes
        # 2. Calculate the probabilities for these points for all modes except the one for which we know the true values
        # 3. Subtract the probability over all other modes from the total probability
        # 4. Check if the resulting probability (adjusted for the weight of the mode) is equal to the
        #   expected value
        for current_mode, (test_points, expected_probs) in enumerate(zip(all_test_points, all_expected_probs)):
            test_point_probs_at_once_all_modes = mvg.ln_prob(test_points)
            prob_other_modes = np.full(len(test_points), -np.inf)
            for mode, weight in enumerate(weights):
                if mode == current_mode:
                    continue
                prob_other_modes = np.logaddexp(prob_other_modes, np.log(weight) + mvg.ln_prob(test_points, mode=mode))
            ln_prob_current_mode = np.log(np.exp(test_point_probs_at_once_all_modes) - np.exp(prob_other_modes))
            ln_prob_current_mode -= np.log(weights[current_mode])
            np.testing.assert_allclose(ln_prob_current_mode, expected_probs)

    def test_mvg_unit_scales(self):
        # test using order-unity standard deviations and correlations
        sigmas_1 = 0.3 * np.ones(3)
        corrcoefs_1 = np.identity(3)
        mus_1 = np.array([3, 1, 2])

        sigmas_2 = 0.4 * np.ones(3)
        corrcoefs_2 = np.identity(3)
        mus_2 = np.array([3, 1, 2])

        sigmas_3 = 0.1 * np.ones(3)
        corrcoefs_3 = np.identity(3)
        mus_3 = np.array([3.2, 1., 2.5])
        weights = [0.5, 0.3, 0.2]
        mvg = bilby.core.prior.MultivariateGaussianDist(
            nmodes=3,
            names=['a', 'b', 'c'],
            mus=[mus_1, mus_2, mus_3],
            sigmas=[sigmas_1, sigmas_2, sigmas_3],
            corrcoefs=[corrcoefs_1, corrcoefs_2, corrcoefs_3],
            weights=weights
        )

        self._test_mvg_ln_prob_diff_expected(mvg, mvg.weights, mvg.mus, mvg.sigmas, mvg.corrcoefs)

    def test_mvg_cw_scales(self):
        # test using standard deviations and correlations from the
        # inverse Fisher information matrix for the frequency/spindown
        # parameters of a continuous wave signal
        T = 365.25 * 86400
        rho = 10
        sigmas = np.array([
            5 * np.sqrt(3) / (2 * np.pi * T * rho),
            6 * np.sqrt(5) / (np.pi * T**2 * rho),
            60 * np.sqrt(7) / (np.pi * T**3 * rho)
        ])
        corrcoefs = np.identity(3)
        corrcoefs[0, 2] = corrcoefs[2, 0] = -np.sqrt(21) / 5

        # test MultivariateGaussianDist() can handle parameters with very different scales:
        # - f ~ 100, fd ~ 1/T, fdd ~ 1/T^2
        mus = [123.4, -5.6e-8, 9e-18]
        mvg = bilby.core.prior.MultivariateGaussianDist(
            names=["f", "fd", "fdd"],
            mus=mus,
            sigmas=sigmas,
            corrcoefs=corrcoefs,
        )

        self._test_mvg_ln_prob_diff_expected(mvg, [1], [mus], [sigmas], [corrcoefs])


if __name__ == "__main__":
    unittest.main()
