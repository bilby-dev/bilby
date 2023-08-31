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
    def _test_mvg_ln_prob_diff_expected(self, mvg, mus, sigmas, corrcoefs):
        # the columns of the Cholesky decompsition give the directions along which
        # the multivariate Gaussian PDF will decrease by exact differences per unit
        # sigma; test that these are as expected
        ln_prob_mus = mvg.ln_prob(mus)
        d = np.linalg.cholesky(corrcoefs)
        for i in np.ndindex(4, 4, 4):
            ln_prob_mus_sigmas_d_i = mvg.ln_prob(mus + sigmas * (d @ i))
            diff_ln_prob = ln_prob_mus - ln_prob_mus_sigmas_d_i
            diff_ln_prob_expected = 0.5 * np.sum(np.array(i)**2)
            self.assertTrue(
                np.allclose(diff_ln_prob, diff_ln_prob_expected)
            )

    def test_mvg_unit_scales(self):
        # test using order-unity standard deviations and correlations
        sigmas = 0.3 * np.ones(3)
        corrcoefs = np.identity(3)
        mus = np.array([3, 1, 2])
        mvg = bilby.core.prior.MultivariateGaussianDist(
            names=['a', 'b', 'c'],
            mus=mus,
            sigmas=sigmas,
            corrcoefs=corrcoefs,
        )

        self._test_mvg_ln_prob_diff_expected(mvg, mus, sigmas, corrcoefs)

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

        self._test_mvg_ln_prob_diff_expected(mvg, mus, sigmas, corrcoefs)


if __name__ == "__main__":
    unittest.main()
