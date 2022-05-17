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
                    self.assertTrue(type(d1) == type(d2))
            elif isinstance(item, (list, tuple, np.ndarray)):
                self.assertTrue(
                    np.all(np.array(item) == np.array(fromstr.__getattribute__(key)))
                )


if __name__ == "__main__":
    unittest.main()
