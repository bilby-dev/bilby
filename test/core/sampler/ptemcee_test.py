import unittest

from bilby.core.likelihood import GaussianLikelihood
from bilby.core.prior import Uniform, PriorDict
from bilby.core.sampler.ptemcee import Ptemcee
from bilby.core.sampler.base_sampler import MCMCSampler
import numpy as np
import os


class TestPTEmcee(unittest.TestCase):
    def setUp(self):
        self.likelihood = GaussianLikelihood(
            x=np.linspace(0, 1, 2),
            y=np.linspace(0, 1, 2),
            func=lambda x, **kwargs: x,
            sigma=1,
        )
        self.priors = PriorDict(dict(a=Uniform(0, 1), b=Uniform(0, 1)))
        self._args = (self.likelihood, self.priors)
        self._kwargs = dict(
            outdir="outdir",
            label="label",
            use_ratio=False,
            plot=False,
            skip_import_verification=True,
        )
        self.sampler = Ptemcee(*self._args, **self._kwargs)
        self.expected = dict(
            ntemps=10,
            nwalkers=100,
            Tmax=None,
            betas=None,
            a=2.0,
            adaptation_lag=10000,
            adaptation_time=100,
            random=None,
            adapt=False,
            swap_ratios=False,
        )

    def tearDown(self):
        del self.likelihood
        del self.priors
        del self.sampler

    def test_default_kwargs(self):
        self.assertDictEqual(self.expected, self.sampler.kwargs)

    def test_translate_kwargs(self):
        for equiv in MCMCSampler.nwalkers_equiv_kwargs:
            new_kwargs = self.sampler.kwargs.copy()
            del new_kwargs["nwalkers"]
            new_kwargs[equiv] = 100
            self.sampler.kwargs = new_kwargs
            self.assertDictEqual(self.expected, self.sampler.kwargs)

    def test_set_pos0_using_array(self):
        """
        Verify that setting the initial points from an array matches the
        default method.
        """
        pos0 = self.sampler.get_pos0()
        new_sampler = Ptemcee(*self._args, **self._kwargs, pos0=pos0)
        self.assertTrue(np.array_equal(new_sampler.get_pos0(), pos0))

    def test_set_pos0_using_dict(self):
        """
        Verify that setting the initial points from a dictionary matches the
        default method.
        """
        old = np.array(self.sampler.get_pos0())
        pos0 = np.moveaxis(old, -1, 0)
        pos0 = {
            key: points for key, points in
            zip(self.sampler.search_parameter_keys, pos0)
        }
        new_sampler = Ptemcee(*self._args, **self._kwargs, pos0=pos0)
        new = new_sampler.get_pos0()
        self.assertTrue(np.array_equal(new, old))

    def test_set_pos0_from_minimize(self):
        """
        Verify that the minimize method of setting the initial points
        returns the same shape as the default.
        """
        old = self.sampler.get_pos0().shape
        new_sampler = Ptemcee(*self._args, **self._kwargs, pos0="minimize")
        new = new_sampler.get_pos0().shape
        self.assertEqual(old, new)


def test_get_expected_outputs():
    label = "par0"
    outdir = os.path.join("some", "bilby_pipe", "dir")
    filenames, directories = Ptemcee.get_expected_outputs(
        outdir=outdir, label=label
    )
    assert len(filenames) == 1
    assert len(directories) == 0
    assert os.path.join(outdir, f"{label}_checkpoint_resume.pickle") in filenames


if __name__ == "__main__":
    unittest.main()
