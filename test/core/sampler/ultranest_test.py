import shutil
import unittest
from unittest.mock import MagicMock

import bilby
import bilby.core.sampler.ultranest


class TestUltranest(unittest.TestCase):

    def setUp(self):
        self.maxDiff = None
        self.likelihood = MagicMock()
        self.priors = bilby.core.prior.PriorDict(
            dict(a=bilby.core.prior.Uniform(0, 1),
                 b=bilby.core.prior.Uniform(0, 1)))
        self.priors["a"] = bilby.core.prior.Prior(boundary="periodic")
        self.priors["b"] = bilby.core.prior.Prior(boundary="reflective")
        self.sampler = bilby.core.sampler.ultranest.Ultranest(
            self.likelihood,
            self.priors,
            outdir="outdir",
            label="label",
            use_ratio=False,
            plot=False,
            skip_import_verification=True,
        )

    def tearDown(self):
        del self.likelihood
        del self.priors
        del self.sampler
        shutil.rmtree("outdir")

    def test_default_kwargs(self):
        expected = dict(
            resume="overwrite",
            show_status=True,
            num_live_points=None,
            wrapped_params=None,
            derived_param_names=None,
            run_num=None,
            vectorized=False,
            num_test_samples=2,
            draw_multiple=True,
            num_bootstraps=30,
            update_interval_iter=None,
            update_interval_ncall=None,
            log_interval=None,
            dlogz=None,
            max_iters=None,
            update_interval_volume_fraction=0.2,
            viz_callback=None,
            dKL=0.5,
            frac_remain=0.01,
            Lepsilon=0.001,
            min_ess=400,
            max_ncalls=None,
            max_num_improvement_loops=-1,
            min_num_live_points=400,
            cluster_num_live_points=40,
            step_sampler=None,
        )
        self.assertListEqual([1, 0], self.sampler.kwargs["wrapped_params"])  # Check this separately
        self.sampler.kwargs["wrapped_params"] = None  # The dict comparison can't handle lists
        self.sampler.kwargs["derived_param_names"] = None
        self.sampler.kwargs["viz_callback"] = None
        self.assertDictEqual(expected, self.sampler.kwargs)

    def test_translate_kwargs(self):
        expected = dict(
            resume="overwrite",
            show_status=True,
            num_live_points=123,
            wrapped_params=None,
            derived_param_names=None,
            run_num=None,
            vectorized=False,
            num_test_samples=2,
            draw_multiple=True,
            num_bootstraps=30,
            update_interval_iter=None,
            update_interval_ncall=None,
            log_interval=None,
            dlogz=None,
            max_iters=None,
            update_interval_volume_fraction=0.2,
            viz_callback=None,
            dKL=0.5,
            frac_remain=0.01,
            Lepsilon=0.001,
            min_ess=400,
            max_ncalls=None,
            max_num_improvement_loops=-1,
            min_num_live_points=400,
            cluster_num_live_points=40,
            step_sampler=None,
        )
        for equiv in bilby.core.sampler.base_sampler.NestedSampler.npoints_equiv_kwargs:
            new_kwargs = self.sampler.kwargs.copy()
            del new_kwargs['num_live_points']
            new_kwargs[equiv] = 123
            self.sampler.kwargs = new_kwargs
            self.sampler.kwargs["wrapped_params"] = None
            self.sampler.kwargs["derived_param_names"] = None
            self.sampler.kwargs["viz_callback"] = None
            self.assertDictEqual(expected, self.sampler.kwargs)


if __name__ == "__main__":
    unittest.main()
