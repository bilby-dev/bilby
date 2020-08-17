import unittest

import numpy as np
from mock import MagicMock

import bilby


class TestDynesty(unittest.TestCase):
    def setUp(self):
        self.likelihood = MagicMock()
        self.priors = bilby.core.prior.PriorDict(
            dict(a=bilby.core.prior.Uniform(0, 1), b=bilby.core.prior.Uniform(0, 1))
        )
        self.sampler = bilby.core.sampler.Dynesty(
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

    def test_default_kwargs(self):
        expected = dict(
            bound="multi",
            sample="rwalk",
            periodic=None,
            reflective=None,
            verbose=True,
            check_point_delta_t=600,
            nlive=1000,
            first_update=None,
            npdim=None,
            rstate=None,
            queue_size=1,
            pool=None,
            use_pool=None,
            live_points=None,
            logl_args=None,
            logl_kwargs=None,
            ptform_args=None,
            ptform_kwargs=None,
            enlarge=1.5,
            bootstrap=None,
            vol_dec=0.5,
            vol_check=8.0,
            facc=0.2,
            slices=5,
            dlogz=0.1,
            maxiter=None,
            maxcall=None,
            logl_max=np.inf,
            add_live=True,
            print_progress=True,
            save_bounds=False,
            walks=100,
            update_interval=600,
            print_func="func",
            n_effective=None,
            maxmcmc=5000,
            nact=5,
        )
        self.sampler.kwargs[
            "print_func"
        ] = "func"  # set this manually as this is not testable otherwise
        # DictEqual can't handle lists so we check these separately
        self.assertEqual([], self.sampler.kwargs["periodic"])
        self.assertEqual([], self.sampler.kwargs["reflective"])
        self.sampler.kwargs["periodic"] = expected["periodic"]
        self.sampler.kwargs["reflective"] = expected["reflective"]
        for key in self.sampler.kwargs.keys():
            print(
                "key={}, expected={}, actual={}".format(
                    key, expected[key], self.sampler.kwargs[key]
                )
            )
        self.assertDictEqual(expected, self.sampler.kwargs)

    def test_translate_kwargs(self):
        expected = dict(
            bound="multi",
            sample="rwalk",
            periodic=[],
            reflective=[],
            verbose=True,
            check_point_delta_t=600,
            nlive=1000,
            first_update=None,
            npdim=None,
            rstate=None,
            queue_size=1,
            pool=None,
            use_pool=None,
            live_points=None,
            logl_args=None,
            logl_kwargs=None,
            ptform_args=None,
            ptform_kwargs=None,
            enlarge=1.5,
            bootstrap=None,
            vol_dec=0.5,
            vol_check=8.0,
            facc=0.2,
            slices=5,
            dlogz=0.1,
            maxiter=None,
            maxcall=None,
            logl_max=np.inf,
            add_live=True,
            print_progress=True,
            save_bounds=False,
            walks=100,
            update_interval=600,
            print_func="func",
            n_effective=None,
            maxmcmc=5000,
            nact=5,
        )

        for equiv in bilby.core.sampler.base_sampler.NestedSampler.npoints_equiv_kwargs:
            new_kwargs = self.sampler.kwargs.copy()
            del new_kwargs["nlive"]
            new_kwargs[equiv] = 1000
            self.sampler.kwargs = new_kwargs
            self.sampler.kwargs[
                "print_func"
            ] = "func"  # set this manually as this is not testable otherwise
            self.assertDictEqual(expected, self.sampler.kwargs)

    def test_prior_boundary(self):
        self.priors["a"] = bilby.core.prior.Prior(boundary="periodic")
        self.priors["b"] = bilby.core.prior.Prior(boundary="reflective")
        self.priors["c"] = bilby.core.prior.Prior(boundary=None)
        self.priors["d"] = bilby.core.prior.Prior(boundary="reflective")
        self.priors["e"] = bilby.core.prior.Prior(boundary="periodic")
        self.sampler = bilby.core.sampler.Dynesty(
            self.likelihood,
            self.priors,
            outdir="outdir",
            label="label",
            use_ratio=False,
            plot=False,
            skip_import_verification=True,
        )
        self.assertEqual([0, 4], self.sampler.kwargs["periodic"])
        self.assertEqual(self.sampler._periodic, self.sampler.kwargs["periodic"])
        self.assertEqual([1, 3], self.sampler.kwargs["reflective"])
        self.assertEqual(self.sampler._reflective, self.sampler.kwargs["reflective"])


if __name__ == "__main__":
    unittest.main()
