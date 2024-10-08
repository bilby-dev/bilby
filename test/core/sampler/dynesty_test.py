import unittest
from copy import deepcopy

from attr import define
import bilby
import numpy as np
import parameterized
import bilby.core.sampler.dynesty
from bilby.core.sampler import dynesty_utils
from scipy.stats import gamma, ks_1samp, uniform, powerlaw
import shutil
import os


@define
class Dummy:
    u: np.ndarray
    axes: np.ndarray
    scale: float = 1
    rseed: float = 1234
    kwargs: dict = dict(walks=500, live=np.zeros((2, 4)), periodic=None, reflective=None)
    prior_transform: callable = lambda x: x
    loglikelihood: callable = lambda x: 0
    loglstar: float = -1


class DummyLikelihood(bilby.core.likelihood.Likelihood):
    """
    A trivial likelihood used for testing. Add some randomness so the likelihood
    isn't flat everywhere as that can cause issues for nested samplers.
    """

    def __init__(self):
        super().__init__(dict())

    def log_likelihood(self):
        return np.random.uniform(0, 0.01)


class TestDynesty(unittest.TestCase):
    def setUp(self):
        self.likelihood = DummyLikelihood()
        self.priors = bilby.core.prior.PriorDict(
            dict(a=bilby.core.prior.Uniform(0, 1), b=bilby.core.prior.Uniform(0, 1))
        )
        self.sampler = bilby.core.sampler.dynesty.Dynesty(
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
        """Only test the kwargs where we specify different defaults to dynesty"""
        expected = dict(
            sample="act-walk",
            facc=0.2,
            save_bounds=False,
            dlogz=0.1,
            bound="live",
            update_interval=600,
        )
        for key in expected:
            self.assertEqual(expected[key], self.sampler.kwargs[key])

    def test_translate_kwargs(self):
        expected = 1000
        for equiv in bilby.core.sampler.base_sampler.NestedSampler.npoints_equiv_kwargs:
            new_kwargs = deepcopy(self.sampler.kwargs)
            del new_kwargs["nlive"]
            new_kwargs[equiv] = expected
            self.sampler._translate_kwargs(new_kwargs)
            self.assertEqual(new_kwargs["nlive"], expected)

    def test_prior_boundary(self):
        self.priors["a"] = bilby.core.prior.Prior(boundary="periodic")
        self.priors["b"] = bilby.core.prior.Prior(boundary="reflective")
        self.priors["c"] = bilby.core.prior.Prior(boundary=None)
        self.priors["d"] = bilby.core.prior.Prior(boundary="reflective")
        self.priors["e"] = bilby.core.prior.Prior(boundary="periodic")
        self.sampler = bilby.core.sampler.dynesty.Dynesty(
            self.likelihood,
            self.priors,
            outdir="outdir",
            label="label",
            use_ratio=False,
            plot=False,
            skip_import_verification=True,
        )
        self.assertEqual([0, 4], self.sampler.kwargs["periodic"])
        self.assertEqual([1, 3], self.sampler.kwargs["reflective"])

    def test_run_test_runs(self):
        self.sampler._run_test()


def test_get_expected_outputs():
    label = "par0"
    outdir = os.path.join("some", "bilby_pipe", "dir")
    filenames, directories = bilby.core.sampler.dynesty.Dynesty.get_expected_outputs(
        outdir=outdir, label=label
    )
    assert len(filenames) == 2
    assert len(directories) == 0
    assert os.path.join(outdir, f"{label}_resume.pickle") in filenames
    assert os.path.join(outdir, f"{label}_dynesty.pickle") in filenames


class ProposalsTest(unittest.TestCase):

    def test_boundaries(self):
        inputs = np.array([0.1, 1.1, -1.3])
        expected = np.array([0.1, 0.1, 0.7])
        periodic = [1]
        reflective = [2]
        self.assertLess(max(abs(
            dynesty_utils.apply_boundaries_(inputs, periodic, reflective) - expected
        )), 1e-10)

    def test_boundaries_returns_none_outside_bound(self):
        inputs = np.array([0.1, 1.1, -1.3])
        self.assertIsNone(dynesty_utils.apply_boundaries_(inputs, None, None))

    def test_propose_volumetric(self):
        proposal_func = dynesty_utils.proposal_funcs["volumetric"]
        rng = np.random.default_rng(12345)
        axes = np.array([[2, 0], [0, 2]])
        start = np.zeros(4)
        new_samples = list()
        for _ in range(1000):
            new_samples.append(proposal_func(start, axes, 1, 4, 2, rng))
        new_samples = np.array(new_samples)
        self.assertGreater(ks_1samp(new_samples[:, 2:].flatten(), uniform(0, 1).cdf).pvalue, 0.01)
        self.assertGreater(ks_1samp(np.linalg.norm(new_samples[:, :2], axis=-1), powerlaw(2, scale=2).cdf).pvalue, 0.01)

    def test_propose_differential_evolution_mode_hopping(self):
        proposal_func = dynesty_utils.proposal_funcs["diff"]
        rng = np.random.default_rng(12345)
        live = np.array([[1, 1], [0, 0]])
        start = np.zeros(4)
        new_samples = list()
        for _ in range(1000):
            new_samples.append(proposal_func(start, live, 4, 2, rng, mix=0))
        new_samples = np.array(new_samples)
        self.assertGreater(ks_1samp(new_samples[:, 2:].flatten(), uniform(0, 1).cdf).pvalue, 0.01)
        self.assertLess(np.max(abs(new_samples[:, :2]) - np.array([1, 1])), 1e-10)

    @parameterized.parameterized.expand(((1,), (None,), (5,)))
    def test_propose_differential_evolution(self, scale):
        proposal_func = dynesty_utils.proposal_funcs["diff"]
        rng = np.random.default_rng(12345)
        live = np.array([[1, 1], [0, 0]])
        start = np.zeros(4)
        new_samples = list()
        for _ in range(1000):
            new_samples.append(proposal_func(start, live, 4, 2, rng, mix=1, scale=scale))
        new_samples = np.array(new_samples)
        if scale is None:
            scale = 1.17
        self.assertGreater(ks_1samp(new_samples[:, 2:].flatten(), uniform(0, 1).cdf).pvalue, 0.01)
        self.assertGreater(ks_1samp(np.abs(new_samples[:, :2].flatten()), gamma(4, scale=scale / 4).cdf).pvalue, 0.01)

    def test_get_proposal_kwargs_diff(self):
        args = Dummy(u=-np.ones(4), axes=np.zeros((2, 2)), scale=4)
        dynesty_utils._SamplingContainer.proposals = ["diff"]
        proposals, common, specific = dynesty_utils._get_proposal_kwargs(args)
        del common["rstate"]
        self.assertTrue(np.array_equal(proposals, np.array(["diff"] * args.kwargs["walks"])))
        self.assertDictEqual(common, dict(n=len(args.u), n_cluster=len(args.axes)))
        assert np.array_equal(args.kwargs["live"][:1, :2], specific["diff"]["live"])
        del specific["diff"]["live"]
        self.assertDictEqual(specific, dict(diff=dict(mix=0.5, scale=1.19)))

    def test_get_proposal_kwargs_volumetric(self):
        args = Dummy(u=-np.ones(4), axes=np.zeros((2, 2)), scale=4)
        dynesty_utils._SamplingContainer.proposals = ["volumetric"]
        proposals, common, specific = dynesty_utils._get_proposal_kwargs(args)
        del common["rstate"]
        self.assertTrue(np.array_equal(proposals, np.array(["volumetric"] * args.kwargs["walks"])))
        self.assertDictEqual(common, dict(n=len(args.u), n_cluster=len(args.axes)))
        self.assertDictEqual(specific, dict(volumetric=dict(axes=args.axes, scale=args.scale)))

    def test_proposal_functions_run(self):
        args = Dummy(u=np.ones(4) / 2, axes=np.ones((2, 2)))
        args.kwargs["live"][0] += 1
        for proposals in [
            ["diff"],
            ["volumetric"],
            ["diff", "volumetric"],
            {"diff": 5, "volumetric": 1},
        ]:
            dynesty_utils._SamplingContainer.proposals = proposals
            dynesty_utils.FixedRWalk()(args)
            dynesty_utils.AcceptanceTrackingRWalk()(args)
            dynesty_utils.ACTTrackingRWalk()(args)


@parameterized.parameterized_class(("kind", ), [("live",), ("live-multi",)])
class TestCustomSampler(unittest.TestCase):
    def setUp(self):
        if self.kind == "live":
            cls = dynesty_utils.LivePointSampler
        elif self.kind == "live-multi":
            cls = dynesty_utils.MultiEllipsoidLivePointSampler
        else:
            raise ValueError(f"Unknown sampler class {self.kind}")

        self.sampler = cls(
            loglikelihood=lambda x: 1,
            prior_transform=lambda x: x,
            ndim=4,
            live_points=(np.zeros((1000, 4)), np.zeros((1000, 4)), np.zeros(1000)),
            update_interval=None,
            first_update=dict(),
            queue_size=1,
            pool=None,
            use_pool=dict(),
            ncdim=2,
            method="rwalk",
            rstate=np.random.default_rng(1234),
            kwargs=dict(walks=100)
        )
        self.blob = dict(accept=5, reject=35, scale=1)

    def tearDown(self):
        dynesty_utils._SamplingContainer.proposals = None

    def test_update_with_update(self):
        """
        If there is only one element left in the cache for ACT tracking we
        need to add the flag to rebuild the cache. After rebuilding, this is
        then reset to False.
        """
        self.sampler.rebuild = False
        self.blob["remaining"] = 1
        self.sampler.update_user(blob=self.blob, update=True)
        self.assertEqual(self.sampler.kwargs["rebuild"], True)
        self.blob["remaining"] = 2
        self.sampler.update_user(blob=self.blob, update=True)
        self.assertEqual(self.sampler.kwargs["rebuild"], False)

    def test_diff_update(self):
        """
        Sampler updates do different things depending on whether ellipsoid
        bounding is used. For the `live-multi` case we reproduce the dynesty
        behaviour. For the `live` case, we overwrite the scale attribute to
        store acceptance information.
        """
        dynesty_utils._SamplingContainer.proposals = ["diff"]
        dynesty_utils._SamplingContainer.maxmcmc = 10000
        dynesty_utils._SamplingContainer.naccept = 10
        self.sampler.update_user(blob=self.blob, update=False)
        if self.kind == "live":
            self.assertEqual(self.sampler.scale, self.blob["accept"])
        else:
            self.assertEqual(self.sampler.scale, self.blob["scale"])
        self.assertEqual(self.sampler.walks, 101)


class TestEstimateNMCMC(unittest.TestCase):
    def test_converges_to_correct_value(self):
        """
        NMCMC convergence should convergence to
        safety * (2 / accept_ratio - 1)
        """
        sampler = dynesty_utils.AcceptanceTrackingRWalk()
        dynesty_utils.AcceptanceTrackingRWalk.old_act = None
        for _ in range(10):
            accept_ratio = np.random.uniform()
            safety = np.random.randint(2, 8)
            expected = safety * (2 / accept_ratio - 1)
            for _ in range(1000):
                estimated = sampler.estimate_nmcmc(
                    accept_ratio=accept_ratio,
                    safety=safety,
                    tau=1000,
                )
            self.assertAlmostEqual(estimated, expected)


class TestReproducibility(unittest.TestCase):

    @staticmethod
    def model(x, m, c):
        return m * x + c

    def setUp(self):
        bilby.core.utils.random.seed(42)
        bilby.core.utils.command_line_args.bilby_test_mode = False
        rng = bilby.core.utils.random.rng
        self.x = np.linspace(0, 1, 11)
        self.injection_parameters = dict(m=0.5, c=0.2)
        self.sigma = 0.1
        self.y = self.model(self.x, **self.injection_parameters) + rng.normal(
            0, self.sigma, len(self.x)
        )
        self.likelihood = bilby.likelihood.GaussianLikelihood(
            self.x, self.y, self.model, self.sigma
        )

        self.priors = bilby.core.prior.PriorDict()
        self.priors["m"] = bilby.core.prior.Uniform(0, 5, boundary="periodic")
        self.priors["c"] = bilby.core.prior.Uniform(-2, 2, boundary="reflective")
        # Evaluate prior once to ensure normalization constant have been set
        theta = self.priors.sample()
        self.priors.ln_prob(theta)
        self._remove_tree()
        bilby.core.utils.check_directory_exists_and_if_not_mkdir("outdir")

    def tearDown(self):
        del self.likelihood
        del self.priors
        bilby.core.utils.command_line_args.bilby_test_mode = False
        self._remove_tree()

    def _remove_tree(self):
        try:
            shutil.rmtree("outdir")
        except OSError:
            pass

    def _run_sampler(self, **kwargs):
        bilby.core.utils.random.seed(42)
        return bilby.run_sampler(
            likelihood=self.likelihood,
            priors=self.priors,
            sampler="dynesty",
            save=False,
            resume=False,
            dlogz=1.0,
            nlive=20,
            **kwargs,
        )

    def test_reproducibility_seed(self):
        res0 = self._run_sampler(seed=1234)
        res1 = self._run_sampler(seed=1234)
        assert res0.log_evidence == res1.log_evidence

    def test_reproducibility_state(self):
        rstate = np.random.default_rng(1234)
        res0 = self._run_sampler(rstate=rstate)
        rstate = np.random.default_rng(1234)
        res1 = self._run_sampler(rstate=rstate)
        assert res0.log_evidence == res1.log_evidence

    def test_reproducibility_state_and_seed(self):
        rstate = np.random.default_rng(1234)
        res0 = self._run_sampler(rstate=rstate)
        res1 = self._run_sampler(seed=1234)
        assert res0.log_evidence == res1.log_evidence


if __name__ == "__main__":
    unittest.main()
