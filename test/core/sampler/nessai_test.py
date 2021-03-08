import unittest

from mock import MagicMock, patch, mock_open

import bilby


class TestNessai(unittest.TestCase):
    maxDiff = None

    def setUp(self):
        self.likelihood = MagicMock()
        self.priors = bilby.core.prior.PriorDict(
            dict(a=bilby.core.prior.Uniform(0, 1), b=bilby.core.prior.Uniform(0, 1))
        )
        self.sampler = bilby.core.sampler.Nessai(
            self.likelihood,
            self.priors,
            outdir="outdir",
            label="label",
            use_ratio=False,
            plot=False,
            skip_import_verification=True,
        )
        self.expected = dict(
            output="outdir/label_nessai/",
            nlive=1000,
            stopping=0.1,
            resume=True,
            max_iteration=None,
            checkpointing=True,
            seed=1234,
            acceptance_threshold=0.01,
            analytic_priors=False,
            maximum_uninformed=1000,
            uninformed_proposal=None,
            uninformed_proposal_kwargs=None,
            flow_class=None,
            flow_config=None,
            training_frequency=None,
            reset_weights=False,
            reset_permutations=False,
            reset_acceptance=False,
            train_on_empty=True,
            cooldown=100,
            memory=False,
            poolsize=None,
            drawsize=None,
            max_poolsize_scale=10,
            update_poolsize=False,
            latent_prior='truncated_gaussian',
            draw_latent_kwargs=None,
            compute_radius_with_all=False,
            min_radius=False,
            max_radius=50,
            check_acceptance=False,
            fuzz=1.0,
            expansion_fraction=1.0,
            rescale_parameters=True,
            rescale_bounds=[-1, 1],
            update_bounds=False,
            boundary_inversion=False,
            inversion_type='split', detect_edges=False,
            detect_edges_kwargs=None,
            reparameterisations=None,
            n_pool=None,
            max_threads=1,
            pytorch_threads=None,
            plot=False,
            proposal_plots=False
        )

    def tearDown(self):
        del self.likelihood
        del self.priors
        del self.sampler
        del self.expected

    def test_default_kwargs(self):
        expected = self.expected.copy()
        self.assertDictEqual(expected, self.sampler.kwargs)

    def test_translate_kwargs_nlive(self):
        expected = self.expected.copy()
        for equiv in bilby.core.sampler.base_sampler.NestedSampler.npoints_equiv_kwargs:
            new_kwargs = self.sampler.kwargs.copy()
            del new_kwargs["nlive"]
            new_kwargs[equiv] = 1000
            self.sampler.kwargs = new_kwargs
            self.assertDictEqual(expected, self.sampler.kwargs)

    def test_translate_kwargs_npool(self):
        expected = self.expected.copy()
        expected["n_pool"] = None
        for equiv in bilby.core.sampler.base_sampler.NestedSampler.npool_equiv_kwargs:
            new_kwargs = self.sampler.kwargs.copy()
            del new_kwargs["n_pool"]
            new_kwargs[equiv] = None
            self.sampler.kwargs = new_kwargs
            self.assertDictEqual(expected, self.sampler.kwargs)

    def test_translate_kwargs_seed(self):
        expected = self.expected.copy()
        expected["seed"] = 150914
        for equiv in bilby.core.sampler.nessai.Nessai.seed_equiv_kwargs:
            new_kwargs = self.sampler.kwargs.copy()
            del new_kwargs["seed"]
            new_kwargs[equiv] = 150914
            self.sampler.kwargs = new_kwargs
            self.assertDictEqual(expected, self.sampler.kwargs)

    def test_npool_max_threads(self):
        expected = self.expected.copy()
        expected["n_pool"] = None
        new_kwargs = self.sampler.kwargs.copy()
        new_kwargs["n_pool"] = 1
        self.sampler.kwargs = new_kwargs
        self.assertDictEqual(expected, self.sampler.kwargs)

    @patch("builtins.open", mock_open(read_data='{"nlive": 2000}'))
    def test_update_from_config_file(self):
        expected = self.expected.copy()
        expected["nlive"] = 2000
        new_kwargs = self.expected.copy()
        new_kwargs["config_file"] = "config_file.json"
        self.sampler.kwargs = new_kwargs
        self.assertDictEqual(expected, self.sampler.kwargs)


if __name__ == "__main__":
    unittest.main()
