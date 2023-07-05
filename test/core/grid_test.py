import unittest
import numpy as np
import shutil
import os
from scipy.stats import multivariate_normal

import bilby


# set 2D multivariate Gaussian likelihood
class MultiGaussian(bilby.Likelihood):
    def __init__(self, mean, cov):
        super(MultiGaussian, self).__init__(parameters=dict())
        self.cov = np.array(cov)
        self.mean = np.array(mean)
        self.sigma = np.sqrt(np.diag(self.cov))
        self.pdf = multivariate_normal(mean=self.mean, cov=self.cov)

    @property
    def dim(self):
        return len(self.cov[0])

    def log_likelihood(self):
        x = np.array([self.parameters["x{0}".format(i)] for i in range(self.dim)])
        return self.pdf.logpdf(x)


class TestGrid(unittest.TestCase):
    def setUp(self):
        bilby.core.utils.random.seed(7)

        # set 2D multivariate Gaussian (zero mean, unit variance)
        self.mus = [0.0, 0.0]
        self.cov = [[1.0, 0.0], [0.0, 1.0]]
        dim = len(self.mus)
        self.likelihood = MultiGaussian(self.mus, self.cov)

        # set priors out to +/- 5 sigma
        self.priors = bilby.core.prior.PriorDict()
        self.priors.update(
            {
                "x{0}".format(i): bilby.core.prior.Uniform(-5, 5, "x{0}".format(i))
                for i in range(dim)
            }
        )

        # expected evidence integral should be (1/V) where V is the prior volume
        log_prior_vol = np.sum(
            np.log(
                [prior.maximum - prior.minimum for key, prior in self.priors.items()]
            )
        )
        self.expected_ln_evidence = -log_prior_vol

        self.grid_size = 100

        grid = bilby.core.grid.Grid(
            label="label",
            outdir="outdir",
            priors=self.priors,
            grid_size=self.grid_size,
            likelihood=self.likelihood,
            save=True,
        )

        self.grid = grid
        pass

    def tearDown(self):
        bilby.utils.command_line_args.bilby_test_mode = True
        try:
            shutil.rmtree(self.grid.outdir)
        except OSError:
            pass
        del self.grid
        pass

    def test_grid_file_name_default(self):
        outdir = "outdir"
        label = "label"
        self.assertEqual(
            bilby.core.grid.grid_file_name(outdir, label),
            "{}/{}_grid.json".format(outdir, label),
        )
        self.assertEqual(
            bilby.core.grid.grid_file_name(outdir, label, True),
            "{}/{}_grid.json.gz".format(outdir, label),
        )

    def test_fail_save_and_load(self):
        with self.assertRaises(ValueError):
            bilby.core.grid.Grid.read()

        with self.assertRaises(IOError):
            bilby.core.grid.Grid.read(filename="not/a/file.json")

    def test_fail_marginalize(self):
        with self.assertRaises(TypeError):
            self.grid.marginalize_posterior(parameters=2.4)

        with self.assertRaises(TypeError):
            self.grid.marginalize_posterior(not_parameters=4.7)

        with self.assertRaises(ValueError):
            self.grid.marginalize_posterior(parameters="jkgsd")

    def test_parameter_names(self):
        self.assertListEqual(list(self.priors.keys()), self.grid.parameter_names)
        self.assertEqual(2, self.grid.n_dims)

    def test_no_marginalization(self):
        # test arrays are the same if no parameters are given to marginalize
        # over
        self.assertTrue(np.array_equal(
            self.grid.ln_likelihood,
            self.grid.marginalize_ln_likelihood(not_parameters=self.grid.parameter_names)))

    def test_marginalization_shapes(self):
        self.assertEqual(0, len(self.grid.marginalize_ln_likelihood().shape))
        marg1 = self.grid.marginalize_ln_likelihood(parameters=self.grid.parameter_names[0])
        marg2 = self.grid.marginalize_ln_likelihood(parameters=self.grid.parameter_names[1])
        self.assertTupleEqual((self.grid_size,), marg1.shape)
        self.assertTupleEqual((self.grid_size,), marg2.shape)
        self.assertTupleEqual((self.grid_size, self.grid_size), self.grid.ln_likelihood.shape)
        self.assertTupleEqual((self.grid_size, self.grid_size), self.grid.ln_posterior.shape)

    def test_marginalization_opposite(self):
        self.assertTrue(np.array_equal(
            self.grid.marginalize_ln_likelihood(parameters=self.grid.parameter_names[0]),
            self.grid.marginalize_ln_likelihood(not_parameters=self.grid.parameter_names[1])))

        self.assertTrue(np.array_equal(
            self.grid.marginalize_ln_likelihood(parameters=self.grid.parameter_names[1]),
            self.grid.marginalize_ln_likelihood(not_parameters=self.grid.parameter_names[0])))

    def test_max_marginalized_likelihood(self):
        # marginalised likelihoods should have max values of 1 (as they are not
        # properly normalised)
        self.assertEqual(1.0, self.grid.marginalize_likelihood(self.grid.parameter_names[0]).max())
        self.assertEqual(1.0, self.grid.marginalize_likelihood(self.grid.parameter_names[1]).max())

    def test_ln_evidence(self):
        self.assertAlmostEqual(self.expected_ln_evidence, self.grid.ln_evidence, places=5)

    def test_fail_grid_size(self):
        with self.assertRaises(TypeError):
            bilby.core.grid.Grid(
                label="label",
                outdir="outdir",
                priors=self.priors,
                grid_size=2.3,
                likelihood=self.likelihood,
                save=True,
            )

    def test_mesh_grid(self):
        self.assertTupleEqual((self.grid_size, self.grid_size), self.grid.mesh_grid[0].shape)
        self.assertEqual(self.priors[self.grid.parameter_names[0]].minimum, self.grid.mesh_grid[0][0, 0])
        self.assertEqual(self.priors[self.grid.parameter_names[1]].maximum, self.grid.mesh_grid[0][-1, -1])

    def test_grid_integer_points(self):
        n_points = [10, 20]
        grid = bilby.core.grid.Grid(
            label="label",
            outdir="outdir",
            priors=self.priors,
            grid_size=n_points,
            likelihood=self.likelihood
        )

        self.assertTupleEqual(tuple(n_points), grid.mesh_grid[0].shape)
        self.assertEqual(grid.mesh_grid[0][0, 0], self.priors[self.grid.parameter_names[0]].minimum)
        self.assertEqual(grid.mesh_grid[0][-1, -1], self.priors[self.grid.parameter_names[1]].maximum)

    def test_grid_dict_points(self):
        n_points = {"x0": 15, "x1": 18}
        grid = bilby.core.grid.Grid(
            label="label",
            outdir="outdir",
            priors=self.priors,
            grid_size=n_points,
            likelihood=self.likelihood
        )
        self.assertTupleEqual((n_points["x0"], n_points["x1"]), grid.mesh_grid[0].shape)
        self.assertEqual(grid.mesh_grid[0][0, 0], self.priors[self.grid.parameter_names[0]].minimum)
        self.assertEqual(grid.mesh_grid[0][-1, -1], self.priors[self.grid.parameter_names[1]].maximum)

    def test_grid_from_array(self):
        x0s = np.linspace(self.priors["x0"].minimum, self.priors["x0"].maximum, 13)
        x1s = np.linspace(self.priors["x0"].minimum, self.priors["x0"].maximum, 14)
        n_points = {"x0": x0s, "x1": x1s}

        grid = bilby.core.grid.Grid(
            label="label",
            outdir="outdir",
            priors=self.priors,
            grid_size=n_points,
            likelihood=self.likelihood,
        )

        self.assertTupleEqual((len(x0s), len(x1s)), grid.mesh_grid[0].shape)
        self.assertEqual(grid.mesh_grid[0][0, 0], self.priors[self.grid.parameter_names[0]].minimum)
        self.assertEqual(grid.mesh_grid[0][-1, -1], self.priors[self.grid.parameter_names[1]].maximum)

        self.assertTrue(np.array_equal(grid.sample_points["x0"], x0s))
        self.assertTrue(np.array_equal(grid.sample_points["x1"], x1s))

    def test_save_and_load_from_filename(self):
        filename = os.path.join("outdir", "test_output.json")
        self.grid.save_to_file(filename=filename)
        new_grid = bilby.core.grid.Grid.read(filename=filename)

        self.assertListEqual(new_grid.parameter_names, self.grid.parameter_names)
        self.assertEqual(new_grid.n_dims, self.grid.n_dims)
        self.assertTrue(np.array_equal(new_grid.mesh_grid[0], self.grid.mesh_grid[0]))
        for par in new_grid.parameter_names:
            self.assertTrue(np.array_equal(new_grid.sample_points[par], self.grid.sample_points[par]))
        self.assertEqual(new_grid.ln_evidence, self.grid.ln_evidence)
        self.assertTrue(np.array_equal(new_grid.ln_likelihood, self.grid.ln_likelihood))
        self.assertTrue(np.array_equal(new_grid.ln_posterior, self.grid.ln_posterior))

    def test_save_and_load_from_outdir_label(self):
        self.grid.save_to_file(overwrite=True, outdir="outdir")
        new_grid = bilby.core.grid.Grid.read(outdir="outdir", label="label")

        self.assertListEqual(self.grid.parameter_names, new_grid.parameter_names)
        self.assertEqual(self.grid.n_dims, new_grid.n_dims)
        self.assertTrue(np.array_equal(new_grid.mesh_grid[0], self.grid.mesh_grid[0]))
        for par in new_grid.parameter_names:
            self.assertTrue(np.array_equal(
                new_grid.sample_points[par], self.grid.sample_points[par])
            )
        self.assertEqual(self.grid.ln_evidence, new_grid.ln_evidence)
        self.assertTrue(np.array_equal(self.grid.ln_likelihood, new_grid.ln_likelihood))
        self.assertTrue(np.array_equal(self.grid.ln_posterior, new_grid.ln_posterior))
        del new_grid

    def test_save_and_load_gzip(self):
        filename = os.path.join("outdir", "test_output.json.gz")
        self.grid.save_to_file(filename=filename)
        new_grid = bilby.core.grid.Grid.read(filename=filename)

        self.assertListEqual(self.grid.parameter_names, new_grid.parameter_names)
        self.assertEqual(self.grid.n_dims, new_grid.n_dims)
        self.assertTrue(np.array_equal(self.grid.mesh_grid[0], new_grid.mesh_grid[0]))
        for par in new_grid.parameter_names:
            self.assertTrue(np.array_equal(self.grid.sample_points[par], new_grid.sample_points[par]))
        self.assertEqual(self.grid.ln_evidence, new_grid.ln_evidence)
        self.assertTrue(np.array_equal(self.grid.ln_likelihood, new_grid.ln_likelihood))
        self.assertTrue(np.array_equal(self.grid.ln_posterior, new_grid.ln_posterior))

        self.grid.save_to_file(overwrite=True, outdir="outdir", gzip=True)
        _ = bilby.core.grid.Grid.read(outdir="outdir", label="label", gzip=True)
