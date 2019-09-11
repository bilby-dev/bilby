from __future__ import absolute_import, division

import unittest
import numpy as np
import pandas as pd
import shutil
import os
import json
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
        np.random.seed(7)
        
        # set 2D multivariate Gaussian (zero mean, unit variance)
        self.mus = [0., 0.]
        self.cov = [[1., 0.], [0., 1.]]
        dim = len(self.mus)
        self.likelihood = MultiGaussian(self.mus, self.cov)
        
        # set priors out to +/- 5 sigma
        self.priors = bilby.core.prior.PriorDict()
        self.priors.update(
            {"x{0}".format(i): bilby.core.prior.Uniform(-5, 5, "x{0}".format(i)) for i in range(dim)}
        )

        # expected evidence integral should be (1/V) where V is the prior volume
        log_prior_vol = np.sum(np.log([prior.maximum - prior.minimum for key, prior in self.priors.items()]))
        self.expected_ln_evidence = -log_prior_vol

        self.grid_size = 100

        grid = bilby.core.grid.Grid(
            label='label', outdir='outdir', priors=self.priors,
            grid_size=self.grid_size, likelihood=self.likelihood,
            save=True
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
        outdir = 'outdir'
        label = 'label'
        self.assertEqual(bilby.core.grid.grid_file_name(outdir, label),
                         '{}/{}_grid.json'.format(outdir, label))
        self.assertEqual(bilby.core.grid.grid_file_name(outdir, label, True),
                         '{}/{}_grid.json.gz'.format(outdir, label))

    def test_fail_save_and_load(self):
        with self.assertRaises(ValueError):
            bilby.core.grid.Grid.read()

        with self.assertRaises(IOError):
            bilby.core.grid.Grid.read(filename='not/a/file.json')

    def test_fail_marginalize(self):
        with self.assertRaises(TypeError):
            self.grid.marginalize_posterior(parameters=2.4)
        
        with self.assertRaises(TypeError):
            self.grid.marginalize_posterior(not_parameters=4.7)
        
        with self.assertRaises(ValueError):
            self.grid.marginalize_posterior(parameters='jkgsd')

    def test_parameter_names(self):
        assert list(self.priors.keys()) == self.grid.parameter_names
        assert self.grid.n_dims == 2

    def test_no_marginalization(self):
        # test arrays are the same if no parameters are given to marginalize
        # over
        assert np.array_equal(self.grid.ln_likelihood,
                              self.grid.marginalize_ln_likelihood(not_parameters=self.grid.parameter_names))
    
    def test_marginalization_shapes(self):
        assert len(self.grid.marginalize_ln_likelihood().shape) == 0
        
        marg1 = self.grid.marginalize_ln_likelihood(parameters=self.grid.parameter_names[0])
        assert marg1.shape == (self.grid_size,)

        marg2 = self.grid.marginalize_ln_likelihood(parameters=self.grid.parameter_names[1])
        assert marg2.shape == (self.grid_size,)

        assert self.grid.ln_likelihood.shape == (self.grid_size, self.grid_size)
        assert self.grid.ln_posterior.shape == (self.grid_size, self.grid_size)

    def test_marginalization_opposite(self):
        assert np.array_equal(self.grid.marginalize_ln_likelihood(parameters=self.grid.parameter_names[0]),
                              self.grid.marginalize_ln_likelihood(not_parameters=self.grid.parameter_names[1]))
        assert np.array_equal(self.grid.marginalize_ln_likelihood(parameters=self.grid.parameter_names[1]),
                              self.grid.marginalize_ln_likelihood(not_parameters=self.grid.parameter_names[0]))
    
    def test_max_marginalized_likelihood(self):
        # marginalised likelihoods should have max values of 1 (as they are not
        # properly normalised)
        assert self.grid.marginalize_likelihood(self.grid.parameter_names[0]).max() == 1.
        assert self.grid.marginalize_likelihood(self.grid.parameter_names[1]).max() == 1.

    def test_ln_evidence(self):
        assert np.isclose(self.grid.ln_evidence, self.expected_ln_evidence)

    def test_fail_grid_size(self):
        with self.assertRaises(TypeError):
            grid = bilby.core.grid.Grid(
                label='label', outdir='outdir', priors=self.priors,
                grid_size=2.3, likelihood=self.likelihood,
                save=True
            )

    def test_mesh_grid(self):
        assert self.grid.mesh_grid[0].shape == (self.grid_size, self.grid_size)
        assert self.grid.mesh_grid[0][0,0] == self.priors[self.grid.parameter_names[0]].minimum
        assert self.grid.mesh_grid[0][-1,-1] == self.priors[self.grid.parameter_names[1]].maximum

    def test_different_grids(self):
        npoints = [10, 20]

        grid = bilby.core.grid.Grid(
            label='label', outdir='outdir', priors=self.priors,
            grid_size=npoints, likelihood=self.likelihood
        )

        assert grid.mesh_grid[0].shape == tuple(npoints)
        assert grid.mesh_grid[0][0,0] == self.priors[self.grid.parameter_names[0]].minimum
        assert grid.mesh_grid[0][-1,-1] == self.priors[self.grid.parameter_names[1]].maximum

        del grid

        npoints = {'x0': 15, 'x1': 18}

        grid = bilby.core.grid.Grid(
            label='label', outdir='outdir', priors=self.priors,
            grid_size=npoints, likelihood=self.likelihood
        )

        assert grid.mesh_grid[0].shape == (npoints['x0'], npoints['x1'])
        assert grid.mesh_grid[0][0,0] == self.priors[self.grid.parameter_names[0]].minimum
        assert grid.mesh_grid[0][-1,-1] == self.priors[self.grid.parameter_names[1]].maximum

        del grid

        x0s = np.linspace(self.priors['x0'].minimum, self.priors['x0'].maximum, 13)
        x1s = np.linspace(self.priors['x0'].minimum, self.priors['x0'].maximum, 14)
        npoints = {'x0': x0s,
                   'x1': x1s}

        grid = bilby.core.grid.Grid(
            label='label', outdir='outdir', priors=self.priors,
            grid_size=npoints, likelihood=self.likelihood
        )

        assert grid.mesh_grid[0].shape == (len(x0s), len(x1s))
        assert grid.mesh_grid[0][0,0] == self.priors[self.grid.parameter_names[0]].minimum
        assert grid.mesh_grid[0][-1,-1] == self.priors[self.grid.parameter_names[1]].maximum
        assert np.array_equal(grid.sample_points['x0'], x0s)
        assert np.array_equal(grid.sample_points['x1'], x1s)

    def test_save_and_load(self):
        filename = os.path.join('outdir', 'test_output.json')

        self.grid.save_to_file(filename=filename)

        # load file
        newgrid = bilby.core.grid.Grid.read(filename=filename)

        assert newgrid.parameter_names == self.grid.parameter_names
        assert newgrid.n_dims == self.grid.n_dims
        assert np.array_equal(newgrid.mesh_grid[0], self.grid.mesh_grid[0])
        for par in newgrid.parameter_names:
            assert np.array_equal(newgrid.sample_points[par], self.grid.sample_points[par])
        assert newgrid.ln_evidence == self.grid.ln_evidence
        assert np.array_equal(newgrid.ln_likelihood, self.grid.ln_likelihood)
        assert np.array_equal(newgrid.ln_posterior, self.grid.ln_posterior)

        del newgrid

        self.grid.save_to_file(overwrite=True, outdir='outdir')

        # load file
        newgrid = bilby.core.grid.Grid.read(outdir='outdir',
                                            label='label')

        assert newgrid.parameter_names == self.grid.parameter_names
        assert newgrid.n_dims == self.grid.n_dims
        assert np.array_equal(newgrid.mesh_grid[0], self.grid.mesh_grid[0])
        for par in newgrid.parameter_names:
            assert np.array_equal(newgrid.sample_points[par], self.grid.sample_points[par])
        assert newgrid.ln_evidence == self.grid.ln_evidence
        assert np.array_equal(newgrid.ln_likelihood, self.grid.ln_likelihood)
        assert np.array_equal(newgrid.ln_posterior, self.grid.ln_posterior)

        del newgrid

    def test_save_and_load_gzip(self):
        filename = os.path.join('outdir', 'test_output.json.gz')

        self.grid.save_to_file(filename=filename)

        # load file
        newgrid = bilby.core.grid.Grid.read(filename=filename)

        assert newgrid.parameter_names == self.grid.parameter_names
        assert newgrid.n_dims == self.grid.n_dims
        assert np.array_equal(newgrid.mesh_grid[0], self.grid.mesh_grid[0])
        for par in newgrid.parameter_names:
            assert np.array_equal(newgrid.sample_points[par], self.grid.sample_points[par])
        assert newgrid.ln_evidence == self.grid.ln_evidence
        assert np.array_equal(newgrid.ln_likelihood, self.grid.ln_likelihood)
        assert np.array_equal(newgrid.ln_posterior, self.grid.ln_posterior)

        del newgrid

        self.grid.save_to_file(overwrite=True, outdir='outdir',
                               gzip=True)

        # load file
        newgrid = bilby.core.grid.Grid.read(outdir='outdir', label='label',
                                            gzip=True)

        assert newgrid.parameter_names == self.grid.parameter_names
        assert newgrid.n_dims == self.grid.n_dims
        assert np.array_equal(newgrid.mesh_grid[0], self.grid.mesh_grid[0])
        for par in newgrid.parameter_names:
            assert np.array_equal(newgrid.sample_points[par], self.grid.sample_points[par])
        assert newgrid.ln_evidence == self.grid.ln_evidence
        assert np.array_equal(newgrid.ln_likelihood, self.grid.ln_likelihood)
        assert np.array_equal(newgrid.ln_posterior, self.grid.ln_posterior)

        del newgrid
