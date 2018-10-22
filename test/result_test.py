from __future__ import absolute_import, division

import bilby
import unittest
import numpy as np
import pandas as pd
import shutil


class TestResult(unittest.TestCase):

    def setUp(self):
        bilby.utils.command_line_args.test = False
        result = bilby.core.result.Result()
        test_directory = 'test_directory'
        result.outdir = test_directory
        result.label = 'test'

        N = 100
        posterior = pd.DataFrame(dict(x=np.random.normal(0, 1, N),
                                      y=np.random.normal(0, 1, N)))
        result.search_parameter_keys = ['x', 'y']
        result.parameter_labels_with_unit = ['x', 'y']
        result.posterior = posterior
        self.result = result
        pass

    def tearDown(self):
        bilby.utils.command_line_args.test = True
        try:
            shutil.rmtree(self.result.outdir)
        except OSError:
            pass
        del self.result
        pass

    def test_plot_corner(self):
        self.result.injection_parameters = dict(x=0.8, y=1.1)
        self.result.plot_corner()
        self.result.plot_corner(parameters=['x', 'y'])
        self.result.plot_corner(parameters=['x', 'y'], truths=[1, 1])
        self.result.plot_corner(parameters=dict(x=1, y=1))
        self.result.plot_corner(truths=dict(x=1, y=1))
        self.result.plot_corner(truth=dict(x=1, y=1))
        with self.assertRaises(ValueError):
            self.result.plot_corner(truths=dict(x=1, y=1),
                                    parameters=dict(x=1, y=1))
        with self.assertRaises(ValueError):
            self.result.plot_corner(truths=[1, 1],
                                    parameters=dict(x=1, y=1))
        with self.assertRaises(ValueError):
            self.result.plot_corner(parameters=['x', 'y'],
                                    truths=dict(x=1, y=1))

    def test_plot_corner_with_injection_parameters(self):
        self.result.plot_corner()
        self.result.plot_corner(parameters=['x', 'y'])
        self.result.plot_corner(parameters=['x', 'y'], truths=[1, 1])
        self.result.plot_corner(parameters=dict(x=1, y=1))

    def test_plot_corner_with_priors(self):
        priors = bilby.core.prior.PriorDict()
        priors['x'] = bilby.core.prior.Uniform(-1, 1, 'x')
        priors['y'] = bilby.core.prior.Uniform(-1, 1, 'y')
        self.result.plot_corner(priors=priors)
        self.result.priors = priors
        self.result.plot_corner(priors=True)
        with self.assertRaises(ValueError):
            self.result.plot_corner(priors='test')


if __name__ == '__main__':
    unittest.main()
