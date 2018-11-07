from __future__ import absolute_import, division

import bilby
import unittest
import numpy as np
import pandas as pd
import shutil
import os


class TestResult(unittest.TestCase):

    def setUp(self):
        bilby.utils.command_line_args.test = False
        priors = bilby.prior.PriorSet(dict(
            x=bilby.prior.Uniform(0, 1, 'x', latex_label='$x$', unit='s'),
            y=bilby.prior.Uniform(0, 1, 'y', latex_label='$y$', unit='m'),
            c=1,
            d=2))
        result = bilby.core.result.Result(
            label='label', outdir='outdir', sampler='nestle',
            search_parameter_keys=['x', 'y'], fixed_parameter_keys=['c', 'd'],
            priors=priors, sampler_kwargs=dict(test='test', func=lambda x: x),
            injection_parameters=dict(x=0.5, y=0.5),
            meta_data=dict(test='test'))

        N = 100
        posterior = pd.DataFrame(dict(x=np.random.normal(0, 1, N),
                                      y=np.random.normal(0, 1, N)))
        result.posterior = posterior
        result.log_evidence = 10
        result.log_evidence_err = 11
        result.log_bayes_factor = 12
        result.log_noise_evidence = 13
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

    def test_result_file_name(self):
        outdir = 'outdir'
        label = 'label'
        self.assertEqual(bilby.core.result.result_file_name(outdir, label),
                         '{}/{}_result.h5'.format(outdir, label))

    def test_fail_save_and_load(self):
        with self.assertRaises(ValueError):
            bilby.core.result.read_in_result()

        with self.assertRaises(IOError):
            bilby.core.result.read_in_result(filename='not/a/file')

    def test_unset_priors(self):
        result = bilby.core.result.Result(
            label='label', outdir='outdir', sampler='nestle',
            search_parameter_keys=['x', 'y'], fixed_parameter_keys=['c', 'd'],
            priors=None, sampler_kwargs=dict(test='test'),
            injection_parameters=dict(x=0.5, y=0.5),
            meta_data=dict(test='test'))
        with self.assertRaises(ValueError):
            result.priors
        self.assertEqual(result.parameter_labels, result.search_parameter_keys)
        self.assertEqual(result.parameter_labels_with_unit, result.search_parameter_keys)

    def test_unknown_priors_fail(self):
        with self.assertRaises(ValueError):
            bilby.core.result.Result(
                label='label', outdir='outdir', sampler='nestle',
                search_parameter_keys=['x', 'y'], fixed_parameter_keys=['c', 'd'],
                priors=['a', 'b'], sampler_kwargs=dict(test='test'),
                injection_parameters=dict(x=0.5, y=0.5),
                meta_data=dict(test='test'))

    def test_set_samples(self):
        samples = [1, 2, 3]
        self.result.samples = samples
        self.assertEqual(samples, self.result.samples)

    def test_set_nested_samples(self):
        nested_samples = [1, 2, 3]
        self.result.nested_samples = nested_samples
        self.assertEqual(nested_samples, self.result.nested_samples)

    def test_set_walkers(self):
        walkers = [1, 2, 3]
        self.result.walkers = walkers
        self.assertEqual(walkers, self.result.walkers)

    def test_set_nburn(self):
        nburn = 1
        self.result.nburn = nburn
        self.assertEqual(nburn, self.result.nburn)

    def test_unset_posterior(self):
        self.result.posterior = None
        with self.assertRaises(ValueError):
            self.result.posterior

    def test_save_and_load(self):
        self.result.save_to_file()
        loaded_result = bilby.core.result.read_in_result(
            outdir=self.result.outdir, label=self.result.label)
        self.assertTrue(
            all(self.result.posterior == loaded_result.posterior))
        self.assertTrue(self.result.fixed_parameter_keys == loaded_result.fixed_parameter_keys)
        self.assertTrue(self.result.search_parameter_keys == loaded_result.search_parameter_keys)
        self.assertEqual(self.result.meta_data, loaded_result.meta_data)
        self.assertEqual(self.result.injection_parameters, loaded_result.injection_parameters)
        self.assertEqual(self.result.log_evidence, loaded_result.log_evidence)
        self.assertEqual(self.result.log_noise_evidence, loaded_result.log_noise_evidence)
        self.assertEqual(self.result.log_evidence_err, loaded_result.log_evidence_err)
        self.assertEqual(self.result.log_bayes_factor, loaded_result.log_bayes_factor)
        self.assertEqual(self.result.priors['x'], loaded_result.priors['x'])
        self.assertEqual(self.result.priors['y'], loaded_result.priors['y'])
        self.assertEqual(self.result.priors['c'], loaded_result.priors['c'])
        self.assertEqual(self.result.priors['d'], loaded_result.priors['d'])

    def test_save_and_dont_overwrite(self):
        shutil.rmtree(
            '{}/{}_result.h5.old'.format(self.result.outdir, self.result.label),
            ignore_errors=True)
        self.result.save_to_file(overwrite=False)
        self.result.save_to_file(overwrite=False)
        self.assertTrue(os.path.isfile(
            '{}/{}_result.h5.old'.format(self.result.outdir, self.result.label)))

    def test_save_and_overwrite(self):
        shutil.rmtree(
            '{}/{}_result.h5.old'.format(self.result.outdir, self.result.label),
            ignore_errors=True)
        self.result.save_to_file(overwrite=True)
        self.result.save_to_file(overwrite=True)
        self.assertFalse(os.path.isfile(
            '{}/{}_result.h5.old'.format(self.result.outdir, self.result.label)))

    def test_save_samples(self):
        self.result.save_posterior_samples()
        filename = '{}/{}_posterior_samples.txt'.format(self.result.outdir, self.result.label)
        self.assertTrue(os.path.isfile(filename))
        df = pd.read_csv(filename)
        self.assertTrue(all(self.result.posterior == df))

    def test_samples_to_posterior(self):
        self.result.posterior = None
        x = [1, 2, 3]
        y = [4, 6, 8]
        log_likelihood = [6, 7, 8]
        self.result.samples = np.array([x, y]).T
        self.result.log_likelihood_evaluations = log_likelihood
        self.result.samples_to_posterior(priors=self.result.priors)
        self.assertTrue(all(self.result.posterior['x'] == x))
        self.assertTrue(all(self.result.posterior['y'] == y))
        self.assertTrue(
            all(self.result.posterior['log_likelihood'] == log_likelihood))
        self.assertTrue(
            all(self.result.posterior['c'] == self.result.priors['c'].peak))
        self.assertTrue(
            all(self.result.posterior['d'] == self.result.priors['d'].peak))

    def test_calculate_prior_values(self):
        self.result.calculate_prior_values(priors=self.result.priors)
        self.assertEqual(len(self.result.posterior), len(self.result.prior_values))

    def test_plot_multiple(self):
        filename='multiple.png'.format(self.result.outdir)
        bilby.core.result.plot_multiple([self.result, self.result],
                                        filename=filename)
        self.assertTrue(os.path.isfile(filename))
        os.remove(filename)

    def test_plot_walkers(self):
        self.result.walkers = np.random.uniform(0, 1, (10, 11, 2))
        self.result.nburn = 5
        self.result.plot_walkers()
        self.assertTrue(
            os.path.isfile('{}/{}_walkers.png'.format(
                self.result.outdir, self.result.label)))

    def test_plot_with_data(self):
        x = np.linspace(0, 1, 10)
        y = np.linspace(0, 1, 10)

        def model(x):
            return x
        self.result.plot_with_data(model, x, y, ndraws=10)
        self.assertTrue(
            os.path.isfile('{}/{}_plot_with_data.png'.format(
                self.result.outdir, self.result.label)))
        self.result.posterior['log_likelihood'] = np.random.uniform(0, 1, len(self.result.posterior))
        self.result.plot_with_data(model, x, y, ndraws=10, xlabel='a', ylabel='y')

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
