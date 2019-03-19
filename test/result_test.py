from __future__ import absolute_import, division

import unittest
import numpy as np
import pandas as pd
import shutil
import os
import json

import bilby


class TestJson(unittest.TestCase):

    def setUp(self):
        self.encoder = bilby.core.utils.BilbyJsonEncoder
        self.decoder = bilby.core.utils.decode_bilby_json

    def test_list_encoding(self):
        data = dict(x=[1, 2, 3.4])
        encoded = json.dumps(data, cls=self.encoder)
        decoded = json.loads(encoded, object_hook=self.decoder)
        self.assertEqual(data.keys(), decoded.keys())
        self.assertEqual(type(data['x']), type(decoded['x']))
        self.assertTrue(np.all(data['x'] == decoded['x']))

    def test_array_encoding(self):
        data = dict(x=np.array([1, 2, 3.4]))
        encoded = json.dumps(data, cls=self.encoder)
        decoded = json.loads(encoded, object_hook=self.decoder)
        self.assertEqual(data.keys(), decoded.keys())
        self.assertEqual(type(data['x']), type(decoded['x']))
        self.assertTrue(np.all(data['x'] == decoded['x']))

    def test_complex_encoding(self):
        data = dict(x=1 + 3j)
        encoded = json.dumps(data, cls=self.encoder)
        decoded = json.loads(encoded, object_hook=self.decoder)
        self.assertEqual(data.keys(), decoded.keys())
        self.assertEqual(type(data['x']), type(decoded['x']))
        self.assertTrue(np.all(data['x'] == decoded['x']))

    def test_dataframe_encoding(self):
        data = dict(data=pd.DataFrame(dict(x=[3, 4, 5], y=[5, 6, 7])))
        encoded = json.dumps(data, cls=self.encoder)
        decoded = json.loads(encoded, object_hook=self.decoder)
        self.assertEqual(data.keys(), decoded.keys())
        self.assertEqual(type(data['data']), type(decoded['data']))
        self.assertTrue(np.all(data['data']['x'] == decoded['data']['x']))
        self.assertTrue(np.all(data['data']['y'] == decoded['data']['y']))


class TestResult(unittest.TestCase):

    def setUp(self):
        np.random.seed(7)
        bilby.utils.command_line_args.test = False
        priors = bilby.prior.PriorDict(dict(
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

        n = 100
        posterior = pd.DataFrame(dict(x=np.random.normal(0, 1, n),
                                      y=np.random.normal(0, 1, n)))
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

    def test_result_file_name_default(self):
        outdir = 'outdir'
        label = 'label'
        self.assertEqual(bilby.core.result.result_file_name(outdir, label),
                         '{}/{}_result.json'.format(outdir, label))

    def test_result_file_name_hdf5(self):
        outdir = 'outdir'
        label = 'label'
        self.assertEqual(bilby.core.result.result_file_name(outdir, label, extension='hdf5'),
                         '{}/{}_result.hdf5'.format(outdir, label))

    def test_fail_save_and_load(self):
        with self.assertRaises(ValueError):
            bilby.core.result.read_in_result()

        with self.assertRaises(ValueError):
            bilby.core.result.read_in_result(filename='no_file_extension')

        with self.assertRaises(IOError):
            bilby.core.result.read_in_result(filename='not/a/file.json')

    def test_unset_priors(self):
        result = bilby.core.result.Result(
            label='label', outdir='outdir', sampler='nestle',
            search_parameter_keys=['x', 'y'], fixed_parameter_keys=['c', 'd'],
            priors=None, sampler_kwargs=dict(test='test'),
            injection_parameters=dict(x=0.5, y=0.5),
            meta_data=dict(test='test'))
        with self.assertRaises(ValueError):
            _ = result.priors
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
            _ = self.result.posterior

    def test_save_and_load_hdf5(self):
        self.result.save_to_file(extension='hdf5')
        loaded_result = bilby.core.result.read_in_result(
            outdir=self.result.outdir, label=self.result.label, extension='hdf5')
        self.assertTrue(pd.DataFrame.equals
                        (self.result.posterior, loaded_result.posterior))
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

    def test_save_and_load_default(self):
        self.result.save_to_file()
        loaded_result = bilby.core.result.read_in_result(
            outdir=self.result.outdir, label=self.result.label)
        self.assertTrue(np.array_equal
                        (self.result.posterior.sort_values(by=['x']),
                            loaded_result.posterior.sort_values(by=['x'])))
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

    def test_save_and_dont_overwrite_default(self):
        shutil.rmtree(
            '{}/{}_result.json.old'.format(self.result.outdir, self.result.label),
            ignore_errors=True)
        self.result.save_to_file(overwrite=False)
        self.result.save_to_file(overwrite=False)
        self.assertTrue(os.path.isfile(
            '{}/{}_result.json.old'.format(self.result.outdir, self.result.label)))

    def test_save_and_dont_overwrite_hdf5(self):
        shutil.rmtree(
            '{}/{}_result.hdf5.old'.format(self.result.outdir, self.result.label),
            ignore_errors=True)
        self.result.save_to_file(overwrite=False, extension='hdf5')
        self.result.save_to_file(overwrite=False, extension='hdf5')
        self.assertTrue(os.path.isfile(
            '{}/{}_result.hdf5.old'.format(self.result.outdir, self.result.label)))

    def test_save_and_overwrite_hdf5(self):
        shutil.rmtree(
            '{}/{}_result.hdf5.old'.format(self.result.outdir, self.result.label),
            ignore_errors=True)
        self.result.save_to_file(overwrite=True, extension='hdf5')
        self.result.save_to_file(overwrite=True, extension='hdf5')
        self.assertFalse(os.path.isfile(
            '{}/{}_result.hdf5.old'.format(self.result.outdir, self.result.label)))

    def test_save_and_overwrite_default(self):
        shutil.rmtree(
            '{}/{}_result.json.old'.format(self.result.outdir, self.result.label),
            ignore_errors=True)
        self.result.save_to_file(overwrite=True, extension='hdf5')
        self.result.save_to_file(overwrite=True, extension='hdf5')
        self.assertFalse(os.path.isfile(
            '{}/{}_result.h5.old'.format(self.result.outdir, self.result.label)))

    def test_save_and_overwrite_default(self):
        shutil.rmtree(
            '{}/{}_result.json.old'.format(self.result.outdir, self.result.label),
            ignore_errors=True)
        self.result.save_to_file(overwrite=True)
        self.result.save_to_file(overwrite=True)
        self.assertFalse(os.path.isfile(
            '{}/{}_result.json.old'.format(self.result.outdir, self.result.label)))

    def test_save_samples(self):
        self.result.save_posterior_samples()
        filename = '{}/{}_posterior_samples.txt'.format(self.result.outdir, self.result.label)
        self.assertTrue(os.path.isfile(filename))
        df = pd.read_csv(filename)
        self.assertTrue(np.allclose(self.result.posterior.values, df.values))

    def test_samples_to_posterior(self):
        self.result.posterior = None
        x = [1, 2, 3]
        y = [4, 6, 8]
        log_likelihood = np.array([6, 7, 8])
        self.result.samples = np.array([x, y]).T
        self.result.log_likelihood_evaluations = log_likelihood
        self.result.samples_to_posterior(priors=self.result.priors)
        self.assertTrue(all(self.result.posterior['x'] == x))
        self.assertTrue(all(self.result.posterior['y'] == y))
        self.assertTrue(np.array_equal(self.result.posterior.log_likelihood.values, log_likelihood))
        self.assertTrue(all(self.result.posterior.c.values == self.result.priors['c'].peak))
        self.assertTrue(all(self.result.posterior.d.values == self.result.priors['d'].peak))

    def test_calculate_prior_values(self):
        self.result.calculate_prior_values(priors=self.result.priors)
        self.assertEqual(len(self.result.posterior), len(self.result.prior_values))

    def test_plot_multiple(self):
        filename = 'multiple.png'.format(self.result.outdir)
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

        def model(xx, theta):
            return xx
        self.result.posterior = pd.DataFrame(dict(theta=[1, 2, 3]))
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

    def test_get_credible_levels(self):
        levels = self.result.get_all_injection_credible_levels()
        self.assertDictEqual(levels, dict(x=0.68, y=0.72))

    def test_get_credible_levels_raises_error_if_no_injection_parameters(self):
        self.result.injection_parameters = None
        with self.assertRaises(TypeError):
            self.result.get_all_injection_credible_levels()

    def test_kde(self):
        kde = self.result.kde
        import scipy.stats
        self.assertEqual(type(kde), scipy.stats.kde.gaussian_kde)
        self.assertEqual(kde.d, 2)

    def test_posterior_probability(self):
        sample = dict(x=0, y=0.1)
        self.assertTrue(
            isinstance(self.result.posterior_probability(sample), np.ndarray))
        self.assertTrue(
            len(self.result.posterior_probability(sample)), 1)
        self.assertEqual(
            self.result.posterior_probability(sample)[0],
            self.result.kde([0, 0.1]))

    def test_multiple_posterior_probability(self):
        sample = [dict(x=0, y=0.1), dict(x=0.8, y=0)]
        self.assertTrue(
            isinstance(self.result.posterior_probability(sample), np.ndarray))
        self.assertTrue(np.array_equal(self.result.posterior_probability(sample),
                                       self.result.kde([[0, 0.1], [0.8, 0]])))


if __name__ == '__main__':
    unittest.main()
