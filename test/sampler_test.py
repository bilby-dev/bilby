from __future__ import absolute_import
import bilby
from bilby.core import prior
import unittest
from mock import MagicMock
import numpy as np
import os
import sys
import shutil
import copy


class TestSampler(unittest.TestCase):

    def setUp(self):
        likelihood = bilby.core.likelihood.Likelihood()
        likelihood.parameters = dict(a=1, b=2, c=3)
        delta_prior = prior.DeltaFunction(peak=0)
        delta_prior.rescale = MagicMock(return_value=prior.DeltaFunction(peak=1))
        delta_prior.prob = MagicMock(return_value=1)
        delta_prior.sample = MagicMock(return_value=0)
        uniform_prior = prior.Uniform(0, 1)
        uniform_prior.rescale = MagicMock(return_value=prior.Uniform(0, 2))
        uniform_prior.prob = MagicMock(return_value=1)
        uniform_prior.sample = MagicMock(return_value=0.5)

        priors = dict(a=delta_prior, b='string', c=uniform_prior)
        likelihood.log_likelihood_ratio = MagicMock(return_value=1)
        likelihood.log_likelihood = MagicMock(return_value=2)
        test_directory = 'test_directory'
        if os.path.isdir(test_directory):
            os.rmdir(test_directory)
        self.sampler = bilby.core.sampler.Sampler(
            likelihood=likelihood, priors=priors,
            outdir=test_directory, use_ratio=False,
            skip_import_verification=True)

    def tearDown(self):
        del self.sampler

    def test_search_parameter_keys(self):
        expected_search_parameter_keys = ['c']
        self.assertListEqual(self.sampler.search_parameter_keys, expected_search_parameter_keys)

    def test_fixed_parameter_keys(self):
        expected_fixed_parameter_keys = ['a']
        self.assertListEqual(self.sampler.fixed_parameter_keys, expected_fixed_parameter_keys)

    def test_ndim(self):
        self.assertEqual(self.sampler.ndim, 1)

    def test_kwargs(self):
        self.assertDictEqual(self.sampler.kwargs, {})

    def test_label(self):
        self.assertEqual(self.sampler.label, 'label')

    def test_prior_transform_transforms_search_parameter_keys(self):
        self.sampler.prior_transform([0])
        expected_prior = prior.Uniform(0, 1)
        self.assertListEqual([self.sampler.priors['c'].minimum,
                              self.sampler.priors['c'].maximum],
                             [expected_prior.minimum,
                              expected_prior.maximum])

    def test_prior_transform_does_not_transform_fixed_parameter_keys(self):
        self.sampler.prior_transform([0])
        self.assertEqual(self.sampler.priors['a'].peak,
                         prior.DeltaFunction(peak=0).peak)

    def test_log_prior(self):
        self.assertEqual(self.sampler.log_prior({1}), 0.0)

    def test_log_likelihood_with_use_ratio(self):
        self.sampler.use_ratio = True
        self.assertEqual(self.sampler.log_likelihood([0]), 1)

    def test_log_likelihood_without_use_ratio(self):
        self.sampler.use_ratio = False
        self.assertEqual(self.sampler.log_likelihood([0]), 2)

    def test_log_likelihood_correctly_sets_parameters(self):
        expected_dict = dict(a=0,
                             b=2,
                             c=0)
        _ = self.sampler.log_likelihood([0])
        self.assertDictEqual(self.sampler.likelihood.parameters, expected_dict)

    def test_get_random_draw(self):
        self.assertEqual(self.sampler.get_random_draw_from_prior(), np.array([0.5]))

    def test_base_run_sampler(self):
        sampler_copy = copy.copy(self.sampler)
        self.sampler.run_sampler()
        self.assertDictEqual(sampler_copy.__dict__, self.sampler.__dict__)


class TestCPNest(unittest.TestCase):

    def setUp(self):
        self.likelihood = MagicMock()
        self.priors = dict()
        self.sampler = bilby.core.sampler.Cpnest(self.likelihood, self.priors,
                                                 outdir='outdir', label='label',
                                                 use_ratio=False, plot=False,
                                                 skip_import_verification=True)

    def tearDown(self):
        del self.likelihood
        del self.priors
        del self.sampler

    def test_default_kwargs(self):
        expected = dict(verbose=1, nthreads=1, nlive=500, maxmcmc=1000,
                        seed=None, poolsize=100, nhamiltonian=0, resume=True,
                        output='outdir/cpnest_label/', proposals=None)
        self.assertDictEqual(expected, self.sampler.kwargs)

    def test_translate_kwargs(self):
        expected = dict(verbose=1, nthreads=1, nlive=250, maxmcmc=1000,
                        seed=None, poolsize=100, nhamiltonian=0, resume=True,
                        output='outdir/cpnest_label/', proposals=None)
        for equiv in bilby.core.sampler.base_sampler.NestedSampler.npoints_equiv_kwargs:
            new_kwargs = self.sampler.kwargs.copy()
            del new_kwargs['nlive']
            new_kwargs[equiv] = 250
            self.sampler.kwargs = new_kwargs
            self.assertDictEqual(expected, self.sampler.kwargs)


class TestDynesty(unittest.TestCase):

    def setUp(self):
        self.likelihood = MagicMock()
        self.priors = bilby.core.prior.PriorDict()
        self.priors['a'] = bilby.core.prior.Prior(periodic_boundary=True)
        self.priors['b'] = bilby.core.prior.Prior(periodic_boundary=False)
        self.sampler = bilby.core.sampler.Dynesty(self.likelihood, self.priors,
                                                  outdir='outdir', label='label',
                                                  use_ratio=False, plot=False,
                                                  skip_import_verification=True)

    def tearDown(self):
        del self.likelihood
        del self.priors
        del self.sampler

    def test_default_kwargs(self):
        expected = dict(bound='multi', sample='rwalk', periodic=None, verbose=True,
                        check_point_delta_t=600, nlive=500, first_update=None,
                        npdim=None, rstate=None, queue_size=None, pool=None,
                        use_pool=None, live_points=None, logl_args=None, logl_kwargs=None,
                        ptform_args=None, ptform_kwargs=None,
                        enlarge=None, bootstrap=None, vol_dec=0.5, vol_check=2.0,
                        facc=0.5, slices=5, dlogz=0.1, maxiter=None, maxcall=None,
                        logl_max=np.inf, add_live=True, print_progress=True, save_bounds=True,
                        walks=10, update_interval=300, print_func='func')
        self.sampler.kwargs['print_func'] = 'func'  # set this manually as this is not testable otherwise
        self.assertListEqual([0], self.sampler.kwargs['periodic'])  # Check this separately
        self.sampler.kwargs['periodic'] = None  # The dict comparison can't handle lists
        for key in self.sampler.kwargs.keys():
            print(key)
            print(expected[key])
            print(self.sampler.kwargs[key])
        self.assertDictEqual(expected, self.sampler.kwargs)

    def test_translate_kwargs(self):
        expected = dict(bound='multi', sample='rwalk', periodic=[0], verbose=True,
                        check_point_delta_t=600, nlive=250, first_update=None,
                        npdim=None, rstate=None, queue_size=None, pool=None,
                        use_pool=None, live_points=None, logl_args=None, logl_kwargs=None,
                        ptform_args=None, ptform_kwargs=None,
                        enlarge=None, bootstrap=None, vol_dec=0.5, vol_check=2.0,
                        facc=0.5, slices=5, dlogz=0.1, maxiter=None, maxcall=None,
                        logl_max=np.inf, add_live=True, print_progress=True, save_bounds=True,
                        walks=10, update_interval=300, print_func='func')

        for equiv in bilby.core.sampler.base_sampler.NestedSampler.npoints_equiv_kwargs:
            new_kwargs = self.sampler.kwargs.copy()
            del new_kwargs['nlive']
            new_kwargs[equiv] = 250
            self.sampler.kwargs = new_kwargs
            self.sampler.kwargs['print_func'] = 'func'  # set this manually as this is not testable otherwise
            self.assertDictEqual(expected, self.sampler.kwargs)


class TestEmcee(unittest.TestCase):

    def setUp(self):
        self.likelihood = MagicMock()
        self.priors = dict()
        self.sampler = bilby.core.sampler.Emcee(self.likelihood, self.priors,
                                                outdir='outdir', label='label',
                                                use_ratio=False, plot=False,
                                                skip_import_verification=True)

    def tearDown(self):
        del self.likelihood
        del self.priors
        del self.sampler

    def test_default_kwargs(self):
        expected = dict(nwalkers=500, a=2, args=[], kwargs={},
                        postargs=None, pool=None, live_dangerously=False,
                        runtime_sortingfn=None, lnprob0=None, rstate0=None,
                        blobs0=None, iterations=100, thin=1, storechain=True, mh_proposal=None
                        )
        self.assertDictEqual(expected, self.sampler.kwargs)

    def test_translate_kwargs(self):
        expected = dict(nwalkers=100, a=2, args=[], kwargs={},
                        postargs=None, pool=None, live_dangerously=False,
                        runtime_sortingfn=None, lnprob0=None, rstate0=None,
                        blobs0=None, iterations=100, thin=1, storechain=True, mh_proposal=None)
        for equiv in bilby.core.sampler.base_sampler.MCMCSampler.nwalkers_equiv_kwargs:
            new_kwargs = self.sampler.kwargs.copy()
            del new_kwargs['nwalkers']
            new_kwargs[equiv] = 100
            self.sampler.kwargs = new_kwargs
            self.assertDictEqual(expected, self.sampler.kwargs)


class TestNestle(unittest.TestCase):

    def setUp(self):
        self.likelihood = MagicMock()
        self.priors = dict()
        self.sampler = bilby.core.sampler.Nestle(self.likelihood, self.priors,
                                                 outdir='outdir', label='label',
                                                 use_ratio=False, plot=False,
                                                 skip_import_verification=True,
                                                 verbose=False)

    def tearDown(self):
        del self.likelihood
        del self.priors
        del self.sampler

    def test_default_kwargs(self):
        expected = dict(verbose=False, method='multi', npoints=500,
                        update_interval=None, npdim=None, maxiter=None,
                        maxcall=None, dlogz=None, decline_factor=None,
                        rstate=None, callback=None)
        self.assertDictEqual(expected, self.sampler.kwargs)

    def test_translate_kwargs(self):
        expected = dict(verbose=False, method='multi', npoints=345,
                        update_interval=None, npdim=None, maxiter=None,
                        maxcall=None, dlogz=None, decline_factor=None,
                        rstate=None, callback=None)
        self.sampler.kwargs['npoints'] = 123
        for equiv in bilby.core.sampler.base_sampler.NestedSampler.npoints_equiv_kwargs:
            new_kwargs = self.sampler.kwargs.copy()
            del new_kwargs['npoints']
            new_kwargs[equiv] = 345
            self.sampler.kwargs = new_kwargs
            self.assertDictEqual(expected, self.sampler.kwargs)


class TestPolyChord(unittest.TestCase):

    def setUp(self):
        self.likelihood = MagicMock()
        self.priors = dict(a=bilby.prior.Uniform(0, 1))
        self.sampler = bilby.core.sampler.PyPolyChord(self.likelihood, self.priors,
                                                      outdir='outdir', label='polychord',
                                                      use_ratio=False, plot=False,
                                                      skip_import_verification=True)

    def tearDown(self):
        del self.likelihood
        del self.priors
        del self.sampler

    def test_default_kwargs(self):
        expected = dict(use_polychord_defaults=False, nlive=self.sampler.ndim*25, num_repeats=self.sampler.ndim*5,
                        nprior=-1, do_clustering=True, feedback=1, precision_criterion=0.001,
                        logzero=-1e30, max_ndead=-1, boost_posterior=0.0, posteriors=True,
                        equals=True, cluster_posteriors=True, write_resume=True,
                        write_paramnames=False, read_resume=True, write_stats=True,
                        write_live=True, write_dead=True, write_prior=True,
                        compression_factor=np.exp(-1), base_dir='outdir',
                        file_root='polychord', seed=-1, grade_dims=list([self.sampler.ndim]),
                        grade_frac=list([1.0]*len([self.sampler.ndim])), nlives={})
        self.sampler._setup_dynamic_defaults()
        self.assertDictEqual(expected, self.sampler.kwargs)

    def test_translate_kwargs(self):
        expected = dict(use_polychord_defaults=False, nlive=123, num_repeats=self.sampler.ndim*5,
                        nprior=-1, do_clustering=True, feedback=1, precision_criterion=0.001,
                        logzero=-1e30, max_ndead=-1, boost_posterior=0.0, posteriors=True,
                        equals=True, cluster_posteriors=True, write_resume=True,
                        write_paramnames=False, read_resume=True, write_stats=True,
                        write_live=True, write_dead=True, write_prior=True,
                        compression_factor=np.exp(-1), base_dir='outdir',
                        file_root='polychord', seed=-1, grade_dims=list([self.sampler.ndim]),
                        grade_frac=list([1.0]*len([self.sampler.ndim])), nlives={})
        self.sampler._setup_dynamic_defaults()
        for equiv in bilby.core.sampler.base_sampler.NestedSampler.npoints_equiv_kwargs:
            new_kwargs = self.sampler.kwargs.copy()
            del new_kwargs['nlive']
            new_kwargs[equiv] = 123
            self.sampler.kwargs = new_kwargs
            self.assertDictEqual(expected, self.sampler.kwargs)


class TestPTEmcee(unittest.TestCase):

    def setUp(self):
        self.likelihood = MagicMock()
        self.priors = dict()
        self.sampler = bilby.core.sampler.Ptemcee(self.likelihood, self.priors,
                                                  outdir='outdir', label='label',
                                                  use_ratio=False, plot=False,
                                                  skip_import_verification=True)

    def tearDown(self):
        del self.likelihood
        del self.priors
        del self.sampler

    def test_default_kwargs(self):
        expected = dict(ntemps=2, nwalkers=500,
                        Tmax=None, betas=None,
                        threads=1, pool=None, a=2.0,
                        loglargs=[], logpargs=[],
                        loglkwargs={}, logpkwargs={},
                        adaptation_lag=10000, adaptation_time=100,
                        random=None, iterations=100, thin=1,
                        storechain=True, adapt=True,
                        swap_ratios=False,
                        )
        self.assertDictEqual(expected, self.sampler.kwargs)

    def test_translate_kwargs(self):
        expected = dict(ntemps=2, nwalkers=150,
                        Tmax=None, betas=None,
                        threads=1, pool=None, a=2.0,
                        loglargs=[], logpargs=[],
                        loglkwargs={}, logpkwargs={},
                        adaptation_lag=10000, adaptation_time=100,
                        random=None, iterations=100, thin=1,
                        storechain=True, adapt=True,
                        swap_ratios=False,
                        )
        for equiv in bilby.core.sampler.base_sampler.MCMCSampler.nwalkers_equiv_kwargs:
            new_kwargs = self.sampler.kwargs.copy()
            del new_kwargs['nwalkers']
            new_kwargs[equiv] = 150
            self.sampler.kwargs = new_kwargs
            self.assertDictEqual(expected, self.sampler.kwargs)


class TestPyMC3(unittest.TestCase):

    def setUp(self):
        self.likelihood = MagicMock()
        self.priors = dict()
        self.sampler = bilby.core.sampler.Pymc3(self.likelihood, self.priors,
                                                outdir='outdir', label='label',
                                                use_ratio=False, plot=False,
                                                skip_import_verification=True)

    def tearDown(self):
        del self.likelihood
        del self.priors
        del self.sampler

    def test_default_kwargs(self):
        expected = dict(
            draws=500, step=None, init='auto', n_init=200000, start=None, trace=None, chain_idx=0,
            chains=2, cores=1, tune=500, nuts_kwargs=None, step_kwargs=None, progressbar=True,
            model=None, random_seed=None, live_plot=False, discard_tuned_samples=True,
            live_plot_kwargs=None, compute_convergence_checks=True, use_mmap=False)
        self.assertDictEqual(expected, self.sampler.kwargs)

    def test_translate_kwargs(self):
        expected = dict(
            draws=500, step=None, init='auto', n_init=200000, start=None, trace=None, chain_idx=0,
            chains=2, cores=1, tune=500, nuts_kwargs=None, step_kwargs=None, progressbar=True,
            model=None, random_seed=None, live_plot=False, discard_tuned_samples=True,
            live_plot_kwargs=None, compute_convergence_checks=True, use_mmap=False)
        self.sampler.kwargs['draws'] = 123
        for equiv in bilby.core.sampler.base_sampler.NestedSampler.npoints_equiv_kwargs:
            new_kwargs = self.sampler.kwargs.copy()
            del new_kwargs['draws']
            new_kwargs[equiv] = 500
            self.sampler.kwargs = new_kwargs
            self.assertDictEqual(expected, self.sampler.kwargs)


class TestPymultinest(unittest.TestCase):

    def setUp(self):
        self.likelihood = MagicMock()
        self.priors = bilby.core.prior.PriorDict()
        self.priors['a'] = bilby.core.prior.Prior(periodic_boundary=True)
        self.priors['b'] = bilby.core.prior.Prior(periodic_boundary=False)
        self.sampler = bilby.core.sampler.Pymultinest(self.likelihood, self.priors,
                                                      outdir='outdir', label='label',
                                                      use_ratio=False, plot=False,
                                                      skip_import_verification=True)

    def tearDown(self):
        del self.likelihood
        del self.priors
        del self.sampler

    def test_default_kwargs(self):
        expected = dict(importance_nested_sampling=False, resume=True,
                        verbose=True, sampling_efficiency='parameter',
                        outputfiles_basename='outdir/pm_label/',
                        n_live_points=500, n_params=None,
                        n_clustering_params=None, wrapped_params=None,
                        multimodal=True, const_efficiency_mode=False,
                        evidence_tolerance=0.5,
                        n_iter_before_update=100, null_log_evidence=-1e90,
                        max_modes=100, mode_tolerance=-1e90, seed=-1,
                        context=0, write_output=True, log_zero=-1e100,
                        max_iter=0, init_MPI=False, dump_callback=None)
        self.assertListEqual([1, 0], self.sampler.kwargs['wrapped_params'])  # Check this separately
        self.sampler.kwargs['wrapped_params'] = None  # The dict comparison can't handle lists
        self.assertDictEqual(expected, self.sampler.kwargs)

    def test_translate_kwargs(self):
        expected = dict(importance_nested_sampling=False, resume=True,
                        verbose=True, sampling_efficiency='parameter',
                        outputfiles_basename='outdir/pm_label/',
                        n_live_points=123, n_params=None,
                        n_clustering_params=None, wrapped_params=None,
                        multimodal=True, const_efficiency_mode=False,
                        evidence_tolerance=0.5,
                        n_iter_before_update=100, null_log_evidence=-1e90,
                        max_modes=100, mode_tolerance=-1e90, seed=-1,
                        context=0, write_output=True, log_zero=-1e100,
                        max_iter=0, init_MPI=False, dump_callback=None)
        for equiv in bilby.core.sampler.base_sampler.NestedSampler.npoints_equiv_kwargs:
            new_kwargs = self.sampler.kwargs.copy()
            del new_kwargs['n_live_points']
            new_kwargs['wrapped_params'] = None  # The dict comparison can't handle lists
            new_kwargs[equiv] = 123
            self.sampler.kwargs = new_kwargs
            self.assertDictEqual(expected, self.sampler.kwargs)


class TestRunningSamplers(unittest.TestCase):

    def setUp(self):
        np.random.seed(42)
        bilby.core.utils.command_line_args.test = False
        self.x = np.linspace(0, 1, 11)
        self.model = lambda x, m, c: m * x + c
        self.injection_parameters = dict(m=0.5, c=0.2)
        self.sigma = 0.1
        self.y = self.model(self.x, **self.injection_parameters) +\
            np.random.normal(0, self.sigma, len(self.x))
        self.likelihood = bilby.likelihood.GaussianLikelihood(
            self.x, self.y, self.model, self.sigma)

        self.priors = bilby.core.prior.PriorDict()
        self.priors['m'] = bilby.core.prior.Uniform(0, 5, periodic_boundary=False)
        self.priors['c'] = bilby.core.prior.Uniform(-2, 2, periodic_boundary=False)
        bilby.core.utils.check_directory_exists_and_if_not_mkdir('outdir')

    def tearDown(self):
        del self.likelihood
        del self.priors
        bilby.core.utils.command_line_args.test = False
        shutil.rmtree('outdir')

    def test_run_cpnest(self):
        _ = bilby.run_sampler(
            likelihood=self.likelihood, priors=self.priors, sampler='cpnest',
            nlive=100, save=False)

    def test_run_dynesty(self):
        _ = bilby.run_sampler(
            likelihood=self.likelihood, priors=self.priors, sampler='dynesty',
            nlive=100, save=False)

    def test_run_emcee(self):
        _ = bilby.run_sampler(
            likelihood=self.likelihood, priors=self.priors, sampler='emcee',
            nsteps=1000, nwalkers=10, save=False)

    def test_run_nestle(self):
        _ = bilby.run_sampler(
            likelihood=self.likelihood, priors=self.priors, sampler='nestle',
            nlive=100, save=False)

    def test_run_pypolychord(self):
        _ = bilby.run_sampler(
            likelihood=self.likelihood, priors=self.priors,
            sampler='pypolychord', nlive=100, save=False)

    def test_run_ptemcee(self):
        _ = bilby.run_sampler(
            likelihood=self.likelihood, priors=self.priors, sampler='ptemcee',
            nsteps=1000, nwalkers=10, ntemps=10, save=False)

    def test_run_pymc3(self):
        _ = bilby.run_sampler(
            likelihood=self.likelihood, priors=self.priors, sampler='pymc3',
            draws=50, tune=50, n_init=1000, save=False)

    def test_run_pymultinest(self):
        _ = bilby.run_sampler(
            likelihood=self.likelihood, priors=self.priors,
            sampler='pymultinest', nlive=100, save=False)

    def test_run_PTMCMCSampler(self):
        _ = bilby.run_sampler(
            likelihood=self.likelihood, priors=self.priors,
            sampler= 'PTMCMCsampler', Niter=101, burn =2,
            isave = 100 ,save=False)


if __name__ == '__main__':
    unittest.main()
