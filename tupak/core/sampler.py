from __future__ import print_function, division, absolute_import

import inspect
import os
import sys
import numpy as np
import datetime
import deepdish
import pandas as pd

from tupak.core.utils import logger
from tupak.core.result import Result, read_in_result
from tupak.core.prior import Prior
from tupak.core import utils
import tupak


class Sampler(object):
    """ A sampler object to aid in setting up an inference run

    Parameters
    ----------
    likelihood: likelihood.Likelihood
        A  object with a log_l method
    priors: tupak.core.prior.PriorSet, dict
        Priors to be used in the search.
        This has attributes for each parameter to be sampled.
    external_sampler: str, Sampler, optional
        A string containing the module name of the sampler or an instance of this class
    outdir: str, optional
        Name of the output directory
    label: str, optional
        Naming scheme of the output files
    use_ratio: bool, optional
        Switch to set whether or not you want to use the log-likelihood ratio or just the log-likelihood
    plot: bool, optional
        Switch to set whether or not you want to create traceplots
    **kwargs: dict

    Attributes
    -------
    likelihood: likelihood.Likelihood
        A  object with a log_l method
    priors: tupak.core.prior.PriorSet
        Priors to be used in the search.
        This has attributes for each parameter to be sampled.
    external_sampler: Module
        An external module containing an implementation of a sampler.
    outdir: str
        Name of the output directory
    label: str
        Naming scheme of the output files
    use_ratio: bool
        Switch to set whether or not you want to use the log-likelihood ratio or just the log-likelihood
    plot: bool
        Switch to set whether or not you want to create traceplots
    result: tupak.core.result.Result
        Container for the results of the sampling run
    kwargs: dict
        Dictionary of keyword arguments that can be used in the external sampler

    Raises
    -------
    TypeError:
        If external_sampler is neither a string nor an instance of this class
        If not all likelihood.parameters have been defined
    ImportError:
        If the external_sampler string does not refer to a sampler that is installed on this system
    AttributeError:
        If some of the priors can't be sampled
    """

    def __init__(
            self, likelihood, priors, external_sampler='dynesty',
            outdir='outdir', label='label', use_ratio=False, plot=False,
            **kwargs):
        self.likelihood = likelihood
        if isinstance(priors, tupak.prior.PriorSet):
            self.priors = priors
        else:
            self.priors = tupak.prior.PriorSet(priors)
        self.label = label
        self.outdir = outdir
        self.use_ratio = use_ratio
        self.external_sampler = external_sampler
        self.external_sampler_function = None
        self.plot = plot

        self.__search_parameter_keys = []
        self.__fixed_parameter_keys = []
        self._initialise_parameters()
        self._verify_parameters()
        self._verify_use_ratio()
        self.kwargs = kwargs

        self._check_cached_result()

        self._log_summary_for_sampler()

        if os.path.isdir(outdir) is False:
            os.makedirs(outdir)

        self.result = self._initialise_result()

    @property
    def search_parameter_keys(self):
        """list: List of parameter keys that are being sampled"""
        return self.__search_parameter_keys

    @property
    def fixed_parameter_keys(self):
        """list: List of parameter keys that are not being sampled"""
        return self.__fixed_parameter_keys

    @property
    def ndim(self):
        """int: Number of dimensions of the search parameter space"""
        return len(self.__search_parameter_keys)

    @property
    def kwargs(self):
        """dict: Container for the **kwargs. Has more sophisticated logic in subclasses"""
        return self.__kwargs

    @kwargs.setter
    def kwargs(self, kwargs):
        self.__kwargs = kwargs

    @property
    def external_sampler(self):
        """Module: An external sampler module imported to this code."""
        return self.__external_sampler

    @external_sampler.setter
    def external_sampler(self, sampler):
        if type(sampler) is str:
            try:
                self.__external_sampler = __import__(sampler)
            except ImportError:
                raise ImportError(
                    "Sampler {} not installed on this system".format(sampler))
        elif isinstance(sampler, Sampler):
            self.__external_sampler = sampler
        else:
            raise TypeError('sampler must either be a string referring to built in sampler or a custom made class that '
                            'inherits from sampler')

    def _verify_kwargs_against_external_sampler_function(self):
        """Check if the kwargs are contained in the list of available arguments of the external sampler."""
        args = inspect.getargspec(self.external_sampler_function).args
        bad_keys = []
        for user_input in self.kwargs.keys():
            if user_input not in args:
                logger.warning(
                    "Supplied argument '{}' not an argument of '{}', removing."
                    .format(user_input, self.external_sampler_function))
                bad_keys.append(user_input)
        for key in bad_keys:
            self.kwargs.pop(key)

    def _initialise_parameters(self):
        """
        Go through the list of priors and add keys to the fixed and search parameter key list depending on whether
        the respective parameter is fixed.

        """
        for key in self.priors:
            if isinstance(self.priors[key], Prior) \
                    and self.priors[key].is_fixed is False:
                self.__search_parameter_keys.append(key)
            elif isinstance(self.priors[key], Prior) \
                    and self.priors[key].is_fixed is True:
                self.likelihood.parameters[key] = \
                    self.priors[key].sample()
                self.__fixed_parameter_keys.append(key)

        logger.info("Search parameters:")
        for key in self.__search_parameter_keys:
            logger.info('  {} = {}'.format(key, self.priors[key]))
        for key in self.__fixed_parameter_keys:
            logger.info('  {} = {}'.format(key, self.priors[key].peak))

    def _initialise_result(self):
        """
        Returns
        -------
        tupak.core.result.Result: An initial template for the result

        """
        result = Result()
        result.sampler = self.__class__.__name__.lower()
        result.search_parameter_keys = self.__search_parameter_keys
        result.fixed_parameter_keys = self.__fixed_parameter_keys
        result.parameter_labels = [
            self.priors[k].latex_label for k in
            self.__search_parameter_keys]
        result.label = self.label
        result.outdir = self.outdir
        result.kwargs = self.kwargs
        return result

    def _check_if_priors_can_be_sampled(self):
        """Check if all priors can be sampled properly. Raises AttributeError if prior can't be sampled."""
        for key in self.priors:
            try:
                self.likelihood.parameters[key] = self.priors[key].sample()
            except AttributeError as e:
                logger.warning('Cannot sample from {}, {}'.format(key, e))

    def _verify_parameters(self):
        """ Sets initial values for likelihood.parameters. Raises TypeError if likelihood can't be evaluated."""
        self._check_if_priors_can_be_sampled()
        try:
            t1 = datetime.datetime.now()
            self.likelihood.log_likelihood()
            self._sample_log_likelihood_eval = (datetime.datetime.now() - t1).total_seconds()
            logger.info("Single likelihood evaluation took {:.3e} s".format(self._sample_log_likelihood_eval))
        except TypeError as e:
            raise TypeError(
                "Likelihood evaluation failed with message: \n'{}'\n"
                "Have you specified all the parameters:\n{}"
                .format(e, self.likelihood.parameters))

    def _verify_use_ratio(self):
        """Checks if use_ratio is set. Prints a warning if use_ratio is set but not properly implemented."""
        self._check_if_priors_can_be_sampled()
        if self.use_ratio is False:
            logger.debug("use_ratio set to False")
            return

        ratio_is_nan = np.isnan(self.likelihood.log_likelihood_ratio())

        if self.use_ratio is True and ratio_is_nan:
            logger.warning(
                "You have requested to use the loglikelihood_ratio, but it "
                " returns a NaN")
        elif self.use_ratio is None and not ratio_is_nan:
            logger.debug(
                "use_ratio not spec. but gives valid answer, setting True")
            self.use_ratio = True

    def prior_transform(self, theta):
        """ Prior transform method that is passed into the external sampler.

        Parameters
        ----------
        theta: list
            List of sampled values on a unit interval

        Returns
        -------
        list: Properly rescaled sampled values
        """
        return self.priors.rescale(self.__search_parameter_keys, theta)

    def log_prior(self, theta):
        """

        Parameters
        ----------
        theta: list
            List of sampled values on a unit interval

        Returns
        -------
        float: TODO: Fill in proper explanation of what this is.

        """
        return self.priors.ln_prob({key: t for key, t in zip(self.__search_parameter_keys, theta)})

    def log_likelihood(self, theta):
        """

        Parameters
        ----------
        theta: list
            List of values for the likelihood parameters

        Returns
        -------
        float: Log-likelihood or log-likelihood-ratio given the current likelihood.parameter values

        """
        for i, k in enumerate(self.__search_parameter_keys):
            self.likelihood.parameters[k] = theta[i]
        if self.use_ratio:
            return self.likelihood.log_likelihood_ratio()
        else:
            return self.likelihood.log_likelihood()

    def get_random_draw_from_prior(self):
        """ Get a random draw from the prior distribution

        Returns
        draw: array_like
            An ndim-length array of values drawn from the prior. Parameters
            with delta-function (or fixed) priors are not returned

        """
        new_sample = self.priors.sample()
        draw = np.array(list(new_sample[key] for key in self.__search_parameter_keys))
        self.check_draw(draw)
        return draw

    def check_draw(self, draw):
        """ Checks if the draw will generate an infinite prior or likelihood """
        if np.isinf(self.log_likelihood(draw)):
            logger.warning('Prior draw {} has inf likelihood'.format(draw))
        if np.isinf(self.log_prior(draw)):
            logger.warning('Prior draw {} has inf prior'.format(draw))

    def _run_external_sampler(self):
        """A template method to run in subclasses"""
        pass

    def _run_test(self):
        """
        TODO: Implement this method
        Raises
        -------
        ValueError: in any case
        """
        raise ValueError("Method not yet implemented")

    def _check_cached_result(self):
        """ Check if the cached data file exists and can be used """

        if utils.command_line_args.clean:
            logger.debug("Command line argument clean given, forcing rerun")
            self.cached_result = None
            return

        try:
            self.cached_result = read_in_result(self.outdir, self.label)
        except ValueError:
            self.cached_result = None

        if utils.command_line_args.use_cached:
            logger.debug("Command line argument cached given, no cache check performed")
            return

        logger.debug("Checking cached data")
        if self.cached_result:
            check_keys = ['search_parameter_keys', 'fixed_parameter_keys',
                          'kwargs']
            use_cache = True
            for key in check_keys:
                if self.cached_result.check_attribute_match_to_other_object(
                        key, self) is False:
                    logger.debug("Cached value {} is unmatched".format(key))
                    use_cache = False
            if use_cache is False:
                self.cached_result = None

    def _log_summary_for_sampler(self):
        """Print a summary of the sampler used and its kwargs"""
        if self.cached_result is None:
            kwargs_print = self.kwargs.copy()
            for k in kwargs_print:
                if type(kwargs_print[k]) in (list, np.ndarray):
                    array_repr = np.array(kwargs_print[k])
                    if array_repr.size > 10:
                        kwargs_print[k] = ('array_like, shape={}'
                                           .format(array_repr.shape))
                elif type(kwargs_print[k]) == pd.core.frame.DataFrame:
                    kwargs_print[k] = ('DataFrame, shape={}'
                                       .format(kwargs_print[k].shape))
            logger.info("Using sampler {} with kwargs {}".format(
                self.__class__.__name__, kwargs_print))


class Nestle(Sampler):

    @property
    def kwargs(self):
        """ Ensures that proper keyword arguments are used for the Nestle sampler.

        Returns
        -------
        dict: Keyword arguments used for the Nestle Sampler

        """
        return self.__kwargs

    @kwargs.setter
    def kwargs(self, kwargs):
        self.__kwargs = dict(verbose=True, method='multi')
        self.__kwargs.update(kwargs)

        if 'npoints' not in self.__kwargs:
            for equiv in ['nlive', 'nlives', 'n_live_points']:
                if equiv in self.__kwargs:
                    self.__kwargs['npoints'] = self.__kwargs.pop(equiv)

    def _run_external_sampler(self):
        """ Runs Nestle sampler with given kwargs and returns the result

        Returns
        -------
        tupak.core.result.Result: Packaged information about the result

        """
        nestle = self.external_sampler
        self.external_sampler_function = nestle.sample
        if 'verbose' in self.kwargs:
            if self.kwargs['verbose']:
                self.kwargs['callback'] = nestle.print_progress
            self.kwargs.pop('verbose')
        self._verify_kwargs_against_external_sampler_function()

        out = self.external_sampler_function(
            loglikelihood=self.log_likelihood,
            prior_transform=self.prior_transform,
            ndim=self.ndim, **self.kwargs)
        print("")

        self.result.sampler_output = out
        self.result.samples = nestle.resample_equal(out.samples, out.weights)
        self.result.log_likelihood_evaluations = out.logl
        self.result.log_evidence = out.logz
        self.result.log_evidence_err = out.logzerr
        return self.result

    def _run_test(self):
        """ Runs to test whether the sampler is properly running with the given kwargs without actually running to the
            end

        Returns
        -------
        tupak.core.result.Result: Dummy container for sampling results.

        """
        nestle = self.external_sampler
        self.external_sampler_function = nestle.sample
        self.external_sampler_function(
            loglikelihood=self.log_likelihood,
            prior_transform=self.prior_transform,
            ndim=self.ndim, maxiter=2, **self.kwargs)
        self.result.samples = np.random.uniform(0, 1, (100, self.ndim))
        self.result.log_evidence = np.nan
        self.result.log_evidence_err = np.nan
        return self.result


class Dynesty(Sampler):

    @property
    def kwargs(self):
        return self.__kwargs

    @kwargs.setter
    def kwargs(self, kwargs):
        self.__kwargs = dict(dlogz=0.1, bound='multi', sample='rwalk', resume=True,
                             walks=self.ndim * 5, verbose=True, check_point_delta_t=60*10)
        self.__kwargs.update(kwargs)
        if 'nlive' not in self.__kwargs:
            for equiv in ['nlives', 'n_live_points', 'npoint', 'npoints']:
                if equiv in self.__kwargs:
                    self.__kwargs['nlive'] = self.__kwargs.pop(equiv)
        if 'nlive' not in self.__kwargs:
            self.__kwargs['nlive'] = 250
        if 'update_interval' not in self.__kwargs:
            self.__kwargs['update_interval'] = int(0.6 * self.__kwargs['nlive'])
        if 'n_check_point' not in kwargs:
            # checkpointing done by default ~ every 10 minutes
            n_check_point_raw = (self.__kwargs['check_point_delta_t']
                                 / self._sample_log_likelihood_eval)
            n_check_point_rnd = int(float("{:1.0g}".format(n_check_point_raw)))
            self.__kwargs['n_check_point'] = n_check_point_rnd

    def _print_func(self, results, niter, ncall, dlogz, *args, **kwargs):
        """ Replacing status update for dynesty.result.print_func """

        # Extract results at the current iteration.
        (worst, ustar, vstar, loglstar, logvol, logwt,
         logz, logzvar, h, nc, worst_it, boundidx, bounditer,
         eff, delta_logz) = results

        # Adjusting outputs for printing.
        if delta_logz > 1e6:
            delta_logz = np.inf
        if logzvar >= 0. and logzvar <= 1e6:
            logzerr = np.sqrt(logzvar)
        else:
            logzerr = np.nan
        if logz <= -1e6:
            logz = -np.inf
        if loglstar <= -1e6:
            loglstar = -np.inf

        if self.use_ratio:
            key = 'logz ratio'
        else:
            key = 'logz'

        # Constructing output.
        print_str = "\r {}| {}={:6.3f} +/- {:6.3f} | dlogz: {:6.3f} > {:6.3f}".format(
            niter, key, logz, logzerr, delta_logz, dlogz)

        # Printing.
        sys.stderr.write(print_str)
        sys.stderr.flush()

    def _run_external_sampler(self):
        dynesty = self.external_sampler

        if self.kwargs.get('dynamic', False) is False:
            nested_sampler = dynesty.NestedSampler(
                loglikelihood=self.log_likelihood,
                prior_transform=self.prior_transform,
                ndim=self.ndim, **self.kwargs)

            if self.kwargs['resume']:
                resume = self.read_saved_state(nested_sampler, continuing=True)
                if resume:
                    logger.info('Resuming from previous run.')

            old_ncall = nested_sampler.ncall
            maxcall = self.kwargs['n_check_point']
            while True:
                maxcall += self.kwargs['n_check_point']
                nested_sampler.run_nested(
                    dlogz=self.kwargs['dlogz'],
                    print_progress=self.kwargs['verbose'],
                    print_func=self._print_func, maxcall=maxcall,
                    add_live=False)
                if nested_sampler.ncall == old_ncall:
                    break
                old_ncall = nested_sampler.ncall

                self.write_current_state(nested_sampler)

            self.read_saved_state(nested_sampler)

            nested_sampler.run_nested(
                dlogz=self.kwargs['dlogz'],
                print_progress=self.kwargs['verbose'],
                print_func=self._print_func, add_live=True)
        else:
            nested_sampler = dynesty.DynamicNestedSampler(
                loglikelihood=self.log_likelihood,
                prior_transform=self.prior_transform,
                ndim=self.ndim, **self.kwargs)
            nested_sampler.run_nested(print_progress=self.kwargs['verbose'])
        print("")
        out = nested_sampler.results

        # self.result.sampler_output = out
        weights = np.exp(out['logwt'] - out['logz'][-1])
        self.result.samples = dynesty.utils.resample_equal(
            out.samples, weights)
        self.result.log_likelihood_evaluations = out.logl
        self.result.log_evidence = out.logz[-1]
        self.result.log_evidence_err = out.logzerr[-1]

        if self.plot:
            self.generate_trace_plots(out)

        self._remove_checkpoint()
        return self.result

    def _remove_checkpoint(self):
        """Remove checkpointed state"""
        if os.path.isfile('{}/{}_resume.h5'.format(self.outdir, self.label)):
            os.remove('{}/{}_resume.h5'.format(self.outdir, self.label))

    def read_saved_state(self, nested_sampler, continuing=False):
        """
        Read a saved state of the sampler to disk.

        The required information to reconstruct the state of the run is read from an hdf5 file.
        This currently adds the whole chain to the sampler.
        We then remove the old checkpoint and write all unnecessary items back to disk.
        FIXME: Load only the necessary quantities, rather than read/write?

        Parameters
        ----------
        nested_sampler: `dynesty.NestedSampler`
            NestedSampler instance to reconstruct from the saved state.
        continuing: bool
            Whether the run is continuing or terminating, if True, the loaded state is mostly
            written back to disk.
        """
        resume_file = '{}/{}_resume.h5'.format(self.outdir, self.label)

        if os.path.isfile(resume_file):
            saved_state = deepdish.io.load(resume_file)

            nested_sampler.saved_u = list(saved_state['unit_cube_samples'])
            nested_sampler.saved_v = list(saved_state['physical_samples'])
            nested_sampler.saved_logl = list(saved_state['sample_likelihoods'])
            nested_sampler.saved_logvol = list(saved_state['sample_log_volume'])
            nested_sampler.saved_logwt = list(saved_state['sample_log_weights'])
            nested_sampler.saved_logz = list(saved_state['cumulative_log_evidence'])
            nested_sampler.saved_logzvar = list(saved_state['cumulative_log_evidence_error'])
            nested_sampler.saved_id = list(saved_state['id'])
            nested_sampler.saved_it = list(saved_state['it'])
            nested_sampler.saved_nc = list(saved_state['nc'])
            nested_sampler.saved_boundidx = list(saved_state['boundidx'])
            nested_sampler.saved_bounditer = list(saved_state['bounditer'])
            nested_sampler.saved_scale = list(saved_state['scale'])
            nested_sampler.saved_h = list(saved_state['cumulative_information'])
            nested_sampler.ncall = saved_state['ncall']
            nested_sampler.live_logl = list(saved_state['live_logl'])
            nested_sampler.it = saved_state['iteration'] + 1
            nested_sampler.live_u = saved_state['live_u']
            nested_sampler.live_v = saved_state['live_v']
            nested_sampler.nlive = saved_state['nlive']
            nested_sampler.live_bound = saved_state['live_bound']
            nested_sampler.live_it = saved_state['live_it']
            nested_sampler.added_live = saved_state['added_live']
            self._remove_checkpoint()
            if continuing:
                self.write_current_state(nested_sampler)
            return True

        else:
            return False

    def write_current_state(self, nested_sampler):
        """
        Write the current state of the sampler to disk.

        The required information to reconstruct the state of the run are written to an hdf5 file.
        All but the most recent removed live point in the chain are removed from the sampler to reduce memory usage.
        This means it is necessary to not append the first live point to the file if updating a previous checkpoint.

        Parameters
        ----------
        nested_sampler: `dynesty.NestedSampler`
            NestedSampler to write to disk.
        """
        resume_file = '{}/{}_resume.h5'.format(self.outdir, self.label)

        if os.path.isfile(resume_file):
            saved_state = deepdish.io.load(resume_file)

            current_state = dict(
                unit_cube_samples=np.vstack([saved_state['unit_cube_samples'], nested_sampler.saved_u[1:]]),
                physical_samples=np.vstack([saved_state['physical_samples'], nested_sampler.saved_v[1:]]),
                sample_likelihoods=np.concatenate([saved_state['sample_likelihoods'], nested_sampler.saved_logl[1:]]),
                sample_log_volume=np.concatenate([saved_state['sample_log_volume'], nested_sampler.saved_logvol[1:]]),
                sample_log_weights=np.concatenate([saved_state['sample_log_weights'], nested_sampler.saved_logwt[1:]]),
                cumulative_log_evidence=np.concatenate([saved_state['cumulative_log_evidence'],
                                                        nested_sampler.saved_logz[1:]]),
                cumulative_log_evidence_error=np.concatenate([saved_state['cumulative_log_evidence_error'],
                                                              nested_sampler.saved_logzvar[1:]]),
                cumulative_information=np.concatenate([saved_state['cumulative_information'],
                                                       nested_sampler.saved_h[1:]]),
                id=np.concatenate([saved_state['id'], nested_sampler.saved_id[1:]]),
                it=np.concatenate([saved_state['it'], nested_sampler.saved_it[1:]]),
                nc=np.concatenate([saved_state['nc'], nested_sampler.saved_nc[1:]]),
                boundidx=np.concatenate([saved_state['boundidx'], nested_sampler.saved_boundidx[1:]]),
                bounditer=np.concatenate([saved_state['bounditer'], nested_sampler.saved_bounditer[1:]]),
                scale=np.concatenate([saved_state['scale'], nested_sampler.saved_scale[1:]]),
            )

        else:
            current_state = dict(
                unit_cube_samples=nested_sampler.saved_u,
                physical_samples=nested_sampler.saved_v,
                sample_likelihoods=nested_sampler.saved_logl,
                sample_log_volume=nested_sampler.saved_logvol,
                sample_log_weights=nested_sampler.saved_logwt,
                cumulative_log_evidence=nested_sampler.saved_logz,
                cumulative_log_evidence_error=nested_sampler.saved_logzvar,
                cumulative_information=nested_sampler.saved_h,
                id=nested_sampler.saved_id,
                it=nested_sampler.saved_it,
                nc=nested_sampler.saved_nc,
                boundidx=nested_sampler.saved_boundidx,
                bounditer=nested_sampler.saved_bounditer,
                scale=nested_sampler.saved_scale,
            )

        current_state.update(
            ncall=nested_sampler.ncall, live_logl=nested_sampler.live_logl, iteration=nested_sampler.it - 1,
            live_u=nested_sampler.live_u, live_v=nested_sampler.live_v, nlive=nested_sampler.nlive,
            live_bound=nested_sampler.live_bound, live_it=nested_sampler.live_it, added_live=nested_sampler.added_live
        )

        weights = np.exp(current_state['sample_log_weights'] - current_state['cumulative_log_evidence'][-1])
        current_state['posterior'] = self.external_sampler.utils.resample_equal(
            np.array(current_state['physical_samples']), weights)

        deepdish.io.save(resume_file, current_state)

        nested_sampler.saved_id = [nested_sampler.saved_id[-1]]
        nested_sampler.saved_u = [nested_sampler.saved_u[-1]]
        nested_sampler.saved_v = [nested_sampler.saved_v[-1]]
        nested_sampler.saved_logl = [nested_sampler.saved_logl[-1]]
        nested_sampler.saved_logvol = [nested_sampler.saved_logvol[-1]]
        nested_sampler.saved_logwt = [nested_sampler.saved_logwt[-1]]
        nested_sampler.saved_logz = [nested_sampler.saved_logz[-1]]
        nested_sampler.saved_logzvar = [nested_sampler.saved_logzvar[-1]]
        nested_sampler.saved_h = [nested_sampler.saved_h[-1]]
        nested_sampler.saved_nc = [nested_sampler.saved_nc[-1]]
        nested_sampler.saved_boundidx = [nested_sampler.saved_boundidx[-1]]
        nested_sampler.saved_it = [nested_sampler.saved_it[-1]]
        nested_sampler.saved_bounditer = [nested_sampler.saved_bounditer[-1]]
        nested_sampler.saved_scale = [nested_sampler.saved_scale[-1]]

    def generate_trace_plots(self, dynesty_results):
        filename = '{}/{}_trace.png'.format(self.outdir, self.label)
        logger.debug("Writing trace plot to {}".format(filename))
        from dynesty import plotting as dyplot
        fig, axes = dyplot.traceplot(dynesty_results,
                                     labels=self.result.parameter_labels)
        fig.tight_layout()
        fig.savefig(filename)

    def _run_test(self):
        dynesty = self.external_sampler
        nested_sampler = dynesty.NestedSampler(
            loglikelihood=self.log_likelihood,
            prior_transform=self.prior_transform,
            ndim=self.ndim, **self.kwargs)
        nested_sampler.run_nested(
            dlogz=self.kwargs['dlogz'],
            print_progress=self.kwargs['verbose'],
            maxiter=2)

        self.result.samples = np.random.uniform(0, 1, (100, self.ndim))
        self.result.log_evidence = np.nan
        self.result.log_evidence_err = np.nan
        return self.result


class Pymultinest(Sampler):

    @property
    def kwargs(self):
        return self.__kwargs

    @kwargs.setter
    def kwargs(self, kwargs):
        outputfiles_basename = self.outdir + '/pymultinest_{}/'.format(self.label)
        utils.check_directory_exists_and_if_not_mkdir(outputfiles_basename)
        self.__kwargs = dict(importance_nested_sampling=False, resume=True,
                             verbose=True, sampling_efficiency='parameter',
                             outputfiles_basename=outputfiles_basename)
        self.__kwargs.update(kwargs)
        if self.__kwargs['outputfiles_basename'].endswith('/') is False:
            self.__kwargs['outputfiles_basename'] = '{}/'.format(
                self.__kwargs['outputfiles_basename'])
        if 'n_live_points' not in self.__kwargs:
            for equiv in ['nlive', 'nlives', 'npoints', 'npoint']:
                if equiv in self.__kwargs:
                    self.__kwargs['n_live_points'] = self.__kwargs.pop(equiv)

    def _run_external_sampler(self):
        pymultinest = self.external_sampler
        self.external_sampler_function = pymultinest.run
        self._verify_kwargs_against_external_sampler_function()
        # Note: pymultinest.solve adds some extra steps, but underneath
        # we are calling pymultinest.run - hence why it is used in checking
        # the arguments.
        out = pymultinest.solve(
            LogLikelihood=self.log_likelihood, Prior=self.prior_transform,
            n_dims=self.ndim, **self.kwargs)

        self.result.sampler_output = out
        self.result.samples = out['samples']
        self.result.log_evidence = out['logZ']
        self.result.log_evidence_err = out['logZerr']
        self.result.outputfiles_basename = self.kwargs['outputfiles_basename']
        return self.result


class Emcee(Sampler):
    """ https://github.com/dfm/emcee """

    def _run_external_sampler(self):
        self.nwalkers = self.kwargs.get('nwalkers', 100)
        self.nsteps = self.kwargs.get('nsteps', 100)
        self.nburn = self.kwargs.get('nburn', 50)
        a = self.kwargs.get('a', 2)
        emcee = self.external_sampler
        tqdm = utils.get_progress_bar(self.kwargs.pop('tqdm', 'tqdm'))

        sampler = emcee.EnsembleSampler(
            nwalkers=self.nwalkers, dim=self.ndim, lnpostfn=self.lnpostfn,
            a=a)

        if 'pos0' in self.kwargs:
            logger.debug("Using given initial positions for walkers")
            pos0 = self.kwargs['pos0']
            if type(pos0) == pd.core.frame.DataFrame:
                pos0 = pos0[self.search_parameter_keys].values
            elif type(pos0) in (list, np.ndarray):
                pos0 = np.squeeze(self.kwargs['pos0'])

            if pos0.shape != (self.nwalkers, self.ndim):
                raise ValueError(
                    'Input pos0 should be of shape ndim, nwalkers')
            logger.debug("Checking input pos0")
            for draw in pos0:
                self.check_draw(draw)
        else:
            logger.debug("Generating initial walker positions from prior")
            pos0 = [self.get_random_draw_from_prior()
                    for i in range(self.nwalkers)]

        for result in tqdm(
                sampler.sample(pos0, iterations=self.nsteps), total=self.nsteps):
            pass

        self.result.sampler_output = np.nan
        self.result.samples = sampler.chain[:, self.nburn:, :].reshape(
            (-1, self.ndim))
        self.result.walkers = sampler.chain[:, :, :]
        self.result.nburn = self.nburn
        self.result.log_evidence = np.nan
        self.result.log_evidence_err = np.nan

        try:
            logger.info("Max autocorr time = {}".format(
                         np.max(sampler.get_autocorr_time())))
        except emcee.autocorr.AutocorrError as e:
            logger.info("Unable to calculate autocorr time: {}".format(e))
        return self.result

    def lnpostfn(self, theta):
        p = self.log_prior(theta)
        if np.isinf(p):
            return -np.inf
        else:
            return self.log_likelihood(theta) + p


class Ptemcee(Emcee):
    """ https://github.com/willvousden/ptemcee """

    def _run_external_sampler(self):
        self.ntemps = self.kwargs.pop('ntemps', 2)
        self.nwalkers = self.kwargs.pop('nwalkers', 100)
        self.nsteps = self.kwargs.pop('nsteps', 100)
        self.nburn = self.kwargs.pop('nburn', 50)
        ptemcee = self.external_sampler
        tqdm = utils.get_progress_bar(self.kwargs.pop('tqdm', 'tqdm'))

        sampler = ptemcee.Sampler(
            ntemps=self.ntemps, nwalkers=self.nwalkers, dim=self.ndim,
            logl=self.log_likelihood, logp=self.log_prior,
            **self.kwargs)
        pos0 = [[self.get_random_draw_from_prior()
                 for i in range(self.nwalkers)]
                for j in range(self.ntemps)]

        for result in tqdm(
                sampler.sample(pos0, iterations=self.nsteps, adapt=True),
                total=self.nsteps):
            pass

        self.result.sampler_output = np.nan
        self.result.samples = sampler.chain[0, :, self.nburn:, :].reshape(
            (-1, self.ndim))
        self.result.walkers = sampler.chain[0, :, :, :]
        self.result.log_evidence = np.nan
        self.result.log_evidence_err = np.nan

        logger.info("Max autocorr time = {}"
                     .format(np.max(sampler.get_autocorr_time())))
        logger.info("Tswap frac = {}"
                     .format(sampler.tswap_acceptance_fraction))
        return self.result


def run_sampler(likelihood, priors=None, label='label', outdir='outdir',
                sampler='dynesty', use_ratio=None, injection_parameters=None,
                conversion_function=None, plot=False, default_priors_file=None,
                clean=None, meta_data=None, **kwargs):
    """
    The primary interface to easy parameter estimation

    Parameters
    ----------
    likelihood: `tupak.Likelihood`
        A `Likelihood` instance
    priors: `tupak.PriorSet`
        A PriorSet/dictionary of the priors for each parameter - missing parameters will
        use default priors, if None, all priors will be default
    label: str
        Name for the run, used in output files
    outdir: str
        A string used in defining output files
    sampler: str
        The name of the sampler to use - see
        `tupak.sampler.get_implemented_samplers()` for a list of available
        samplers
    use_ratio: bool (False)
        If True, use the likelihood's log_likelihood_ratio, rather than just
        the log_likelihood.
    injection_parameters: dict
        A dictionary of injection parameters used in creating the data (if
        using simulated data). Appended to the result object and saved.
    plot: bool
        If true, generate a corner plot and, if applicable diagnostic plots
    conversion_function: function, optional
        Function to apply to posterior to generate additional parameters.
    default_priors_file: str
        If given, a file containing the default priors; otherwise defaults to
        the tupak defaults for a binary black hole.
    clean: bool
        If given, override the command line interface `clean` option.
    meta_data: dict
        If given, adds the key-value pairs to the 'results' object before
        saving. For example, if `meta_data={dtype: 'signal'}`. Warning: in case
        of conflict with keys saved by tupak, the meta_data keys will be
        overwritten.
    **kwargs:
        All kwargs are passed directly to the samplers `run` function

    Returns
    -------
    result
        An object containing the results
    """

    if clean:
        utils.command_line_args.clean = clean

    utils.check_directory_exists_and_if_not_mkdir(outdir)
    implemented_samplers = get_implemented_samplers()

    if priors is None:
        priors = dict()

    if type(priors) == dict:
        priors = tupak.core.prior.PriorSet(priors)
    elif isinstance(priors, tupak.core.prior.PriorSet):
        pass
    else:
        raise ValueError

    priors.fill_priors(likelihood, default_priors_file=default_priors_file)
    priors.write_to_file(outdir, label)

    if implemented_samplers.__contains__(sampler.title()):
        sampler_class = globals()[sampler.title()]
        sampler = sampler_class(likelihood, priors=priors, external_sampler=sampler, outdir=outdir,
                                label=label, use_ratio=use_ratio, plot=plot,
                                **kwargs)

        if sampler.cached_result:
            logger.warning("Using cached result")
            return sampler.cached_result

        start_time = datetime.datetime.now()

        if utils.command_line_args.test:
            result = sampler._run_test()
        else:
            result = sampler._run_external_sampler()

        if type(meta_data) == dict:
            result.update(meta_data)

        end_time = datetime.datetime.now()
        result.sampling_time = (end_time - start_time).total_seconds()
        logger.info('Sampling time: {}'.format(end_time - start_time))

        if sampler.use_ratio:
            result.log_noise_evidence = likelihood.noise_log_likelihood()
            result.log_bayes_factor = result.log_evidence
            result.log_evidence = result.log_bayes_factor + result.log_noise_evidence
        else:
            if likelihood.noise_log_likelihood() is not np.nan:
                result.log_noise_evidence = likelihood.noise_log_likelihood()
                result.log_bayes_factor = result.log_evidence - result.log_noise_evidence
        if injection_parameters is not None:
            result.injection_parameters = injection_parameters
            if conversion_function is not None:
                result.injection_parameters = conversion_function(result.injection_parameters)
        result.fixed_parameter_keys = sampler.fixed_parameter_keys
        # result.prior = prior  # Removed as this breaks the saving of the data
        result.samples_to_posterior(likelihood=likelihood, priors=priors,
                                    conversion_function=conversion_function)
        result.kwargs = sampler.kwargs
        result.save_to_file()
        if plot:
            result.plot_corner()
        logger.info("Sampling finished, results saved to {}/".format(outdir))
        logger.info("Summary of results:\n{}".format(result))
        return result
    else:
        raise ValueError(
            "Sampler {} not yet implemented".format(sampler))


def get_implemented_samplers():
    """ Does some introspection magic to figure out which samplers have been implemented yet.

    Returns
    -------
    list: A list of strings with the names of the implemented samplers

    """
    implemented_samplers = []
    for name, obj in inspect.getmembers(sys.modules[__name__]):
        if inspect.isclass(obj):
            implemented_samplers.append(obj.__name__)
    return implemented_samplers
