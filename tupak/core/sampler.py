from __future__ import print_function, division, absolute_import

import inspect
import os
import sys
import numpy as np
import datetime
import deepdish
import pandas as pd
from collections import OrderedDict

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
        """Check if all priors can be sampled properly.

        Raises
        ------
        AttributeError
            prior can't be sampled.
        """
        for key in self.priors:
            try:
                self.likelihood.parameters[key] = self.priors[key].sample()
            except AttributeError as e:
                logger.warning('Cannot sample from {}, {}'.format(key, e))

    def _verify_parameters(self):
        """ Sets initial values for likelihood.parameters.

        Raises
        ------
        TypeError
            Likelihood can't be evaluated.

        """
        self._check_if_priors_can_be_sampled()
        try:
            t1 = datetime.datetime.now()
            self.likelihood.log_likelihood()
            self._log_likelihood_eval_time = (
                datetime.datetime.now() - t1).total_seconds()
            if self._log_likelihood_eval_time == 0:
                self._log_likelihood_eval_time = np.nan
                logger.info("Unable to measure single likelihood time")
            else:
                logger.info("Single likelihood evaluation took {:.3e} s"
                            .format(self._log_likelihood_eval_time))
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
        # Set some default values
        self.__kwargs = dict(dlogz=0.1, bound='multi', sample='rwalk',
                             resume=True, walks=self.ndim * 5, verbose=True,
                             check_point_delta_t=60 * 10, nlive=250)

        # Overwrite default values with user specified values
        self.__kwargs.update(kwargs)

        # Check if nlive was instead given by another name
        if 'nlive' not in self.__kwargs:
            for equiv in ['nlives', 'n_live_points', 'npoint', 'npoints']:
                if equiv in self.__kwargs:
                    self.__kwargs['nlive'] = self.__kwargs.pop(equiv)

        # Set the update interval
        if 'update_interval' not in self.__kwargs:
            self.__kwargs['update_interval'] = int(0.6 * self.__kwargs['nlive'])

        # Set the checking pointing
        # If the log_likelihood_eval_time was not able to be calculated
        # then n_check_point is set to None (no checkpointing)
        if np.isnan(self._log_likelihood_eval_time):
            self.__kwargs['n_check_point'] = None

        # If n_check_point is not already set, set it checkpoint every 10 mins
        if 'n_check_point' not in self.__kwargs:
            n_check_point_raw = (self.__kwargs['check_point_delta_t']
                                 / self._log_likelihood_eval_time)
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
        if 0. <= logzvar <= 1e6:
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

        nested_sampler = dynesty.NestedSampler(
            loglikelihood=self.log_likelihood,
            prior_transform=self.prior_transform,
            ndim=self.ndim, **self.kwargs)

        if self.kwargs['n_check_point']:
            out = self._run_external_sampler_with_checkpointing(nested_sampler)
        else:
            out = self._run_external_sampler_without_checkpointing(nested_sampler)

        # Flushes the output to force a line break
        if self.kwargs["verbose"]:
            print("")

        # self.result.sampler_output = out
        weights = np.exp(out['logwt'] - out['logz'][-1])
        self.result.samples = dynesty.utils.resample_equal(
            out.samples, weights)
        self.result.log_likelihood_evaluations = out.logl
        self.result.log_evidence = out.logz[-1]
        self.result.log_evidence_err = out.logzerr[-1]

        if self.plot:
            self.generate_trace_plots(out)

        return self.result

    def _run_external_sampler_without_checkpointing(self, nested_sampler):
        logger.debug("Running sampler without checkpointing")
        nested_sampler.run_nested(
            dlogz=self.kwargs['dlogz'],
            print_progress=self.kwargs['verbose'],
            print_func=self._print_func)
        return nested_sampler.results

    def _run_external_sampler_with_checkpointing(self, nested_sampler):
        logger.debug("Running sampler with checkpointing")
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
        self._remove_checkpoint()
        return nested_sampler.results

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
        self.nburn = self.kwargs.get('nburn', None)
        self.burn_in_fraction = self.kwargs.get('burn_in_fraction', 0.25)
        self.burn_in_act = self.kwargs.get('burn_in_act', 3)
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
                    for _ in range(self.nwalkers)]

        for _ in tqdm(
                sampler.sample(pos0, iterations=self.nsteps), total=self.nsteps):
            pass

        self.result.sampler_output = np.nan
        self.calculate_autocorrelation(sampler)
        self.setup_nburn()
        self.result.nburn = self.nburn
        self.result.samples = sampler.chain[:, self.nburn:, :].reshape(
            (-1, self.ndim))
        self.result.walkers = sampler.chain[:, :, :]
        self.result.log_evidence = np.nan
        self.result.log_evidence_err = np.nan
        return self.result

    def lnpostfn(self, theta):
        p = self.log_prior(theta)
        if np.isinf(p):
            return -np.inf
        else:
            return self.log_likelihood(theta) + p

    def setup_nburn(self):
        """ Handles calculating nburn, either from a given value or inferred """
        if type(self.nburn) in [float, int]:
            self.nburn = int(self.nburn)
            logger.info("Discarding {} steps for burn-in".format(self.nburn))
        elif self.result.max_autocorrelation_time is None:
            self.nburn = int(self.burn_in_fraction * self.nsteps)
            logger.info("Autocorrelation time not calculated, discarding {} "
                        " steps for burn-in".format(self.nburn))
        else:
            self.nburn = int(
                self.burn_in_act * self.result.max_autocorrelation_time)
            logger.info("Discarding {} steps for burn-in, estimated from "
                        "autocorr".format(self.nburn))

    def calculate_autocorrelation(self, sampler, c=3):
        """ Uses the `emcee.autocorr` module to estimate the autocorrelation

        Parameters
        ----------
        c: float
            The minimum number of autocorrelation times needed to trust the
            estimate (default: `3`). See `emcee.autocorr.integrated_time`.
        """

        import emcee
        try:
            self.result.max_autocorrelation_time = int(np.max(
                sampler.get_autocorr_time(c=c)))
            logger.info("Max autocorr time = {}".format(
                self.result.max_autocorrelation_time))
        except emcee.autocorr.AutocorrError as e:
            self.result.max_autocorrelation_time = None
            logger.info("Unable to calculate autocorr time: {}".format(e))


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
                 for _ in range(self.nwalkers)]
                for _ in range(self.ntemps)]

        for _ in tqdm(
                sampler.sample(pos0, iterations=self.nsteps, adapt=True),
                total=self.nsteps):
            pass

        self.result.nburn = self.nburn
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


class Pymc3(Sampler):
    """ https://docs.pymc.io/ """

    def _verify_parameters(self):
        """
        Change `_verify_parameters()` to just pass, i.e., don't try and
        evaluate the likelihood for PyMC3.
        """
        pass

    def _verify_use_ratio(self):
        """
        Change `_verify_use_ratio() to just pass.
        """
        pass

    def _initialise_parameters(self):
        """
        Change `_initialise_parameters()`, so that it does call the `sample`
        method in the Prior class.

        """

        self.__search_parameter_keys = []
        self.__fixed_parameter_keys = []

        for key in self.priors:
            if isinstance(self.priors[key], Prior) \
                    and self.priors[key].is_fixed is False:
                self.__search_parameter_keys.append(key)
            elif isinstance(self.priors[key], Prior) \
                    and self.priors[key].is_fixed is True:
                self.__fixed_parameter_keys.append(key)

        logger.info("Search parameters:")
        for key in self.__search_parameter_keys:
            logger.info('  {} = {}'.format(key, self.priors[key]))
        for key in self.__fixed_parameter_keys:
            logger.info('  {} = {}'.format(key, self.priors[key].peak))

    def _initialise_result(self):
        """
        Initialise results within Pymc3 subclass.
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

    @property
    def kwargs(self):
        """ Ensures that proper keyword arguments are used for the Pymc3 sampler.

        Returns
        -------
        dict: Keyword arguments used for the Nestle Sampler

        """
        return self.__kwargs

    @kwargs.setter
    def kwargs(self, kwargs):
        self.__kwargs = dict()
        self.__kwargs.update(kwargs)

        # set some defaults

        # set the number of draws
        self.draws = 1000 if 'draws' not in self.__kwargs else self.__kwargs.pop('draws')

        if 'chains' not in self.__kwargs:
            self.__kwargs['chains'] = 2
            self.chains = self.__kwargs['chains']

        if 'cores' not in self.__kwargs:
            self.__kwargs['cores'] = 1

    def setup_prior_mapping(self):
        """
        Set the mapping between predefined tupak priors and the equivalent
        PyMC3 distributions.
        """

        prior_map = {}
        self.prior_map = prior_map
        
        # predefined PyMC3 distributions 
        prior_map['Gaussian'] =          {'pymc3': 'Normal',
                                          'argmap': {'mu': 'mu', 'sigma': 'sd'}}
        prior_map['TruncatedGaussian'] = {'pymc3': 'TruncatedNormal',
                                          'argmap': {'mu': 'mu', 'sigma': 'sd', 'minimum': 'lower', 'maximum': 'upper'}}
        prior_map['HalfGaussian'] =      {'pymc3': 'HalfNormal',
                                          'argmap': {'sigma': 'sd'}}
        prior_map['Uniform'] =           {'pymc3': 'Uniform',
                                          'argmap': {'minimum': 'lower', 'maximum': 'upper'}}
        prior_map['LogNormal'] =         {'pymc3': 'Lognormal',
                                          'argmap': {'mu': 'mu', 'sigma': 'sd'}}
        prior_map['Exponential'] =       {'pymc3': 'Exponential',
                                          'argmap': {'mu': 'lam'},
                                          'argtransform': {'mu': lambda mu: 1./mu}}
        prior_map['StudentT'] =          {'pymc3': 'StudentT',
                                          'argmap': {'df': 'nu', 'mu': 'mu', 'scale': 'sd'}}
        prior_map['Beta'] =              {'pymc3': 'Beta',
                                          'argmap': {'alpha': 'alpha', 'beta': 'beta'}}
        prior_map['Logistic'] =          {'pymc3': 'Logistic',
                                          'argmap': {'mu': 'mu', 'scale': 's'}}
        prior_map['Cauchy'] =            {'pymc3': 'Cauchy',
                                          'argmap': {'alpha': 'alpha', 'beta': 'beta'}}
        prior_map['Gamma'] =             {'pymc3': 'Gamma',
                                          'argmap': {'k': 'alpha', 'theta': 'beta'},
                                          'argtransform': {'theta': lambda theta: 1./theta}}
        prior_map['ChiSquared'] =        {'pymc3': 'ChiSquared',
                                          'argmap': {'nu': 'nu'}}
        prior_map['Interped'] =          {'pymc3': 'Interpolated',
                                          'argmap': {'xx': 'x_points', 'yy': 'pdf_points'}}
        prior_map['Normal'] = prior_map['Gaussian']
        prior_map['TruncatedNormal'] = prior_map['TruncatedGaussian']
        prior_map['HalfNormal'] = prior_map['HalfGaussian']
        prior_map['LogGaussian'] = prior_map['LogNormal']
        prior_map['Lorentzian'] = prior_map['Cauchy']
        prior_map['FromFile'] = prior_map['Interped']

        # internally defined mappings for tupak priors
        prior_map['DeltaFunction'] = {'internal': self._deltafunction_prior}
        prior_map['Sine'] =          {'internal': self._sine_prior}
        prior_map['Cosine'] =        {'internal': self._cosine_prior}
        prior_map['PowerLaw'] =      {'internal': self._powerlaw_prior}
        prior_map['LogUniform'] =    {'internal': self._powerlaw_prior}

    def _deltafunction_prior(self, key, **kwargs):
        """
        Map the tupak delta function prior to a single value for PyMC3.
        """

        from tupak.core.prior import DeltaFunction

        # check prior is a DeltaFunction
        if isinstance(self.priors[key], DeltaFunction):
            return self.priors[key].peak
        else:
            raise ValueError("Prior for '{}' is not a DeltaFunction".format(key))

    def _sine_prior(self, key):
        """
        Map the tupak Sine prior to a PyMC3 style function
        """
 
        from tupak.core.prior import Sine

        # check prior is a Sine
        if isinstance(self.priors[key], Sine):
            pymc3 = self.external_sampler

            try:
                import theano.tensor as tt
                from pymc3.theanof import floatX
            except ImportError:
                raise ImportError("You must have Theano installed to use PyMC3")

            class Pymc3Sine(pymc3.Continuous):
                def __init__(self, lower=0., upper=np.pi):
                    if lower >= upper:
                        raise ValueError("Lower bound is above upper bound!")

                    # set the mode
                    self.lower = lower = tt.as_tensor_variable(floatX(lower))
                    self.upper = upper = tt.as_tensor_variable(floatX(upper))
                    self.norm = (tt.cos(lower) - tt.cos(upper))
                    self.mean = (tt.sin(upper)+lower*tt.cos(lower) - tt.sin(lower) - upper*tt.cos(upper))/self.norm

                    transform = pymc3.distributions.transforms.interval(lower, upper)

                    super(Pymc3Sine, self).__init__(transform=transform)

                def logp(self, value):
                    upper = self.upper
                    lower = self.lower
                    return pymc3.distributions.dist_math.bound(tt.log(tt.sin(value)/self.norm), lower <= value, value <= upper)

            return Pymc3Sine(key, lower=self.priors[key].minimum, upper=self.priors[key].maximum)
        else:
            raise ValueError("Prior for '{}' is not a Sine".format(key))

    def _cosine_prior(self, key):
        """
        Map the tupak Cosine prior to a PyMC3 style function
        """
 
        from tupak.core.prior import Cosine

        # check prior is a Cosine
        if isinstance(self.priors[key], Cosine):
            pymc3 = self.external_sampler

            # import theano
            try:
                import theano.tensor as tt
                from pymc3.theanof import floatX
            except ImportError:
                raise ImportError("You must have Theano installed to use PyMC3")

            class Pymc3Cosine(pymc3.Continuous):
                def __init__(self, lower=-np.pi/2., upper=np.pi/2.):
                    if lower >= upper:
                        raise ValueError("Lower bound is above upper bound!")

                    self.lower = lower = tt.as_tensor_variable(floatX(lower))
                    self.upper = upper = tt.as_tensor_variable(floatX(upper))
                    self.norm = (tt.sin(upper) - tt.sin(lower))
                    self.mean = (upper*tt.sin(upper) + tt.cos(upper)-lower*tt.sin(lower)-tt.cos(lower))/self.norm

                    transform = pymc3.distributions.transforms.interval(lower, upper)

                    super(Pymc3Cosine, self).__init__(transform=transform)

                def logp(self, value):
                    upper = self.upper
                    lower = self.lower
                    return pymc3.distributions.dist_math.bound(tt.log(tt.cos(value)/self.norm), lower <= value, value <= upper)

            return Pymc3Cosine(key, lower=self.priors[key].minimum, upper=self.priors[key].maximum)
        else:
            raise ValueError("Prior for '{}' is not a Cosine".format(key))

    def _powerlaw_prior(self, key):
        """
        Map the tupak PowerLaw prior to a PyMC3 style function
        """
 
        from tupak.core.prior import PowerLaw

        # check prior is a PowerLaw
        if isinstance(self.priors[key], PowerLaw):
            pymc3 = self.external_sampler

            # check power law is set
            if not hasattr(self.priors[key], 'alpha'):
                raise AttributeError("No 'alpha' attribute set for PowerLaw prior")

            # import theano
            try:
                import theano.tensor as tt
                from pymc3.theanof import floatX
            except ImportError:
                raise ImportError("You must have Theano installed to use PyMC3")

            if self.priors[key].alpha < -1.:
                # use Pareto distribution
                palpha = -(1. + self.priors[key].alpha)

                return pymc3.Bound(pymc3.Pareto, upper=self.priors[key].minimum)(key, alpha=palpha, m=self.priors[key].maximum)
            else:
                class Pymc3PowerLaw(pymc3.Continuous):
                    def __init__(self, lower, upper, alpha, testval=1):
                        falpha = alpha
                        self.lower = lower = tt.as_tensor_variable(floatX(lower))
                        self.upper = upper = tt.as_tensor_variable(floatX(upper))
                        self.alpha = alpha = tt.as_tensor_variable(floatX(alpha))

                        if falpha == -1:
                            self.norm = 1./(tt.log(self.upper/self.lower))
                        else:
                            beta = (1. + self.alpha)
                            self.norm = 1. /(beta * (tt.pow(self.upper, beta) 
                                          - tt.pow(self.lower, beta)))

                        transform = pymc3.distributions.transforms.interval(lower, upper)

                        super(Pymc3PowerLaw, self).__init__(transform=transform, testval=testval)

                    def logp(self, value):
                        upper = self.upper
                        lower = self.lower
                        alpha = self.alpha

                        return pymc3.distributions.dist_math.bound(self.alpha*tt.log(value) + tt.log(self.norm), lower <= value, value <= upper)

                return Pymc3PowerLaw(key, lower=self.priors[key].minimum, upper=self.priors[key].maximum, alpha=self.priors[key].alpha)
        else:
            raise ValueError("Prior for '{}' is not a Power Law".format(key))

    def _run_external_sampler(self):
        pymc3 = self.external_sampler

        # set the step method
        from pymc3.sampling import STEP_METHODS

        step_methods = {m.__name__.lower(): m.__name__ for m in STEP_METHODS}
        if 'step' in self.__kwargs:
            step_method = self.__kwargs.pop('step').lower()

            if step_method not in step_methods:
                raise ValueError("Using invalid step method '{}'".format(step_method))
        else:
            step_method = None

        # initialise the PyMC3 model
        self.pymc3_model = pymc3.Model()

        # set the step method
        sm = None if step_method is None else pymc3.__dict__[step_methods[step_method]]()

        # set the prior
        self.set_prior()

        # if a custom log_likelihood function requires a `sampler` argument
        # then use that log_likelihood function, with the assumption that it
        # takes in a Pymc3 Sampler, with a pymc3_model attribute, and defines
        # the likelihood within that context manager
        likeargs = inspect.getargspec(self.likelihood.log_likelihood).args
        if 'sampler' in likeargs:
            self.likelihood.log_likelihood(sampler=self)
        else:
            # set the likelihood function from predefined functions
            self.set_likelihood()

        with self.pymc3_model:
            # perform the sampling
            trace = pymc3.sample(self.draws, step=sm, **self.kwargs)

        nparams = len([key for key in self.priors.keys() if self.priors[key].__class__.__name__ != 'DeltaFunction'])
        nsamples = len(trace)*self.chains

        self.result.samples = np.zeros((nsamples, nparams))
        count = 0
        for key in self.priors.keys():
            if self.priors[key].__class__.__name__ != 'DeltaFunction': # ignore DeltaFunction variables
                self.result.samples[:,count] = trace[key]
                count += 1

        self.result.sampler_output = np.nan
        self.calculate_autocorrelation(self.result.samples)
        self.result.log_evidence = np.nan
        self.result.log_evidence_err = np.nan
        return self.result

    def set_prior(self):
        """
        Set the PyMC3 prior distributions.
        """

        self.setup_prior_mapping()

        self.pymc3_priors = dict()

        pymc3 = self.external_sampler

        # set the parameter prior distributions (in the model context manager)
        with self.pymc3_model:
            for key in self.priors:
                # if the prior contains ln_prob method that takes a 'sampler' argument
                # then try using that
                lnprobargs = inspect.getargspec(self.priors[key].ln_prob).args
                if 'sampler' in lnprobargs:
                    try:
                        self.pymc3_priors[key] = self.priors[key].ln_prob(sampler=self)
                    except RuntimeError:
                        raise RuntimeError(("Problem setting PyMC3 prior for ",
                            "'{}'".format(key)))
                else:
                    # use Prior distribution name
                    distname = self.priors[key].__class__.__name__

                    if distname in self.prior_map:
                        # check if we have a predefined PyMC3 distribution
                        if 'pymc3' in self.prior_map[distname] and 'argmap' in self.prior_map[distname]:
                            # check the required arguments for the PyMC3 distribution
                            pymc3distname = self.prior_map[distname]['pymc3']

                            if pymc3distname not in pymc3.__dict__:
                                raise ValueError("Prior '{}' is not a known PyMC3 distribution.".format(pymc3distname))

                            reqargs = inspect.getargspec(pymc3.__dict__[pymc3distname].__init__).args[1:]

                            # set keyword arguments
                            priorkwargs = {}
                            for (targ, parg) in self.prior_map[distname]['argmap'].items():
                                if hasattr(self.priors[key], targ):
                                    if parg in reqargs:
                                        if 'argtransform' in self.prior_map[distname]:
                                            if targ in self.prior_map[distname]['argtransform']:
                                                tfunc = self.prior_map[distname]['argtransform'][targ]
                                            else:
                                                tfunc = lambda x: x
                                        else:
                                            tfunc = lambda x: x

                                        priorkwargs[parg] = tfunc(getattr(self.priors[key], targ))
                                    else:
                                        raise ValueError("Unknown argument {}".format(parg))
                                else:
                                    if parg in reqargs:
                                        priorkwargs[parg] = None
                            self.pymc3_priors[key] = pymc3.__dict__[pymc3distname](key, **priorkwargs)
                        elif 'internal' in self.prior_map[distname]:
                            self.pymc3_priors[key] = self.prior_map[distname]['internal'](key)
                        else:
                            raise ValueError("Prior '{}' is not a known distribution.".format(distname))
                    else:
                        raise ValueError("Prior '{}' is not a known distribution.".format(distname))

    def set_likelihood(self):
        """
        Convert any tupak likelihoods to PyMC3 distributions.
        """

        pymc3 = self.external_sampler

        with self.pymc3_model:
            #  check if it is a predefined likelhood function
            if self.likelihood.__class__.__name__ == 'GaussianLikelihood':
                # check required attributes exist
                if (not hasattr(self.likelihood, 'sigma') or
                    not hasattr(self.likelihood, 'x') or
                    not hasattr(self.likelihood, 'y') or
                    not hasattr(self.likelihood, 'function') or
                    not hasattr(self.likelihood, 'function_keys')):
                    raise ValueError("Gaussian Likelihood does not have all the correct attributes!")
                
                if 'sigma' in self.pymc3_priors:
                    # if sigma is suppled use that value
                    if self.likelihood.sigma is None:
                        self.likelihood.sigma = self.pymc3_priors.pop('sigma')
                    else:
                        del self.pymc3_priors['sigma']

                for key in self.pymc3_priors:
                    if key not in self.likelihood.function_keys:
                        raise ValueError("Prior key '{}' is not a function key!".format(key))

                model = self.likelihood.function(self.likelihood.x, **self.pymc3_priors)

                # set the distribution
                pymc3.Normal('likelihood', mu=model, sd=self.likelihood.sigma,
                             observed=self.likelihood.y)
            elif self.likelihood.__class__.__name__ == 'PoissonLikelihood':
                # check required attributes exist
                if (not hasattr(self.likelihood, 'x') or
                    not hasattr(self.likelihood, 'y') or
                    not hasattr(self.likelihood, 'function') or
                    not hasattr(self.likelihood, 'function_keys')):
                    raise ValueError("Poisson Likelihood does not have all the correct attributes!")
                
                for key in self.pymc3_priors:
                    if key not in self.likelihood.function_keys:
                        raise ValueError("Prior key '{}' is not a function key!".format(key))

                # get rate function
                model = self.likelihood.function(self.likelihood.x, **self.pymc3_priors)

                # set the distribution
                pymc3.Poisson('likelihood', mu=model, observed=self.likelihood.y)
            elif self.likelihood.__class__.__name__ == 'ExponentialLikelihood':
                # check required attributes exist
                if (not hasattr(self.likelihood, 'x') or
                    not hasattr(self.likelihood, 'y') or
                    not hasattr(self.likelihood, 'function') or
                    not hasattr(self.likelihood, 'function_keys')):
                    raise ValueError("Exponential Likelihood does not have all the correct attributes!")

                for key in self.pymc3_priors:
                    if key not in self.likelihood.function_keys:
                        raise ValueError("Prior key '{}' is not a function key!".format(key))

                # get mean function
                model = self.likelihood.function(self.likelihood.x, **self.pymc3_priors)

                # set the distribution
                pymc3.Exponential('likelihood', lam=1./model, observed=self.likelihood.y)
            elif self.likelihood.__class__.__name__ == 'StudentTLikelihood':
                # check required attributes exist
                if (not hasattr(self.likelihood, 'x') or
                    not hasattr(self.likelihood, 'y') or
                    not hasattr(self.likelihood, 'nu') or
                    not hasattr(self.likelihood, 'sigma') or
                    not hasattr(self.likelihood, 'function') or
                    not hasattr(self.likelihood, 'function_keys')):
                    raise ValueError("StudentT Likelihood does not have all the correct attributes!")

                if 'nu' in self.pymc3_priors:
                    # if nu is suppled use that value
                    if self.likelihood.nu is None:
                        self.likelihood.nu = self.pymc3_priors.pop('nu')
                    else:
                        del self.pymc3_priors['nu']

                for key in self.pymc3_priors:
                    if key not in self.likelihood.function_keys:
                        raise ValueError("Prior key '{}' is not a function key!".format(key))

                model = self.likelihood.function(self.likelihood.x, **self.pymc3_priors)

                # set the distribution
                pymc3.StudentT('likelihood', nu=self.likelihood.nu, mu=model, sd=self.likelihood.sigma, observed=self.likelihood.y)
            else:
                raise ValueError("Unknown likelihood has been provided")

    def calculate_autocorrelation(self, samples, c=3):
        """ Uses the `emcee.autocorr` module to estimate the autocorrelation

        Parameters
        ----------
        c: float
            The minimum number of autocorrelation times needed to trust the
            estimate (default: `3`). See `emcee.autocorr.integrated_time`.
        """

        import emcee
        try:
            self.result.max_autocorrelation_time = int(np.max(
                emcee.autocorr.integrated_time(samples, c=c)))
            logger.info("Max autocorr time = {}".format(
                self.result.max_autocorrelation_time))
        except emcee.autocorr.AutocorrError as e:
            self.result.max_autocorrelation_time = None
            logger.info("Unable to calculate autocorr time: {}".format(e))


def run_sampler(likelihood, priors=None, label='label', outdir='outdir',
                sampler='dynesty', use_ratio=None, injection_parameters=None,
                conversion_function=None, plot=False, default_priors_file=None,
                clean=None, meta_data=None, save=True, **kwargs):
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
    save: bool
        If true, save the priors and results to disk.
    **kwargs:
        All kwargs are passed directly to the samplers `run` function

    Returns
    -------
    result
        An object containing the results
    """

    if clean:
        utils.command_line_args.clean = clean

    implemented_samplers = get_implemented_samplers()

    if priors is None:
        priors = dict()

    if type(priors) in [dict, OrderedDict]:
        priors = tupak.core.prior.PriorSet(priors)
    elif isinstance(priors, tupak.core.prior.PriorSet):
        pass
    else:
        raise ValueError("Input priors not understood")

    priors.fill_priors(likelihood, default_priors_file=default_priors_file)

    if save:
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
        result.samples_to_posterior(likelihood=likelihood, priors=priors,
                                    conversion_function=conversion_function)
        result.kwargs = sampler.kwargs
        if save:
            result.save_to_file()
            logger.info("Results saved to {}/".format(outdir))
        if plot:
            result.plot_corner()
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
