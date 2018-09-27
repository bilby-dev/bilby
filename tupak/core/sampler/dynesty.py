import os
import sys
import numpy as np
from pandas import DataFrame
from deepdish.io import load, save
from ..utils import logger
from .base_sampler import Sampler


class Dynesty(Sampler):
    """
    tupak wrapper of `dynesty.NestedSampler`
    (https://dynesty.readthedocs.io/en/latest/)

    All positional and keyword arguments (i.e., the args and kwargs) passed to
    `run_sampler` will be propagated to `dynesty.NestedSampler`, see
    documentation for that class for further help. Under Keyword Arguments, we
    list commonly used kwargs and the tupak defaults.

    Keyword Arguments
    -----------------
    npoints: int, (250)
        The number of live points, note this can also equivalently be given as
        one of [nlive, nlives, n_live_points]
    bound: {'none', 'single', 'multi', 'balls', 'cubes'}, ('multi')
        Method used to select new points
    sample: {'unif', 'rwalk', 'slice', 'rslice', 'hslice'}, ('rwalk')
        Method used to sample uniformly within the likelihood constraints,
        conditioned on the provided bounds
    walks: int
        Number of walks taken if using `sample='rwalk'`, defaults to `ndim * 5`
    dlogz: float, (0.1)
        Stopping criteria
    verbose: Bool
        If true, print information information about the convergence during
    check_point_delta_t: float (600)
        The approximate checkpoint period (in seconds). Should the run be
        interrupted, it can be resumed from the last checkpoint. Set to
        `None` to turn-off check pointing
    resume: bool
        If true, resume run from checkpoint (if available)
    """

    @property
    def kwargs(self):
        return self.__kwargs

    @kwargs.setter
    def kwargs(self, kwargs):
        # Set some default values
        self.__kwargs = dict(dlogz=0.1, bound='multi', sample='rwalk',
                             resume=True, walks=self.ndim * 5, verbose=True,
                             check_point_delta_t=60 * 10, nlive=250)

        # Check if nlive was instead given by another name
        if 'nlive' not in kwargs:
            for equiv in ['nlives', 'n_live_points', 'npoint', 'npoints']:
                if equiv in kwargs:
                    kwargs['nlive'] = kwargs.pop(equiv)

        # Overwrite default values with user specified values
        self.__kwargs.update(kwargs)

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
            n_check_point_raw = (self.__kwargs['check_point_delta_t'] /
                                 self._log_likelihood_eval_time)
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
        raw_string = "\r {}| {}={:6.3f} +/- {:6.3f} | dlogz: {:6.3f} > {:6.3f}"
        print_str = raw_string.format(
            niter, key, logz, logzerr, delta_logz, dlogz)

        # Printing.
        sys.stderr.write(print_str)
        sys.stderr.flush()

    def _run_external_sampler(self):
        dynesty = self.external_sampler

        sampler = dynesty.NestedSampler(
            loglikelihood=self.log_likelihood,
            prior_transform=self.prior_transform,
            ndim=self.ndim, **self.kwargs)

        if self.kwargs['n_check_point']:
            out = self._run_external_sampler_with_checkpointing(sampler)
        else:
            out = self._run_external_sampler_without_checkpointing(sampler)

        # Flushes the output to force a line break
        if self.kwargs["verbose"]:
            print("")

        # self.result.sampler_output = out
        weights = np.exp(out['logwt'] - out['logz'][-1])
        self.result.samples = dynesty.utils.resample_equal(out.samples, weights)
        self.result.nested_samples = DataFrame(
            out.samples, columns=self.search_parameter_keys)
        self.result.nested_samples['weights'] = weights
        self.result.nested_samples['log_likelihood'] = out.logl
        idxs = [np.unique(np.where(self.result.samples[ii] == out.samples)[0])
                for ii in range(len(out.logl))]
        self.result.log_likelihood_evaluations = out.logl[idxs]
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

    def read_saved_state(self, sampler, continuing=False):
        """
        Read a saved state of the sampler to disk.

        The required information to reconstruct the state of the run is read
        from an hdf5 file.
        This currently adds the whole chain to the sampler.
        We then remove the old checkpoint and write all unnecessary items back
        to disk.
        FIXME: Load only the necessary quantities, rather than read/write?

        Parameters
        ----------
        sampler: `dynesty.NestedSampler`
            NestedSampler instance to reconstruct from the saved state.
        continuing: bool
            Whether the run is continuing or terminating, if True, the loaded
            state is mostly written back to disk.
        """
        resume_file = '{}/{}_resume.h5'.format(self.outdir, self.label)

        if os.path.isfile(resume_file):
            saved = load(resume_file)

            sampler.saved_u = list(saved['unit_cube_samples'])
            sampler.saved_v = list(saved['physical_samples'])
            sampler.saved_logl = list(saved['sample_likelihoods'])
            sampler.saved_logvol = list(saved['sample_log_volume'])
            sampler.saved_logwt = list(saved['sample_log_weights'])
            sampler.saved_logz = list(saved['cumulative_log_evidence'])
            sampler.saved_logzvar = list(saved['cumulative_log_evidence_error'])
            sampler.saved_id = list(saved['id'])
            sampler.saved_it = list(saved['it'])
            sampler.saved_nc = list(saved['nc'])
            sampler.saved_boundidx = list(saved['boundidx'])
            sampler.saved_bounditer = list(saved['bounditer'])
            sampler.saved_scale = list(saved['scale'])
            sampler.saved_h = list(saved['cumulative_information'])
            sampler.ncall = saved['ncall']
            sampler.live_logl = list(saved['live_logl'])
            sampler.it = saved['iteration'] + 1
            sampler.live_u = saved['live_u']
            sampler.live_v = saved['live_v']
            sampler.nlive = saved['nlive']
            sampler.live_bound = saved['live_bound']
            sampler.live_it = saved['live_it']
            sampler.added_live = saved['added_live']
            self._remove_checkpoint()
            if continuing:
                self.write_current_state(sampler)
            return True

        else:
            return False

    def write_current_state(self, sampler):
        """
        Write the current state of the sampler to disk.

        The required information to reconstruct the state of the run are written
        to an hdf5 file.
        All but the most recent removed live point in the chain are removed from
        the sampler to reduce memory usage.
        This means it is necessary to not append the first live point to the
        file if updating a previous checkpoint.

        Parameters
        ----------
        sampler: `dynesty.NestedSampler`
            NestedSampler to write to disk.
        """
        resume_file = '{}/{}_resume.h5'.format(self.outdir, self.label)

        if os.path.isfile(resume_file):
            saved = load(resume_file)

            current_state = dict(
                unit_cube_samples=np.vstack([
                    saved['unit_cube_samples'], sampler.saved_u[1:]]),
                physical_samples=np.vstack([
                    saved['physical_samples'], sampler.saved_v[1:]]),
                sample_likelihoods=np.concatenate([
                    saved['sample_likelihoods'], sampler.saved_logl[1:]]),
                sample_log_volume=np.concatenate([
                    saved['sample_log_volume'], sampler.saved_logvol[1:]]),
                sample_log_weights=np.concatenate([
                    saved['sample_log_weights'], sampler.saved_logwt[1:]]),
                cumulative_log_evidence=np.concatenate([
                    saved['cumulative_log_evidence'], sampler.saved_logz[1:]]),
                cumulative_log_evidence_error=np.concatenate([
                    saved['cumulative_log_evidence_error'],
                    sampler.saved_logzvar[1:]]),
                cumulative_information=np.concatenate([
                    saved['cumulative_information'], sampler.saved_h[1:]]),
                id=np.concatenate([saved['id'], sampler.saved_id[1:]]),
                it=np.concatenate([saved['it'], sampler.saved_it[1:]]),
                nc=np.concatenate([saved['nc'], sampler.saved_nc[1:]]),
                boundidx=np.concatenate([
                    saved['boundidx'], sampler.saved_boundidx[1:]]),
                bounditer=np.concatenate([
                    saved['bounditer'], sampler.saved_bounditer[1:]]),
                scale=np.concatenate([saved['scale'], sampler.saved_scale[1:]]),
            )

        else:
            current_state = dict(
                unit_cube_samples=sampler.saved_u,
                physical_samples=sampler.saved_v,
                sample_likelihoods=sampler.saved_logl,
                sample_log_volume=sampler.saved_logvol,
                sample_log_weights=sampler.saved_logwt,
                cumulative_log_evidence=sampler.saved_logz,
                cumulative_log_evidence_error=sampler.saved_logzvar,
                cumulative_information=sampler.saved_h,
                id=sampler.saved_id,
                it=sampler.saved_it,
                nc=sampler.saved_nc,
                boundidx=sampler.saved_boundidx,
                bounditer=sampler.saved_bounditer,
                scale=sampler.saved_scale,
            )

        current_state.update(
            ncall=sampler.ncall, live_logl=sampler.live_logl,
            iteration=sampler.it - 1, live_u=sampler.live_u,
            live_v=sampler.live_v, nlive=sampler.nlive,
            live_bound=sampler.live_bound, live_it=sampler.live_it,
            added_live=sampler.added_live
        )

        weights = np.exp(current_state['sample_log_weights'] -
                         current_state['cumulative_log_evidence'][-1])
        current_state['posterior'] = self.external_sampler.utils.resample_equal(
            np.array(current_state['physical_samples']), weights)

        save(resume_file, current_state)

        sampler.saved_id = [sampler.saved_id[-1]]
        sampler.saved_u = [sampler.saved_u[-1]]
        sampler.saved_v = [sampler.saved_v[-1]]
        sampler.saved_logl = [sampler.saved_logl[-1]]
        sampler.saved_logvol = [sampler.saved_logvol[-1]]
        sampler.saved_logwt = [sampler.saved_logwt[-1]]
        sampler.saved_logz = [sampler.saved_logz[-1]]
        sampler.saved_logzvar = [sampler.saved_logzvar[-1]]
        sampler.saved_h = [sampler.saved_h[-1]]
        sampler.saved_nc = [sampler.saved_nc[-1]]
        sampler.saved_boundidx = [sampler.saved_boundidx[-1]]
        sampler.saved_it = [sampler.saved_it[-1]]
        sampler.saved_bounditer = [sampler.saved_bounditer[-1]]
        sampler.saved_scale = [sampler.saved_scale[-1]]

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
