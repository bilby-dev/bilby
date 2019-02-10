from __future__ import absolute_import

import numpy as np

from .base_sampler import NestedSampler


class PyPolyChord(NestedSampler):

    """
    Bilby wrapper of PyPolyChord
    https://arxiv.org/abs/1506.00171

    PolyChordLite is available at:
    https://github.com/PolyChord/PolyChordLite

    Follow the installation instructions at their github page.

    Keyword arguments will be passed into `pypolychord.run_polychord` into the `settings`
    argument. See the PolyChord documentation for what all of those mean.

    To see what the keyword arguments are for, see the docstring of PyPolyChordSettings
    """

    default_kwargs = dict(use_polychord_defaults=False, nlive=None, num_repeats=None,
                          nprior=-1, do_clustering=True, feedback=1, precision_criterion=0.001,
                          logzero=-1e30, max_ndead=-1, boost_posterior=0.0, posteriors=True,
                          equals=True, cluster_posteriors=True, write_resume=True,
                          write_paramnames=False, read_resume=True, write_stats=True,
                          write_live=True, write_dead=True, write_prior=True,
                          compression_factor=np.exp(-1), base_dir='outdir',
                          file_root='polychord', seed=-1, grade_dims=None, grade_frac=None, nlives={})

    def run_sampler(self):
        import pypolychord
        from pypolychord.settings import PolyChordSettings
        if self.kwargs['use_polychord_defaults']:
            settings = PolyChordSettings(nDims=self.ndim, nDerived=self.ndim, base_dir=self.outdir,
                                         file_root=self.label)
        else:
            self._setup_dynamic_defaults()
            pc_kwargs = self.kwargs.copy()
            pc_kwargs['base_dir'] = self.outdir
            pc_kwargs['file_root'] = self.label
            pc_kwargs.pop('use_polychord_defaults')
            settings = PolyChordSettings(nDims=self.ndim, nDerived=self.ndim, **pc_kwargs)
        self._verify_kwargs_against_default_kwargs()

        pypolychord.run_polychord(loglikelihood=self.log_likelihood, nDims=self.ndim,
                                  nDerived=self.ndim, settings=settings, prior=self.prior_transform)

        self.result.log_evidence, self.result.log_evidence_err = self._read_out_stats_file()
        self.result.samples = self._read_sample_file()
        return self.result

    def _setup_dynamic_defaults(self):
        """ Sets up some interdependent default argument if none are given by the user """
        if not self.kwargs['grade_dims']:
            self.kwargs['grade_dims'] = [self.ndim]
        if not self.kwargs['grade_frac']:
            self.kwargs['grade_frac'] = [1.0] * len(self.kwargs['grade_dims'])
        if not self.kwargs['nlive']:
            self.kwargs['nlive'] = self.ndim * 25
        if not self.kwargs['num_repeats']:
            self.kwargs['num_repeats'] = self.ndim * 5

    def _translate_kwargs(self, kwargs):
        if 'nlive' not in kwargs:
            for equiv in self.npoints_equiv_kwargs:
                if equiv in kwargs:
                    kwargs['nlive'] = kwargs.pop(equiv)

    def log_likelihood(self, theta):
        """ Overrides the log_likelihood so that PolyChord understands it """
        return super(PyPolyChord, self).log_likelihood(theta), theta

    def _read_out_stats_file(self):
        statsfile = self.outdir + '/' + self.label + '.stats'
        with open(statsfile) as f:
            for line in f:
                if line.startswith('log(Z)'):
                    line = line.replace('log(Z)', '')
                    line = line.replace('=', '')
                    line = line.replace(' ', '')
                    print(line)
                    z = line.split('+/-')
                    log_z = float(z[0])
                    log_z_err = float(z[1])
                    return log_z, log_z_err

    def _read_sample_file(self):
        sample_file = self.outdir + '/' + self.label + '_equal_weights.txt'
        samples = np.loadtxt(sample_file)
        return samples[:, -self.ndim:]  # extract last ndim columns
