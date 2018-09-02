import os
from distutils.version import LooseVersion
import numpy as np
import deepdish
import pandas as pd
import corner
import matplotlib
import matplotlib.pyplot as plt

from tupak.core import utils
from tupak.core.utils import logger


def result_file_name(outdir, label):
    """ Returns the standard filename used for a result file

    Parameters
    ----------
    outdir: str
        Name of the output directory
    label: str
        Naming scheme of the output file

    Returns
    -------
    str: File name of the output file
    """
    return '{}/{}_result.h5'.format(outdir, label)


def read_in_result(outdir=None, label=None, filename=None):
    """ Read in a saved .h5 data file

    Parameters
    ----------
    outdir, label: str
        If given, use the default naming convention for saved results file
    filename: str
        If given, try to load from this filename

    Returns
    -------
    result: tupak.core.result.Result

    Raises
    -------
    ValueError: If no filename is given and either outdir or label is None
                If no tupak.core.result.Result is found in the path

    """
    if filename is None:
        filename = result_file_name(outdir, label)
    elif (outdir is None or label is None) and filename is None:
        raise ValueError("No information given to load file")
    if os.path.isfile(filename):
        return Result(deepdish.io.load(filename))
    else:
        raise ValueError("No result found")


class Result(dict):
    def __init__(self, dictionary=None):
        """ A class to save the results of the sampling run.

        Parameters
        ----------
        dictionary: dict
            A dictionary containing values to be set in this instance
        """

        dict.__init__(self)
        if type(dictionary) is dict:
            for key in dictionary:
                val = self._standardise_a_string(dictionary[key])
                setattr(self, key, val)

    def __add__(self, other):
        matches = ['sampler', 'search_parameter_keys']
        for match in matches:
            # The 1 and 0 here ensure that if either doesn't have a match for
            # some reason, a error will be thrown.
            if getattr(other, match, 1) != getattr(self, match, 0):
                raise ValueError(
                    "Unable to add results generated with different {}".format(match))

        self.samples = np.concatenate([self.samples, other.samples])
        self.posterior = pd.concat([self.posterior, other.posterior])
        return self

    def __dir__(self):
        """ Adds tab completion in ipython

        See: http://ipython.org/ipython-doc/dev/config/integrating.html

        """
        methods = ['plot_corner', 'save_to_file', 'save_posterior_samples']
        return self.keys() + methods

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name)

    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

    def __repr__(self):
        """Print a summary """
        if hasattr(self, 'posterior'):
            if hasattr(self, 'log_noise_evidence'):
                return ("nsamples: {:d}\n"
                        "log_noise_evidence: {:6.3f}\n"
                        "log_evidence: {:6.3f} +/- {:6.3f}\n"
                        "log_bayes_factor: {:6.3f} +/- {:6.3f}\n"
                        .format(len(self.posterior), self.log_noise_evidence, self.log_evidence,
                                self.log_evidence_err, self.log_bayes_factor,
                                self.log_evidence_err))
            else:
                return ("nsamples: {:d}\n"
                        "log_evidence: {:6.3f} +/- {:6.3f}\n"
                        .format(len(self.posterior), self.log_evidence, self.log_evidence_err))
        else:
            return ''

    @staticmethod
    def _standardise_a_string(item):
        """ When reading in data, ensure all strings are decoded correctly

        Parameters
        ----------
        item: str

        Returns
        -------
        str: decoded string
        """
        if type(item) in [bytes]:
            return item.decode()
        else:
            return item

    @staticmethod
    def _standardise_strings(item):
        """

        Parameters
        ----------
        item: list
            List of strings to be decoded

        Returns
        -------
        list: list of decoded strings in item

        """
        if type(item) in [list]:
            item = [Result._standardise_a_string(i) for i in item]
        return item

    def save_to_file(self):
        """ Writes the Result to a deepdish h5 file """
        file_name = result_file_name(self.outdir, self.label)
        utils.check_directory_exists_and_if_not_mkdir(self.outdir)
        if os.path.isfile(file_name):
            logger.debug(
                'Renaming existing file {} to {}.old'.format(file_name,
                                                             file_name))
            os.rename(file_name, file_name + '.old')

        logger.debug("Saving result to {}".format(file_name))
        try:
            deepdish.io.save(file_name, dict(self))
        except Exception as e:
            logger.error("\n\n Saving the data has failed with the "
                         "following message:\n {} \n\n".format(e))

    def save_posterior_samples(self):
        """Saves posterior samples to a file"""
        filename = '{}/{}_posterior_samples.txt'.format(self.outdir, self.label)
        self.posterior.to_csv(filename, index=False, header=True)

    def get_latex_labels_from_parameter_keys(self, keys):
        """ Returns a list of latex_labels corresponding to the given keys

        Parameters
        ----------
        keys: list
            List of strings corresponding to the desired latex_labels

        Returns
        -------
        list: The desired latex_labels

        """
        latex_labels = []
        for k in keys:
            if k in self.search_parameter_keys:
                idx = self.search_parameter_keys.index(k)
                latex_labels.append(self.parameter_labels[idx])
            elif k in self.parameter_labels:
                latex_labels.append(k)
            else:
                raise ValueError('key {} not a parameter label or latex label'
                                 .format(k))
        return latex_labels

    @property
    def covariance_matrix(self):
        """ The covariance matrix of the samples the posterior """
        samples = self.posterior[self.search_parameter_keys].values
        return np.cov(samples.T)

    @property
    def posterior_volume(self):
        """ The posterior volume """
        if self.covariance_matrix.ndim == 0:
            return np.sqrt(self.covariance_matrix)
        else:
            return 1 / np.sqrt(np.abs(np.linalg.det(
                1 / self.covariance_matrix)))

    @staticmethod
    def prior_volume(priors):
        """ The prior volume, given a set of priors """
        return np.prod([priors[k].maximum - priors[k].minimum for k in priors])

    def occam_factor(self, priors):
        """ The Occam factor,

        See Chapter 28, `Mackay "Information Theory, Inference, and Learning
        Algorithms" <http://www.inference.org.uk/itprnn/book.html>`_ Cambridge
        University Press (2003).

        """
        return self.posterior_volume / self.prior_volume(priors)

    def plot_corner(self, parameters=None, save=True, dpi=300, **kwargs):
        """ Plot a corner-plot using corner

        See https://corner.readthedocs.io/en/latest/ for a detailed API.

        Parameters
        ----------
        parameters: list, optional
            If given, a list of the parameter names to include
        save: bool, optional
            If true, save the image using the given label and outdir
        dpi: int, optional
            Dots per inch resolution of the plot
        **kwargs:
            Other keyword arguments are passed to `corner.corner`. We set some
            defaults to improve the basic look and feel, but these can all be
            overridden.

        Returns
        -------
        fig:
            A matplotlib figure instance

        """
        if utils.command_line_args.test:
            return

        defaults_kwargs = dict(
            bins=50, smooth=0.9, label_kwargs=dict(fontsize=16),
            title_kwargs=dict(fontsize=16), color='#0072C1',
            truth_color='tab:orange', show_titles=True,
            quantiles=[0.16, 0.84],
            levels=(1 - np.exp(-0.5), 1 - np.exp(-2), 1 - np.exp(-9 / 2.)),
            plot_density=False, plot_datapoints=True, fill_contours=True,
            max_n_ticks=3)

        if LooseVersion(matplotlib.__version__) < "2.1":
            defaults_kwargs['hist_kwargs'] = dict(normed=True)
        else:
            defaults_kwargs['hist_kwargs'] = dict(density=True)

        defaults_kwargs.update(kwargs)
        kwargs = defaults_kwargs

        if 'truth' in kwargs:
            kwargs['truths'] = kwargs.pop('truth')

        if getattr(self, 'injection_parameters', None) is not None:
            injection_parameters = [self.injection_parameters.get(key, None)
                                    for key in self.search_parameter_keys]
            kwargs['truths'] = kwargs.get('truths', injection_parameters)

        if parameters is None:
            parameters = self.search_parameter_keys

        if 'lionize' in kwargs and kwargs['lionize'] is True:
            defaults_kwargs['truth_color'] = 'tab:blue'
            defaults_kwargs['color'] = '#FF8C00'

        xs = self.posterior[parameters].values
        kwargs['labels'] = kwargs.get(
            'labels', self.get_latex_labels_from_parameter_keys(
                parameters))

        if type(kwargs.get('truths')) == dict:
            truths = [kwargs['truths'][k] for k in parameters]
            kwargs['truths'] = truths

        fig = corner.corner(xs, **kwargs)

        if save:
            utils.check_directory_exists_and_if_not_mkdir(self.outdir)
            filename = '{}/{}_corner.png'.format(self.outdir, self.label)
            logger.debug('Saving corner plot to {}'.format(filename))
            fig.savefig(filename, dpi=dpi)

        return fig

    def plot_walkers(self, **kwargs):
        """ Method to plot the trace of the walkers in an ensemble MCMC plot """
        if hasattr(self, 'walkers') is False:
            logger.warning("Cannot plot_walkers as no walkers are saved")
            return

        if utils.command_line_args.test:
            return

        nwalkers, nsteps, ndim = self.walkers.shape
        idxs = np.arange(nsteps)
        fig, axes = plt.subplots(nrows=ndim, figsize=(6, 3 * ndim))
        walkers = self.walkers[:, :, :]
        for i, ax in enumerate(axes):
            ax.plot(idxs[:self.nburn + 1], walkers[:, :self.nburn + 1, i].T,
                    lw=0.1, color='r')
            ax.set_ylabel(self.parameter_labels[i])

        for i, ax in enumerate(axes):
            ax.plot(idxs[self.nburn:], walkers[:, self.nburn:, i].T, lw=0.1,
                    color='k')
            ax.set_ylabel(self.parameter_labels[i])

        fig.tight_layout()
        filename = '{}/{}_walkers.png'.format(self.outdir, self.label)
        logger.debug('Saving walkers plot to {}'.format('filename'))
        utils.check_directory_exists_and_if_not_mkdir(self.outdir)
        fig.savefig(filename)

    def samples_to_posterior(self, likelihood=None, priors=None,
                             conversion_function=None):
        """
        Convert array of samples to posterior (a Pandas data frame).

        Parameters
        ----------
        likelihood: tupak.likelihood.GravitationalWaveTransient, optional
            GravitationalWaveTransient likelihood used for sampling.
        priors: dict, optional
            Dictionary of prior object, used to fill in delta function priors.
        conversion_function: function, optional
            Function which adds in extra parameters to the data frame,
            should take the data_frame, likelihood and prior as arguments.
        """
        data_frame = pd.DataFrame(
            self.samples, columns=self.search_parameter_keys)
        data_frame['log_likelihood'] = getattr(self, 'log_likelihood_evaluations', np.nan)
        if conversion_function is not None:
            data_frame = conversion_function(data_frame, likelihood, priors)
        self.posterior = data_frame
        # We save the samples in the posterior and remove the array of samples
        del self.samples

    def construct_cbc_derived_parameters(self):
        """ Construct widely used derived parameters of CBCs """
        self.posterior['mass_chirp'] = (self.posterior.mass_1 * self.posterior.mass_2) ** 0.6 / (
                self.posterior.mass_1 + self.posterior.mass_2) ** 0.2
        self.search_parameter_keys.append('mass_chirp')
        self.parameter_labels.append('$\mathcal{M}$')

        self.posterior['q'] = self.posterior.mass_2 / self.posterior.mass_1
        self.search_parameter_keys.append('q')
        self.parameter_labels.append('$q$')

        self.posterior['eta'] = (self.posterior.mass_1 * self.posterior.mass_2) / (
                self.posterior.mass_1 + self.posterior.mass_2) ** 2
        self.search_parameter_keys.append('eta')
        self.parameter_labels.append('$\eta$')

        self.posterior['chi_eff'] = (self.posterior.a_1 * np.cos(self.posterior.tilt_1)
                                     + self.posterior.q * self.posterior.a_2 * np.cos(self.posterior.tilt_2)) / (
                                            1 + self.posterior.q)
        self.search_parameter_keys.append('chi_eff')
        self.parameter_labels.append('$\chi_{\mathrm eff}$')

        self.posterior['chi_p'] = np.maximum(self.posterior.a_1 * np.sin(self.posterior.tilt_1),
                                      (4 * self.posterior.q + 3) / (3 * self.posterior.q + 4) * self.posterior.q
                                      * self.posterior.a_2 * np.sin(self.posterior.tilt_2))
        self.search_parameter_keys.append('chi_p')
        self.parameter_labels.append('$\chi_{\mathrm p}$')

    def check_attribute_match_to_other_object(self, name, other_object):
        """ Check attribute name exists in other_object and is the same

        Parameters
        ----------
        name: str
            Name of the attribute in this instance
        other_object: object
            Other object with attributes to compare with

        Returns
        -------
        bool: True if attribute name matches with an attribute of other_object, False otherwise

        """
        A = getattr(self, name, False)
        B = getattr(other_object, name, False)
        logger.debug('Checking {} value: {}=={}'.format(name, A, B))
        if (A is not False) and (B is not False):
            typeA = type(A)
            typeB = type(B)
            if typeA == typeB:
                if typeA in [str, float, int, dict, list]:
                    try:
                        return A == B
                    except ValueError:
                        return False
                elif typeA in [np.ndarray]:
                    return np.all(A == B)
        return False


def plot_multiple(results, filename=None, labels=None, colours=None,
                  save=True, evidences=False, **kwargs):
    """ Generate a corner plot overlaying two sets of results

    Parameters
    ----------
    results: list
        A list of `tupak.core.result.Result` objects containing the samples to
        plot.
    filename: str
        File name to save the figure to. If None (default), a filename is
        constructed from the outdir of the first element of results and then
        the labels for all the result files.
    labels: list
        List of strings to use when generating a legend. If None (default), the
        `label` attribute of each result in `results` is used.
    colours: list
        The colours for each result. If None, default styles are applied.
    save: bool
        If true, save the figure
    kwargs: dict
        All other keyword arguments are passed to `result.plot_corner`.
        However, `show_titles` and `truths` are ignored since they would be
        ambiguous on such a plot.
    evidences: bool, optional
        Add the log-evidence calculations to the legend. If available, the
        Bayes factor will be used instead.

    Returns
    -------
    fig:
        A matplotlib figure instance

    """

    kwargs['show_titles'] = False
    kwargs['truths'] = None

    fig = results[0].plot_corner(save=False, **kwargs)
    default_filename = '{}/{}'.format(results[0].outdir, 'combined')
    lines = []
    default_labels = []
    for i, result in enumerate(results):
        if colours:
            c = colours[i]
        else:
            c = 'C{}'.format(i)
        hist_kwargs = kwargs.get('hist_kwargs', dict())
        hist_kwargs['color'] = c
        fig = result.plot_corner(fig=fig, save=False, color=c, **kwargs)
        default_filename += '_{}'.format(result.label)
        lines.append(matplotlib.lines.Line2D([0], [0], color=c))
        default_labels.append(result.label)

    # Rescale the axes
    for i, ax in enumerate(fig.axes):
        ax.autoscale()
    plt.draw()

    if labels is None:
        labels = default_labels

    if evidences:
        if np.isnan(results[0].log_bayes_factor):
            template = ' $\mathrm{{ln}}(Z)={:1.3g}$'
        else:
            template = ' $\mathrm{{ln}}(B)={:1.3g}$'
        for i, label in enumerate(labels):
            labels[i] = label + template.format(results[i].log_bayes_factor)

    axes = fig.get_axes()
    ndim = int(np.sqrt(len(axes)))
    axes[ndim - 1].legend(lines, labels)

    if filename is None:
        filename = default_filename

    if save:
        fig.savefig(filename)
    return fig
