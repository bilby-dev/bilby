import os
from distutils.version import LooseVersion
from collections import OrderedDict

import numpy as np
import deepdish
import pandas as pd
import corner
import matplotlib
import matplotlib.pyplot as plt

from . import utils
from .utils import logger, infer_parameters_from_function
from .prior import PriorSet, DeltaFunction


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
    result: bilby.core.result.Result

    Raises
    -------
    ValueError: If no filename is given and either outdir or label is None
                If no bilby.core.result.Result is found in the path

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

        # Set some defaults
        self.outdir = '.'
        self.label = 'no_name'

        dict.__init__(self)
        if type(dictionary) is dict:
            for key in dictionary:
                val = self._standardise_a_string(dictionary[key])
                setattr(self, key, val)

        if getattr(self, 'priors', None) is not None:
            self.priors = PriorSet(self.priors)

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

    def save_to_file(self, overwrite=False):
        """
        Writes the Result to a deepdish h5 file

        Parameters
        ----------
        overwrite: bool, optional
            Whether or not to overwrite an existing result file.
            default=False
        """
        file_name = result_file_name(self.outdir, self.label)
        utils.check_directory_exists_and_if_not_mkdir(self.outdir)
        if os.path.isfile(file_name):
            if overwrite:
                logger.debug('Removing existing file {}'.format(file_name))
                os.remove(file_name)
            else:
                logger.debug(
                    'Renaming existing file {} to {}.old'.format(file_name,
                                                                 file_name))
                os.rename(file_name, file_name + '.old')

        logger.debug("Saving result to {}".format(file_name))

        # Convert the prior to a string representation for saving on disk
        dictionary = dict(self)
        if dictionary.get('priors', False):
            dictionary['priors'] = {key: str(self.priors[key]) for key in self.priors}

        # Convert callable kwargs to strings to avoid pickling issues
        if hasattr(self, 'kwargs'):
            for key in self.kwargs:
                if hasattr(self.kwargs[key], '__call__'):
                    self.kwargs[key] = str(self.kwargs[key])

        try:
            deepdish.io.save(file_name, dictionary)
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
                latex_labels.append(self.parameter_labels_with_unit[idx])
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

    def get_one_dimensional_median_and_error_bar(self, key, fmt='.2f',
                                                 quantiles=[0.16, 0.84]):
        """ Calculate the median and error bar for a given key

        Parameters
        ----------
        key: str
            The parameter key for which to calculate the median and error bar
        fmt: str, ('.2f')
            A format string
        quantiles: list
            A length-2 list of the lower and upper-quantiles to calculate
            the errors bars for.

        Returns
        -------
        string: str
            A string of latex-formatted text of the mean and 1-sigma quantiles

        """
        if len(quantiles) != 2:
            raise ValueError("quantiles must be of length 2")

        quants_to_compute = np.array([quantiles[0], 0.5, quantiles[1]])
        quants = np.percentile(self.posterior[key], quants_to_compute * 100)
        median = quants[1]
        upper = quants[2] - median
        lower = median - quants[0]

        fmt = "{{0:{0}}}".format(fmt).format
        string = r"${{{0}}}_{{-{1}}}^{{+{2}}}$"
        return string.format(fmt(median), fmt(lower), fmt(upper))

    def plot_corner(self, parameters=None, priors=None, titles=True, save=True,
                    filename=None, dpi=300, **kwargs):
        """ Plot a corner-plot

        Parameters
        ----------
        parameters: (list, dict), optional
            If given, either a list of the parameter names to include, or a
            dictionary of parameter names and their "true" values to plot.
        priors: {bool (False), bilby.core.prior.PriorSet}
            If true, add the stored prior probability density functions to the
            one-dimensional marginal distributions. If instead a PriorSet
            is provided, this will be plotted.
        titles: bool
            If true, add 1D titles of the median and (by default 1-sigma)
            error bars. To change the error bars, pass in the quantiles kwarg.
            See method `get_one_dimensional_median_and_error_bar` for further
            details). If `quantiles=None` is passed in, no title is added.
        save: bool, optional
            If true, save the image using the given label and outdir
        filename: str, optional
            If given, overwrite the default filename
        dpi: int, optional
            Dots per inch resolution of the plot
        **kwargs:
            Other keyword arguments are passed to `corner.corner`. We set some
            defaults to improve the basic look and feel, but these can all be
            overridden.

        Notes
        -----
            The generation of the corner plot themselves is done by the corner
            python module, see https://corner.readthedocs.io for more
            information.

        Returns
        -------
        fig:
            A matplotlib figure instance

        """

        # If in testing mode, not corner plots are generated
        if utils.command_line_args.test:
            return

        # bilby default corner kwargs. Overwritten by anything passed to kwargs
        defaults_kwargs = dict(
            bins=50, smooth=0.9, label_kwargs=dict(fontsize=16),
            title_kwargs=dict(fontsize=16), color='#0072C1',
            truth_color='tab:orange', quantiles=[0.16, 0.84],
            levels=(1 - np.exp(-0.5), 1 - np.exp(-2), 1 - np.exp(-9 / 2.)),
            plot_density=False, plot_datapoints=True, fill_contours=True,
            max_n_ticks=3)

        if LooseVersion(matplotlib.__version__) < "2.1":
            defaults_kwargs['hist_kwargs'] = dict(normed=True)
        else:
            defaults_kwargs['hist_kwargs'] = dict(density=True)

        if 'lionize' in kwargs and kwargs['lionize'] is True:
            defaults_kwargs['truth_color'] = 'tab:blue'
            defaults_kwargs['color'] = '#FF8C00'

        defaults_kwargs.update(kwargs)
        kwargs = defaults_kwargs

        # Handle if truths was passed in
        if 'truth' in kwargs:
            kwargs['truths'] = kwargs.pop('truth')
        if kwargs.get('truths'):
            truths = kwargs.get('truths')
            if isinstance(parameters, list) and isinstance(truths, list):
                if len(parameters) != len(truths):
                    raise ValueError(
                        "Length of parameters and truths don't match")
            elif isinstance(truths, dict) and parameters is None:
                parameters = kwargs.pop('truths')
            else:
                raise ValueError(
                    "Combination of parameters and truths not understood")

        # If injection parameters where stored, use these as parameter values
        # but do not overwrite input parameters (or truths)
        cond1 = getattr(self, 'injection_parameters', None) is not None
        cond2 = parameters is None
        if cond1 and cond2:
            parameters = {key: self.injection_parameters[key] for key in
                          self.search_parameter_keys}

        # If parameters is a dictionary, use the keys to determine which
        # parameters to plot and the values as truths.
        if isinstance(parameters, dict):
            plot_parameter_keys = list(parameters.keys())
            kwargs['truths'] = list(parameters.values())
        elif parameters is None:
            plot_parameter_keys = self.search_parameter_keys
        else:
            plot_parameter_keys = list(parameters)

        # Get latex formatted strings for the plot labels
        kwargs['labels'] = kwargs.get(
            'labels', self.get_latex_labels_from_parameter_keys(
                plot_parameter_keys))

        # Create the data array to plot and pass everything to corner
        xs = self.posterior[plot_parameter_keys].values
        fig = corner.corner(xs, **kwargs)
        axes = fig.get_axes()

        #  Add the titles
        if titles and kwargs.get('quantiles', None) is not None:
            for i, par in enumerate(plot_parameter_keys):
                ax = axes[i + i * len(plot_parameter_keys)]
                if ax.title.get_text() == '':
                    ax.set_title(self.get_one_dimensional_median_and_error_bar(
                        par, quantiles=kwargs['quantiles']),
                        **kwargs['title_kwargs'])

        #  Add priors to the 1D plots
        if priors is True:
            priors = getattr(self, 'priors', False)
        if isinstance(priors, dict):
            for i, par in enumerate(plot_parameter_keys):
                ax = axes[i + i * len(plot_parameter_keys)]
                theta = np.linspace(ax.get_xlim()[0], ax.get_xlim()[1], 300)
                ax.plot(theta, priors[par].prob(theta), color='C2')
        elif priors in [False, None]:
            pass
        else:
            raise ValueError('Input priors={} not understood'.format(priors))

        if save:
            if filename is None:
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

    def plot_with_data(self, model, x, y, ndraws=1000, npoints=1000,
                       xlabel=None, ylabel=None, data_label='data',
                       data_fmt='o', draws_label=None, filename=None,
                       maxl_label='max likelihood', dpi=300):
        """ Generate a figure showing the data and fits to the data

        Parameters
        ----------
        model: function
            A python function which when called as `model(x, **kwargs)` returns
            the model prediction (here `kwargs` is a dictionary of key-value
            pairs of the model parameters.
        x, y: np.ndarray
            The independent and dependent data to plot
        ndraws: int
            Number of draws from the posterior to plot
        npoints: int
            Number of points used to plot the smoothed fit to the data
        xlabel, ylabel: str
            Labels for the axes
        data_label, draws_label, maxl_label: str
            Label for the data, draws, and max likelihood legend
        data_fmt: str
            Matpltolib fmt code, defaults to `'-o'`
        dpi: int
            Passed to `plt.savefig`
        filename: str
            If given, the filename to use. Otherwise, the filename is generated
            from the outdir and label attributes.

        """

        # Determine model_posterior, the subset of the full posterior which
        # should be passed into the model
        model_keys = infer_parameters_from_function(model)
        model_posterior = self.posterior[model_keys]

        xsmooth = np.linspace(np.min(x), np.max(x), npoints)
        fig, ax = plt.subplots()
        logger.info('Plotting {} draws'.format(ndraws))
        for _ in range(ndraws):
            s = model_posterior.sample().to_dict('records')[0]
            ax.plot(xsmooth, model(xsmooth, **s), alpha=0.25, lw=0.1, color='r',
                    label=draws_label)
        if all(~np.isnan(self.posterior.log_likelihood)):
            logger.info('Plotting maximum likelihood')
            s = model_posterior.ix[self.posterior.log_likelihood.idxmax()]
            ax.plot(xsmooth, model(xsmooth, **s), lw=1, color='k',
                    label=maxl_label)

        ax.plot(x, y, data_fmt, markersize=2, label=data_label)

        if xlabel is not None:
            ax.set_xlabel(xlabel)
        if ylabel is not None:
            ax.set_ylabel(ylabel)

        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = OrderedDict(zip(labels, handles))
        plt.legend(by_label.values(), by_label.keys())
        ax.legend(numpoints=3)
        fig.tight_layout()
        if filename is None:
            filename = '{}/{}_plot_with_data'.format(self.outdir, self.label)
        fig.savefig(filename, dpi=dpi)

    def samples_to_posterior(self, likelihood=None, priors=None,
                             conversion_function=None):
        """
        Convert array of samples to posterior (a Pandas data frame).

        Parameters
        ----------
        likelihood: bilby.likelihood.GravitationalWaveTransient, optional
            GravitationalWaveTransient likelihood used for sampling.
        priors: dict, optional
            Dictionary of prior object, used to fill in delta function priors.
        conversion_function: function, optional
            Function which adds in extra parameters to the data frame,
            should take the data_frame, likelihood and prior as arguments.
        """
        if hasattr(self, 'posterior') is False:
            data_frame = pd.DataFrame(
                self.samples, columns=self.search_parameter_keys)
            for key in priors:
                if isinstance(priors[key], DeltaFunction):
                    data_frame[key] = priors[key].peak
                elif isinstance(priors[key], float):
                    data_frame[key] = priors[key]
            data_frame['log_likelihood'] = getattr(
                self, 'log_likelihood_evaluations', np.nan)
            # remove the array of samples
            del self.samples
        else:
            data_frame = self.posterior
        if conversion_function is not None:
            data_frame = conversion_function(data_frame, likelihood, priors)
        self.posterior = data_frame

    def calculate_prior_values(self, priors):
        """
        Evaluate prior probability for each parameter for each sample.

        Parameters
        ----------
        priors: dict, PriorSet
            Prior distributions
        """
        self.prior_values = pd.DataFrame()
        for key in priors:
            if key in self.posterior.keys():
                if isinstance(priors[key], DeltaFunction):
                    continue
                else:
                    self.prior_values[key]\
                        = priors[key].prob(self.posterior[key].values)

    def construct_cbc_derived_parameters(self):
        """ Construct widely used derived parameters of CBCs """
        self.posterior['mass_chirp'] = (
            (self.posterior.mass_1 * self.posterior.mass_2) ** 0.6 / (
                self.posterior.mass_1 + self.posterior.mass_2) ** 0.2)
        self.search_parameter_keys.append('mass_chirp')
        self.parameter_labels.append('$\mathcal{M}$')

        self.posterior['q'] = self.posterior.mass_2 / self.posterior.mass_1
        self.search_parameter_keys.append('q')
        self.parameter_labels.append('$q$')

        self.posterior['eta'] = (
            (self.posterior.mass_1 * self.posterior.mass_2) / (
                self.posterior.mass_1 + self.posterior.mass_2) ** 2)
        self.search_parameter_keys.append('eta')
        self.parameter_labels.append('$\eta$')

        self.posterior['chi_eff'] = (
            (self.posterior.a_1 * np.cos(self.posterior.tilt_1) +
                self.posterior.q * self.posterior.a_2 *
                np.cos(self.posterior.tilt_2)) / (1 + self.posterior.q))
        self.search_parameter_keys.append('chi_eff')
        self.parameter_labels.append('$\chi_{\mathrm eff}$')

        self.posterior['chi_p'] = (
            np.maximum(self.posterior.a_1 * np.sin(self.posterior.tilt_1),
                       (4 * self.posterior.q + 3) / (3 * self.posterior.q + 4) *
                       self.posterior.q * self.posterior.a_2 *
                       np.sin(self.posterior.tilt_2)))
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
        A list of `bilby.core.result.Result` objects containing the samples to
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
            template = ' $\mathrm{{ln}}(Z)={lnz:1.3g}$'
        else:
            template = ' $\mathrm{{ln}}(B)={lnbf:1.3g}$'
        labels = [template.format(lnz=result.log_evidence,
                                  lnbf=result.log_bayes_factor)
                  for ii, result in enumerate(results)]

    axes = fig.get_axes()
    ndim = int(np.sqrt(len(axes)))
    axes[ndim - 1].legend(lines, labels)

    if filename is None:
        filename = default_filename

    if save:
        fig.savefig(filename)
    return fig
