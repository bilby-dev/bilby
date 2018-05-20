import logging
import os
import numpy as np
import deepdish
import pandas as pd
import tupak

try:
    from chainconsumer import ChainConsumer
except ImportError:
    def ChainConsumer():
        raise ImportError(
            "You do not have the optional module chainconsumer installed")


def result_file_name(outdir, label):
    """ Returns the standard filename used for a result file """
    return '{}/{}_result.h5'.format(outdir, label)


def read_in_result(outdir=None, label=None, filename=None):
    """ Read in a saved .h5 data file

    Parameters
    ----------
    outdir, label: str
        If given, use the default naming convention for saved results file
    filename: str
        If given, try to load from this filename

    Returns:
    result: tupak.result.Result instance

    """
    if filename is None:
        filename = result_file_name(outdir, label)
    if os.path.isfile(filename):
        return Result(deepdish.io.load(filename))
    else:
        raise ValueError("No information given to load file")


class Result(dict):
    def __init__(self, dictionary=None):
        if type(dictionary) is dict:
            for key in dictionary:
                setattr(self, key, dictionary[key])

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name)

    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

    def __repr__(self):
        """Print a summary """
        if hasattr(self, 'samples'):
            return ("nsamples: {:d}\n"
                    "noise_logz: {:6.3f}\n"
                    "logz: {:6.3f} +/- {:6.3f}\n"
                    "log_bayes_factor: {:6.3f} +/- {:6.3f}\n"
                    .format(len(self.samples), self.noise_logz, self.logz,
                            self.logzerr, self.log_bayes_factor, self.logzerr))
        else:
            return ''

    def get_result_dictionary(self):
        return dict(self)

    def save_to_file(self, outdir, label):
        """ Writes the Result to a deepdish h5 file """
        file_name = result_file_name(outdir, label)
        if os.path.isdir(outdir) is False:
            os.makedirs(outdir)
        if os.path.isfile(file_name):
            logging.info(
                'Renaming existing file {} to {}.old'.format(file_name,
                                                             file_name))
            os.rename(file_name, file_name + '.old')

        logging.info("Saving result to {}".format(file_name))
        try:
            deepdish.io.save(file_name, self.get_result_dictionary())
        except Exception as e:
            logging.error(
                "\n\n Saving the data has failed with the following message:\n {} \n\n"
                .format(e))

    def get_latex_labels_from_parameter_keys(self, keys):
        return_list = []
        for k in keys:
            if k in self.search_parameter_keys:
                idx = self.search_parameter_keys.index(k)
                return_list.append(self.parameter_labels[idx])
            elif k in self.parameter_labels:
                return_list.append(k)
            else:
                raise ValueError('key {} not a parameter label or latex label'
                                 .format(k))
        return return_list

    def plot_corner(self, save=True, **kwargs):
        """ Plot a corner-plot using chain-consumer

        Parameters
        ----------
        save: bool
            If true, save the image using the given label and outdir

        Returns
        -------
        fig:
            A matplotlib figure instance
        """

        # Set some defaults (unless already set)
        kwargs['figsize'] = kwargs.get('figsize', 'GROW')
        if save:
            filename = '{}/{}_corner.png'.format(self.outdir, self.label)
            kwargs['filename'] = kwargs.get('filename', filename)
            logging.info('Saving corner plot to {}'.format(kwargs['filename']))
        if getattr(self, 'injection_parameters', None) is not None:
            # If no truth argument given, set these to the injection params
            injection_parameters = [self.injection_parameters[key]
                                    for key in self.search_parameter_keys]
            kwargs['truth'] = kwargs.get('truth', injection_parameters)

        if type(kwargs.get('truth')) == dict:
            old_keys = kwargs['truth'].keys()
            new_keys = self.get_latex_labels_from_parameter_keys(old_keys)
            for old, new in zip(old_keys, new_keys):
                kwargs['truth'][new] = kwargs['truth'].pop(old)
        if 'parameters' in kwargs:
            kwargs['parameters'] = self.get_latex_labels_from_parameter_keys(
                kwargs['parameters'])

        # Check all parameter_labels are a valid string
        for i, label in enumerate(self.parameter_labels):
            if label is None:
                self.parameter_labels[i] = 'Unknown'
        c = ChainConsumer()
        c.add_chain(self.samples, parameters=self.parameter_labels,
                    name=self.label)
        fig = c.plotter.plot(**kwargs)
        return fig

    def plot_walks(self, save=True, **kwargs):
        """ Plot the chain walks using chain-consumer

        Parameters
        ----------
        save: bool
            If true, save the image using the given label and outdir

        Returns
        -------
        fig:
            A matplotlib figure instance
        """

        # Set some defaults (unless already set)
        if save:
            kwargs['filename'] = '{}/{}_walks.png'.format(self.outdir, self.label)
            logging.info('Saving walker plot to {}'.format(kwargs['filename']))
        if self.injection_parameters is not None:
            kwargs['truth'] = [self.injection_parameters[key] for key in self.search_parameter_keys]
        c = ChainConsumer()
        c.add_chain(self.samples, parameters=self.parameter_labels)
        fig = c.plotter.plot_walks(**kwargs)
        return fig

    def plot_distributions(self, save=True, **kwargs):
        """ Plot the chain walks using chain-consumer

        Parameters
        ----------
        save: bool
            If true, save the image using the given label and outdir

        Returns
        -------
        fig:
            A matplotlib figure instance
        """

        # Set some defaults (unless already set)
        if save:
            kwargs['filename'] = '{}/{}_distributions.png'.format(self.outdir, self.label)
            logging.info('Saving distributions plot to {}'.format(kwargs['filename']))
        if self.injection_parameters is not None:
            kwargs['truth'] = [self.injection_parameters[key] for key in self.search_parameter_keys]
        c = ChainConsumer()
        c.add_chain(self.samples, parameters=self.parameter_labels)
        fig = c.plotter.plot_distributions(**kwargs)
        return fig

    def write_prior_to_file(self, outdir):
        """
        Write the prior distribution to file.

        :return:
        """
        outfile = outdir + '.prior'
        with open(outfile, "w") as prior_file:
            for key in self.prior:
                prior_file.write(self.prior[key])

    def samples_to_data_frame(self, likelihood=None, priors=None, conversion_function=None):
        """
        Convert array of samples to data frame.

        Parameters
        ----------
        likelihood: tupak.likelihood.GravitationalWaveTransient
            GravitationalWaveTransient used for sampling.
        priors: dict
            Dictionary of prior object, used to fill in delta function priors.
        conversion_function: function
            Function which adds in extra parameters to the data frame,
            should take the data_frame, likelihood and prior as arguments.
        """
        data_frame = pd.DataFrame(self.samples, columns=self.search_parameter_keys)
        if conversion_function is not None:
            conversion_function(data_frame, likelihood, priors)
        self.posterior = data_frame

    def construct_cbc_derived_parameters(self):
        """
        Construct widely used derived parameters of CBCs

        :return:
        """
        self.posterior['mass_chirp'] = (self.posterior.mass_1 * self.posterior.mass_2) ** 0.6 / (
                self.posterior.mass_1 + self.posterior.mass_2) ** 0.2
        self.posterior['q'] = self.posterior.mass_2 / self.posterior.mass_1
        self.posterior['eta'] = (self.posterior.mass_1 * self.posterior.mass_2) / (
                self.posterior.mass_1 + self.posterior.mass_2) ** 2

        self.posterior['chi_eff'] = (self.posterior.a_1 * np.cos(self.posterior.tilt_1)
                                     + self.posterior.q * self.posterior.a_2 * np.cos(self.posterior.tilt_2)) / (
                                                1 + self.posterior.q)
        self.posterior['chi_p'] = max(self.posterior.a_1 * np.sin(self.posterior.tilt_1),
                                      (4 * self.posterior.q + 3) / (3 * self.posterior.q + 4) * self.posterior.q
                                      * self.posterior.a_2 * np.sin(self.posterior.tilt_2))

    def check_attribute_match_to_other_object(self, name, other_object):
        """ Check attribute name exists in other_object and is the same """
        A = getattr(self, name, False)
        B = getattr(other_object, name, False)
        logging.debug('Checking {} value: {}=={}'.format(name, A, B))
        if (A is not False) and (B is not False):
            typeA = type(A)
            typeB = type(B)
            if typeA == typeB:
                if typeA in [str, float, int, dict, list]:
                    return A == B
                elif typeA in [np.ndarray]:
                    return np.all(A == B)
        return False

