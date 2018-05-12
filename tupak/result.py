import logging
import os
import numpy as np
import deepdish
from chainconsumer import ChainConsumer
import pandas as pd
import tupak


class Result(dict):

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name)

    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

    def __repr__(self):
        """Print a summary """
        return ("nsamples: {:d}\n"
                "noise_logz: {:6.3f}\n"
                "logz: {:6.3f} +/- {:6.3f}\n"
                "log_bayes_factor: {:6.3f} +/- {:6.3f}\n"
                .format(len(self.samples), self.noise_logz, self.logz, self.logzerr, self.log_bayes_factor,
                        self.logzerr))

    def save_to_file(self, outdir, label):
        file_name = '{}/{}_result.h5'.format(outdir, label)
        if os.path.isdir(outdir) is False:
            os.makedirs(outdir)
        if os.path.isfile(file_name):
            logging.info(
                'Renaming existing file {} to {}.old'.format(file_name,
                                                             file_name))
            os.rename(file_name, file_name + '.old')

        logging.info("Saving result to {}".format(file_name))
        try:
            deepdish.io.save(file_name, self)
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
        if self.injection_parameters is not None:
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

    def samples_to_data_frame(self, waveform_generator=None, interferometers=None, priors=None):
        """
        Convert array of samples to data frame.

        Parameters
        ----------
        waveform_generator: tupak.waveform_generator.WaveformGenerator, optional
            If the waveform generator and interferometers are provided, the SNRs will be recorded.
        interferometers: tupak.detector.Interferometer
            If the waveform generator and interferometers are provided, the SNRs will be recorded.
        priors: dict
            Dictionary of prior object, used to fill in delta function priors.
        """
        data_frame = pd.DataFrame(self.samples, columns=self.search_parameter_keys)
        tupak.conversion.generate_all_bbh_parameters(data_frame, waveform_generator, interferometers, priors)
        self.posterior = data_frame
        for key in self.fixed_parameter_keys:
            self.posterior[key] = self.prior[key].sample(len(self.posterior))

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
