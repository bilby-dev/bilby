import logging
import os
import deepdish
from chainconsumer import ChainConsumer


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
        deepdish.io.save(file_name, self)

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
            kwargs['filename'] = '{}/{}_corner.png'.format(self.outdir, self.label)
            logging.info('Saving corner plot to {}'.format(kwargs['filename']))
        c = ChainConsumer()
        c.add_chain(self.samples, parameters=self.parameter_labels)
        fig = c.plotter.plot(**kwargs)
        return fig


