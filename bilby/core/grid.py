import json
import os

import numpy as np

from .prior import Prior, PriorDict
from .utils import (
    logtrapzexp, check_directory_exists_and_if_not_mkdir, logger,
    BilbyJsonEncoder, load_json, move_old_file
)
from .result import FileMovedError


def grid_file_name(outdir, label, gzip=False):
    """ Returns the standard filename used for a grid file

    Parameters
    ==========
    outdir: str
        Name of the output directory
    label: str
        Naming scheme of the output file
    gzip: bool, optional
        Set to True to append `.gz` to the extension for saving in gzipped format

    Returns
    =======
    str: File name of the output file
    """
    if gzip:
        return os.path.join(outdir, '{}_grid.json.gz'.format(label))
    else:
        return os.path.join(outdir, '{}_grid.json'.format(label))


class Grid(object):

    def __init__(self, likelihood=None, priors=None, grid_size=101,
                 save=False, label='no_label', outdir='.', gzip=False):
        """

        Parameters
        ==========
        likelihood: bilby.likelihood.Likelihood
        priors: bilby.prior.PriorDict
        grid_size: int, list, dict
            Size of the grid, can be any of
            - int: all dimensions will have equal numbers of points
            - list: dimensions will use these points/this number of points in
            order of priors
            - dict: as for list
        save: bool
            Set whether to save the results of the grid
        label: str
            The label for the filename to which the grid is saved
        outdir: str
            The output directory to which the grid will be saved
        gzip: bool
            Set whether to gzip the output grid file
        """

        if priors is None:
            priors = dict()
        self.likelihood = likelihood
        self.priors = PriorDict(priors)
        self.n_dims = len(priors)
        self.parameter_names = list(self.priors.keys())

        self.sample_points = dict()
        self._get_sample_points(grid_size)
        # evaluate the prior on the grid points
        if self.n_dims > 0:
            self._ln_prior = self.priors.ln_prob(
                {key: self.mesh_grid[i].flatten() for i, key in
                 enumerate(self.parameter_names)}, axis=0).reshape(
                self.mesh_grid[0].shape)
        self._ln_likelihood = None

        # evaluate the likelihood on the grid points
        if likelihood is not None and self.n_dims > 0:
            self._evaluate()

        self.save = save
        self.label = None
        self.outdir = None
        if self.save:
            if isinstance(label, str):
                self.label = label
            if isinstance(outdir, str):
                self.outdir = os.path.abspath(outdir)
            self.save_to_file(gzip=gzip)

    @property
    def ln_prior(self):
        return self._ln_prior

    @property
    def prior(self):
        return np.exp(self.ln_prior)

    @property
    def ln_likelihood(self):
        if self._ln_likelihood is None:
            self._evaluate()
        return self._ln_likelihood

    @property
    def ln_posterior(self):
        return self.ln_likelihood + self.ln_prior

    def marginalize(self, log_array, parameters=None, not_parameters=None):
        """
        Marginalize over a list of parameters.

        Parameters
        ==========
        log_array: array_like
            A :class:`numpy.ndarray` of log likelihood/posterior values.
        parameters: list, str
            A list, or single string, of parameters to marginalize over. If None
            then all parameters will be marginalized over.
        not_parameters: list, str
            Instead of a list of parameters to marginalize over you can list
            the set of parameter to *not* marginalize over.

        Returns
        =======
        out_array: array_like
            An array containing the marginalized log likelihood/posterior.
        """

        if parameters is None:
            params = list(self.parameter_names)

            if not_parameters is not None:
                if isinstance(not_parameters, str):
                    not_params = [not_parameters]
                elif isinstance(not_parameters, list):
                    not_params = not_parameters
                else:
                    raise TypeError("Parameters names must be a list or string")

                for name in list(params):
                    if name in not_params:
                        params.remove(name)
        elif isinstance(parameters, str):
            params = [parameters]
        elif isinstance(parameters, list):
            params = parameters
        else:
            raise TypeError("Parameters names must be a list or string")

        out_array = log_array.copy()
        names = list(self.parameter_names)

        for name in params:
            out_array = self._marginalize_single(out_array, name, names)

        return out_array

    def _marginalize_single(self, log_array, name, non_marg_names=None):
        """
        Marginalize the log likelihood/posterior over a single given parameter.

        Parameters
        ==========
        log_array: array_like
            A :class:`numpy.ndarray` of log likelihood/posterior values.
        name: str
            The name of the parameter to marginalize over.
        non_marg_names: list
            A list of parameter names that have not been marginalized over.

        Returns
        =======
        out: array_like
            An array containing the marginalized log likelihood/posterior.
        """

        if name not in self.parameter_names:
            raise ValueError("'{}' is not a recognised "
                             "parameter".format(name))

        if non_marg_names is None:
            non_marg_names = list(self.parameter_names)

        axis = non_marg_names.index(name)
        non_marg_names.remove(name)

        places = self.sample_points[name]

        if len(places) > 1:
            dx = np.diff(places)
            out = np.apply_along_axis(
                logtrapzexp, axis, log_array, dx
            )
        else:
            # no marginalisation required, just remove the singleton dimension
            z = log_array.shape
            q = np.arange(0, len(z)).astype(int) != axis
            out = np.reshape(log_array, tuple((np.array(list(z)))[q]))

        return out

    @property
    def ln_evidence(self):
        return self.marginalize(self.ln_posterior)

    @property
    def log_evidence(self):
        return self.ln_evidence

    @property
    def log_noise_evidence(self):
        return self.ln_noise_evidence

    def marginalize_ln_likelihood(self, parameters=None, not_parameters=None):
        """
        Marginalize the ln likelihood over either the specified parameter or
        all but the specified "not_parameter". If neither is specified the
        ln likelihood will be fully marginalized over.

        Parameters
        ==========
        parameters: str, list, optional
            Name of, or list of names of, the parameter(s) to marginalize over.
        not_parameters: str, optional
            Name of, or list of names of, the parameter(s) to not marginalize over.

        Returns
        =======
        array-like:
            The marginalized ln likelihood.
        """
        return self.marginalize(self.ln_likelihood, parameters=parameters,
                                not_parameters=not_parameters)

    def marginalize_ln_posterior(self, parameters=None, not_parameters=None):
        """
        Marginalize the ln posterior over either the specified parameter or all
        but the specified "not_parameter". If neither is specified the
        ln posterior will be fully marginalized over.

        Parameters
        ==========
        parameters: str, list, optional
            Name of, or list of names of, the parameter(s) to marginalize over.
        not_parameters: str, optional
            Name of, or list of names of, the parameter(s) to not marginalize over.

        Returns
        =======
        array-like:
            The marginalized ln posterior.
        """
        return self.marginalize(self.ln_posterior, parameters=parameters,
                                not_parameters=not_parameters)

    def marginalize_likelihood(self, parameters=None, not_parameters=None):
        """
        Marginalize the likelihood over either the specified parameter or all
        but the specified "not_parameter". If neither is specified the
        likelihood will be fully marginalized over.

        Parameters
        ==========
        parameters: str, list, optional
            Name of, or list of names of, the parameter(s) to marginalize over.
        not_parameters: str, optional
            Name of, or list of names of, the parameter(s) to not marginalize over.

        Returns
        =======
        array-like:
            The marginalized likelihood.
        """
        ln_like = self.marginalize(self.ln_likelihood, parameters=parameters,
                                   not_parameters=not_parameters)
        # NOTE: the output will not be properly normalised
        return np.exp(ln_like - np.max(ln_like))

    def marginalize_posterior(self, parameters=None, not_parameters=None):
        """
        Marginalize the posterior over either the specified parameter or all
        but the specified "not_parameters". If neither is specified the
        posterior will be fully marginalized over.

        Parameters
        ==========
        parameters: str, list, optional
            Name of, or list of names of, the parameter(s) to marginalize over.
        not_parameters: str, optional
            Name of, or list of names of, the parameter(s) to not marginalize over.

        Returns
        =======
        array-like:
            The marginalized posterior.
        """
        ln_post = self.marginalize(self.ln_posterior, parameters=parameters,
                                   not_parameters=not_parameters)
        # NOTE: the output will not be properly normalised
        return np.exp(ln_post - np.max(ln_post))

    def _evaluate(self):
        self._ln_likelihood = np.empty(self.mesh_grid[0].shape)
        self._evaluate_recursion(0)
        self.ln_noise_evidence = self.likelihood.noise_log_likelihood()

    def _evaluate_recursion(self, dimension):
        if dimension == self.n_dims:
            current_point = tuple([[int(np.where(
                self.likelihood.parameters[name] ==
                self.sample_points[name])[0])] for name in self.parameter_names])
            self._ln_likelihood[current_point] = self.likelihood.log_likelihood()
        else:
            name = self.parameter_names[dimension]
            for ii in range(self._ln_likelihood.shape[dimension]):
                self.likelihood.parameters[name] = self.sample_points[name][ii]
                self._evaluate_recursion(dimension + 1)

    def _get_sample_points(self, grid_size):
        for ii, key in enumerate(self.parameter_names):
            if isinstance(self.priors[key], Prior):
                if isinstance(grid_size, int):
                    self.sample_points[key] = self.priors[key].rescale(
                        np.linspace(0, 1, grid_size))
                elif isinstance(grid_size, list):
                    if isinstance(grid_size[ii], int):
                        self.sample_points[key] = self.priors[key].rescale(
                            np.linspace(0, 1, grid_size[ii]))
                    else:
                        self.sample_points[key] = grid_size[ii]
                elif isinstance(grid_size, dict):
                    if isinstance(grid_size[key], int):
                        self.sample_points[key] = self.priors[key].rescale(
                            np.linspace(0, 1, grid_size[key]))
                    else:
                        self.sample_points[key] = grid_size[key]
                else:
                    raise TypeError("Unrecognized 'grid_size' type")

        # set the mesh of points
        self.mesh_grid = np.meshgrid(
            *(self.sample_points[key] for key in self.parameter_names),
            indexing='ij')

    def _get_save_data_dictionary(self):
        # This list defines all the parameters saved in the grid object
        save_attrs = [
            'label', 'outdir', 'parameter_names', 'n_dims', 'priors',
            'sample_points', 'ln_likelihood', 'ln_evidence',
            'ln_noise_evidence']
        dictionary = dict()
        for attr in save_attrs:
            try:
                dictionary[attr] = getattr(self, attr)
            except ValueError as e:
                logger.debug("Unable to save {}, message: {}".format(attr, e))
                pass
        return dictionary

    def _safe_outdir_creation(self, outdir=None, caller_func=None):
        if outdir is None:
            outdir = self.outdir
        try:
            check_directory_exists_and_if_not_mkdir(outdir)
        except PermissionError:
            raise FileMovedError("Can not write in the out directory.\n"
                                 "Did you move the here file from another system?\n"
                                 "Try calling " + caller_func.__name__ + " with the 'outdir' "
                                 "keyword argument, e.g. " + caller_func.__name__ + "(outdir='.')")
        return outdir

    def save_to_file(self, filename=None, overwrite=False, outdir=None,
                     gzip=False):
        """
        Writes the Grid to a file.

        Parameters
        ==========
        filename: str, optional
            Filename to write to (overwrites the default)
        overwrite: bool, optional
            Whether or not to overwrite an existing result file.
            default=False
        outdir: str, optional
            Path to the outdir. Default is the one stored in the Grid object.
        gzip: bool, optional
            If true this will gzip the resulting file and add '.gz' to the file
            extension.
        """

        outdir = self._safe_outdir_creation(outdir, self.save_to_file)
        if filename is None:
            if self.label is None:
                raise ValueError("'label' for the output file name is not given")

            filename = grid_file_name(outdir, self.label, gzip)

        move_old_file(filename, overwrite)
        dictionary = self._get_save_data_dictionary()

        try:
            dictionary["priors"] = dictionary["priors"]._get_json_dict()
            if gzip or (os.path.splitext(filename)[-1] == '.gz'):
                import gzip
                # encode to a string
                json_str = json.dumps(dictionary, cls=BilbyJsonEncoder).encode('utf-8')
                with gzip.GzipFile(filename, 'w') as file:
                    file.write(json_str)
            else:
                with open(filename, 'w') as file:
                    json.dump(dictionary, file, indent=2, cls=BilbyJsonEncoder)
        except Exception as e:
            logger.error("\n\n Saving the data has failed with the "
                         "following message:\n {} \n\n".format(e))

    @classmethod
    def read(cls, filename=None, outdir=None, label=None, gzip=False):
        """ Read in a saved .json grid file

        Parameters
        ==========
        filename: str
            If given, try to load from this filename
        outdir, label: str
            If given, use the default naming convention for saved results file
        gzip: bool
            If given, whether the file is gzipped or not (only required if the
            file is gzipped, but does not have the standard '.gz' file
            extension)

        Returns
        =======
        grid: bilby.core.grid.Grid

        Raises
        ======
        ValueError: If no filename is given and either outdir or label is None
                    If no bilby.core.grid.Grid is found in the path

        """

        if filename is None:
            if (outdir is None) and (label is None):
                raise ValueError("No information given to load file")
            else:
                filename = grid_file_name(outdir, label, gzip)

        if os.path.isfile(filename):
            dictionary = load_json(filename, gzip)
            try:
                grid = cls(likelihood=None, priors=dictionary['priors'],
                           grid_size=dictionary['sample_points'],
                           label=dictionary['label'], outdir=dictionary['outdir'])

                # set the likelihood
                grid._ln_likelihood = dictionary['ln_likelihood']
                grid.ln_noise_evidence = dictionary['ln_noise_evidence']

                return grid
            except TypeError as e:
                raise IOError("Unable to load dictionary, error={}".format(e))
        else:
            raise IOError("No result '{}' found".format(filename))
