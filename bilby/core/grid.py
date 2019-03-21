from __future__ import division

import numpy as np

from .prior import Prior, PriorDict
from .utils import logtrapzexp


class Grid(object):

    def __init__(self, likelihood, priors, grid_size=101):
        """

        Parameters
        ----------
        likelihood: bilby.likelihood.Likelihood
        priors: bilby.prior.PriorDict
        grid_size: int, list, dict
            Size of the grid, can be any of
            - int: all dimensions will have equal numbers of points
            - list: dimensions will use these points/this number of points in
            order of priors
            - dict: as for list
        """
        self.likelihood = likelihood
        self.priors = PriorDict(priors)
        self.n_dims = len(priors)
        self.parameter_names = list(self.priors.keys())

        self.sample_points = dict()
        self._get_sample_points(grid_size)
        # evaluate the prior on the grid points
        self._ln_prior = self.priors.ln_prob(
            {key: self.mesh_grid[i].flatten() for i, key in
             enumerate(self.parameter_names)}, axis=0).reshape(
            self.mesh_grid[0].shape)
        self._ln_likelihood = None

        # evaluate the likelihood on the grid points
        self._evaluate()

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
        ----------
        log_array: array_like
            A :class:`numpy.ndarray` of log likelihood/posterior values.
        parameters: list, str
            A list, or single string, of parameters to marginalize over. If None
            then all parameters will be marginalized over.
        not_parameters: list, str
            Instead of a list of parameters to marginalize over you can list
            the set of parameter to *not* marginalize over.

        Returns
        -------
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
        ----------
        log_array: array_like
            A :class:`numpy.ndarray` of log likelihood/posterior values.
        name: str
            The name of the parameter to marginalize over.
        non_marg_names: list
            A list of parameter names that have not been marginalized over.

        Returns
        -------
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
            out = np.apply_along_axis(
                logtrapzexp, axis, log_array, places[1] - places[0])
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

    def marginalize_ln_likelihood(self, parameter=None, not_parameter=None):
        """
        Marginalize the ln likelihood over either the specified parameter or
        all but the specified "not_parameter". If neither is specified the
        ln likelihood will be fully marginalized over.

        Parameters
        ----------
        parameter: str, optional
            Name of the parameter to marginalize over.
        not_parameter: str, optional
            Name of the parameter to not marginalize over.
        Returns
        -------
        array-like:
            The marginalized ln likelihood.
        """
        return self.marginalize(self.ln_likelihood, parameters=parameter,
                                not_parameters=not_parameter)

    def marginalize_ln_posterior(self, parameter=None, not_parameter=None):
        """
        Marginalize the ln posterior over either the specified parameter or all
        but the specified "not_parameter". If neither is specified the
        ln posterior will be fully marginalized over.

        Parameters
        ----------
        parameter: str, optional
            Name of the parameter to marginalize over.
        not_parameter: str, optional
            Name of the parameter to not marginalize over.
        Returns
        -------
        array-like:
            The marginalized ln posterior.
        """
        return self.marginalize(self.ln_posterior, parameters=parameter,
                                not_parameters=not_parameter)

    def marginalize_likelihood(self, parameter=None, not_parameter=None):
        """
        Marginalize the likelihood over either the specified parameter or all
        but the specified "not_parameter". If neither is specified the
        likelihood will be fully marginalized over.

        Parameters
        ----------
        parameter: str, optional
            Name of the parameter to marginalize over.
        not_parameter: str, optional
            Name of the parameter to not marginalize over.
        Returns
        -------
        array-like:
            The marginalized likelihood.
        """
        ln_like = self.marginalize(self.ln_likelihood, parameters=parameter,
                                   not_parameters=not_parameter)
        # NOTE: this outputs will not be properly normalised
        return np.exp(ln_like - np.max(ln_like))

    def marginalize_posterior(self, parameter=None, not_parameter=None):
        """
        Marginalize the posterior over either the specified parameter or all
        but the specified "not_parameter". If neither is specified the
        posterior will be fully marginalized over.

        Parameters
        ----------
        parameter: str, optional
            Name of the parameter to marginalize over.
        not_parameter: str, optional
            Name of the parameter to not marginalize over.
        Returns
        -------
        array-like:
            The marginalized posterior.
        """
        ln_post = self.marginalize(self.ln_posterior, parameters=parameter,
                                   not_parameters=not_parameter)
        # NOTE: this outputs will not be properly normalised
        return np.exp(ln_post - np.max(ln_post))

    def _evaluate(self):
        self._ln_likelihood = np.empty(self.mesh_grid[0].shape)
        self._evaluate_recursion(0)

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

        # set the mesh of points
        self.mesh_grid = np.meshgrid(
            *(self.sample_points[key] for key in self.parameter_names),
            indexing='ij')
