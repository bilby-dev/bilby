from ..core.utils import infer_args_from_function_except_n_args


class Model:
    r"""
    Population model that combines a set of factorizable models.

    This should take population parameters and return the probability.

    .. math::

        p(\theta | \Lambda) = \prod_{i} p_{i}(\theta | \Lambda)
    """

    def __init__(self, model_functions=None, cache=True):
        """
        Parameters
        ==========
        model_functions: list
            List of callables to compute the probability.
            If this includes classes, the :code:`__call__`: method
            should return the probability.
            The requires variables are chosen at run time based on either
            inspection or querying a :code:`variable_names` attribute.
        cache: bool
            Whether to cache the value returned by the model functions,
            default=:code:`True`. The caching only looks at the parameters
            not the data, so should be used with caution. The caching also
            breaks :code:`jax` JIT compilation.
        """
        self.models = model_functions
        self.cache = cache
        self._cached_parameters = {model: None for model in self.models}
        self._cached_probability = {model: None for model in self.models}

        self.parameters = dict()

    def prob(self, data, **kwargs):
        """
        Compute the total population probability for the provided data given
        the keyword arguments.

        Parameters
        ==========
        data: dict
            Dictionary containing the points at which to evaluate the
            population model.
        kwargs: dict
            The population parameters. These cannot include any of
            :code:`["dataset", "data", "self", "cls"]` unless the
            :code:`variable_names` attribute is available for the relevant
            model.
        """
        probability = 1.0
        for ii, function in enumerate(self.models):
            function_parameters = self._get_function_parameters(function)
            if (
                self.cache
                and self._cached_parameters[function] == function_parameters
            ):
                new_probability = self._cached_probability[function]
            else:
                new_probability = function(
                    data, **self._get_function_parameters(function)
                )
                if self.cache:
                    self._cached_parameters[function] = function_parameters
                    self._cached_probability[function] = new_probability
            probability *= new_probability
        return probability

    def _get_function_parameters(self, func):
        """
        If the function is a class method we need to remove more arguments or
        have the variable names provided in the class.
        """
        if hasattr(func, "variable_names"):
            param_keys = func.variable_names
        else:
            param_keys = infer_args_from_function_except_n_args(func, n=0)
            ignore = ["dataset", "data", "self", "cls"]
            for key in ignore:
                if key in param_keys:
                    del param_keys[param_keys.index(key)]
        parameters = {key: self.parameters[key] for key in param_keys}
        return parameters
