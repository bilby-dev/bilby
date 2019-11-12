from ..core.utils import infer_args_from_function_except_n_args


class Model(object):
    """
    Population model

    This should take population parameters and return the probability.
    """

    def __init__(self, model_functions=None):
        """
        Parameters
        ----------
        model_functions: list
            List of functions to compute.
        """
        self.models = model_functions

        self.parameters = dict()

    def prob(self, data, **kwargs):
        probability = 1.0
        for ii, function in enumerate(self.models):
            probability *= function(data, **self._get_function_parameters(function))
        return probability

    def _get_function_parameters(self, func):
        """If the function is a class method we need to remove more arguments"""
        param_keys = infer_args_from_function_except_n_args(func, n=0)
        ignore = ['dataset', 'self', 'cls']
        for key in ignore:
            if key in param_keys:
                del param_keys[param_keys.index(key)]
        parameters = {key: self.parameters[key] for key in param_keys}
        return parameters
