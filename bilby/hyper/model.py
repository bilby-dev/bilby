import inspect


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
        for function in self.models:
            for key in inspect.getargspec(function).args[1:]:
                self.parameters[key] = None

    def prob(self, data):
        for ii, function in enumerate(self.models):
            if ii == 0:
                probability = function(data, **self._get_function_parameters(function))
            else:
                probability *= function(data, **self._get_function_parameters(function))
        return probability

    def _get_function_parameters(self, function):
        parameters = {key: self.parameters[key] for key in inspect.getargspec(function).args[1:]}
        return parameters
