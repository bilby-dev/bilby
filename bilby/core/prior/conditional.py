import numpy as np

from .base import Prior, PriorException
from bilby.core.prior.interpolated import Interped
from bilby.core.prior.analytical import DeltaFunction, PowerLaw, Uniform, LogUniform, \
    SymmetricLogUniform, Cosine, Sine, Gaussian, TruncatedGaussian, HalfGaussian, \
    LogNormal, Exponential, StudentT, Beta, Logistic, Cauchy, Gamma, ChiSquared, FermiDirac
from bilby.core.utils import infer_args_from_method, infer_parameters_from_function


def conditional_prior_factory(prior_class):
    class ConditionalPrior(prior_class):
        def __init__(self, condition_func, name=None, latex_label=None, unit=None,
                     boundary=None, **reference_params):
            """

            Parameters
            ----------
            condition_func: func
                Functional form of the condition for this prior. The first function argument
                has to be a dictionary for the `reference_params` (see below). The following
                arguments are the required variables that are required before we can draw this
                prior.
                It needs to return a dictionary with the modified values for the
                `reference_params` that are being used in the next draw.
                For example if we have a Uniform prior for `x` depending on a different variable `y`
                `p(x|y)` with the boundaries linearly depending on y, then this
                could have the following form:

                ```
                def condition_func(reference_params, y):
                    return dict(minimum=reference_params['minimum'] + y, maximum=reference_params['maximum'] + y)
                ```
            name: str, optional
               See superclass
            latex_label: str, optional
                See superclass
            unit: str, optional
                See superclass
            boundary: str, optional
                See superclass
            reference_params:
                Initial values for attributes such as `minimum`, `maximum`.
                This differs on the `prior_class`, for example for the Gaussian
                prior this is `mu` and `sigma`.
            """
            if 'boundary' in infer_args_from_method(super(ConditionalPrior, self).__init__):
                super(ConditionalPrior, self).__init__(name=name, latex_label=latex_label,
                                                       unit=unit, boundary=boundary, **reference_params)
            else:
                super(ConditionalPrior, self).__init__(name=name, latex_label=latex_label,
                                                       unit=unit, **reference_params)

            self._required_variables = None
            self.condition_func = condition_func
            self._reference_params = reference_params
            self.__class__.__name__ = 'Conditional{}'.format(prior_class.__name__)
            self.__class__.__qualname__ = 'Conditional{}'.format(prior_class.__qualname__)

        def sample(self, size=None, **required_variables):
            """Draw a sample from the prior

            Parameters
            ----------
            size: int or tuple of ints, optional
                See superclass
            required_variables:
                Any required variables that this prior depends on

            Returns
            -------
            float: See superclass

            """
            self.least_recently_sampled = self.rescale(np.random.uniform(0, 1, size), **required_variables)
            return self.least_recently_sampled

        def rescale(self, val, **required_variables):
            """
            'Rescale' a sample from the unit line element to the prior.

            Parameters
            ----------
            val: Union[float, int, array_like]
                See superclass
            required_variables:
                Any required variables that this prior depends on


            """
            self.update_conditions(**required_variables)
            return super(ConditionalPrior, self).rescale(val)

        def prob(self, val, **required_variables):
            """Return the prior probability of val.

            Parameters
            ----------
            val: Union[float, int, array_like]
                See superclass
            required_variables:
                Any required variables that this prior depends on


            Returns
            -------
            float: Prior probability of val
            """
            self.update_conditions(**required_variables)
            return super(ConditionalPrior, self).prob(val)

        def ln_prob(self, val, **required_variables):
            self.update_conditions(**required_variables)
            return super(ConditionalPrior, self).ln_prob(val)

        def update_conditions(self, **required_variables):
            """
            This method updates the conditional parameters (depending on the parent class
            this could be e.g. `minimum`, `maximum`, `mu`, `sigma`, etc.) of this prior
            class depending on the required variables it depends on.

            If no variables are given, the most recently used conditional parameters are kept

            Parameters
            ----------
            required_variables:
                Any required variables that this prior depends on. If none are given,
                self.reference_params will be used.

            """
            if sorted(list(required_variables)) == sorted(self.required_variables):
                parameters = self.condition_func(self.reference_params.copy(), **required_variables)
                for key, value in parameters.items():
                    setattr(self, key, value)
            elif len(required_variables) == 0:
                return
            else:
                raise IllegalRequiredVariablesException("Expected kwargs for {}. Got kwargs for {} instead."
                                                        .format(self.required_variables,
                                                                list(required_variables.keys())))

        @property
        def reference_params(self):
            """
            Initial values for attributes such as `minimum`, `maximum`.
            This depends on the `prior_class`, for example for the Gaussian
            prior this is `mu` and `sigma`. This is read-only.
            """
            return self._reference_params

        @property
        def condition_func(self):
            return self._condition_func

        @condition_func.setter
        def condition_func(self, condition_func):
            if condition_func is None:
                self._condition_func = lambda reference_params: reference_params
            else:
                self._condition_func = condition_func
            self._required_variables = infer_parameters_from_function(self.condition_func)

        @property
        def required_variables(self):
            """ The required variables to pass into the condition function. """
            return self._required_variables

        def get_instantiation_dict(self):
            instantiation_dict = super(ConditionalPrior, self).get_instantiation_dict()
            for key, value in self.reference_params.items():
                instantiation_dict[key] = value
            return instantiation_dict

        def reset_to_reference_parameters(self):
            """
            Reset the object attributes to match the original reference parameters
            """
            for key, value in self.reference_params.items():
                setattr(self, key, value)

        def __repr__(self):
            """Overrides the special method __repr__.

            Returns a representation of this instance that resembles how it is instantiated.
            Works correctly for all child classes

            Returns
            -------
            str: A string representation of this instance

            """
            prior_name = self.__class__.__name__
            instantiation_dict = self.get_instantiation_dict()
            instantiation_dict["condition_func"] = ".".join([
                instantiation_dict["condition_func"].__module__,
                instantiation_dict["condition_func"].__name__
            ])
            args = ', '.join(['{}={}'.format(key, repr(instantiation_dict[key]))
                              for key in instantiation_dict])
            return "{}({})".format(prior_name, args)

    return ConditionalPrior


ConditionalBasePrior = conditional_prior_factory(Prior)  # Only for testing purposes
ConditionalUniform = conditional_prior_factory(Uniform)
ConditionalDeltaFunction = conditional_prior_factory(DeltaFunction)
ConditionalPowerLaw = conditional_prior_factory(PowerLaw)
ConditionalGaussian = conditional_prior_factory(Gaussian)
ConditionalLogUniform = conditional_prior_factory(LogUniform)
ConditionalSymmetricLogUniform = conditional_prior_factory(SymmetricLogUniform)
ConditionalCosine = conditional_prior_factory(Cosine)
ConditionalSine = conditional_prior_factory(Sine)
ConditionalTruncatedGaussian = conditional_prior_factory(TruncatedGaussian)
ConditionalHalfGaussian = conditional_prior_factory(HalfGaussian)
ConditionalLogNormal = conditional_prior_factory(LogNormal)
ConditionalExponential = conditional_prior_factory(Exponential)
ConditionalStudentT = conditional_prior_factory(StudentT)
ConditionalBeta = conditional_prior_factory(Beta)
ConditionalLogistic = conditional_prior_factory(Logistic)
ConditionalCauchy = conditional_prior_factory(Cauchy)
ConditionalGamma = conditional_prior_factory(Gamma)
ConditionalChiSquared = conditional_prior_factory(ChiSquared)
ConditionalFermiDirac = conditional_prior_factory(FermiDirac)
ConditionalInterped = conditional_prior_factory(Interped)


class DirichletElement(ConditionalBeta):
    """
    Single element in a dirichlet distribution

    The probability scales as

    $p(x_order) \propto (x_max - x_order)^(n_dimensions - order - 2)$

    for x_order < x_max, where x_max is the sum of x_i for i < order

    Examples
    --------
    n_dimensions = 1:
    p(x_0) \propto 1 ; 0 < x_0 < 1
    n_dimensions = 2:
    p(x_0) \propto (1 - x_0) ; 0 < x_0 < 1
    p(x_1) \propto 1 ; 0 < x_1 < 1

    Parameters
    ----------
    order: int
        Order of this element of the dirichlet distribution.
    n_dimensions: int
        Total number of elements of the dirichlet distribution
    label: str
        Label for the dirichlet distribution.
        This should be the same for all elements.
    """

    def __init__(self, order, n_dimensions, label):
        super(DirichletElement, self).__init__(
            minimum=0, maximum=1, alpha=1, beta=n_dimensions - order - 1,
            name=label + str(order),
            condition_func=self.dirichlet_condition
        )
        self.label = label
        self.n_dimensions = n_dimensions
        self.order = order
        self._required_variables = [
            label + str(ii) for ii in range(order)
        ]
        self.__class__.__name__ = 'Dirichlet'

    def dirichlet_condition(self, reference_parms, **kwargs):
        remaining = 1 - sum(
            [kwargs[self.label + str(ii)] for ii in range(self.order)]
        )
        return dict(minimum=reference_parms["minimum"], maximum=remaining)

    def __repr__(self):
        return Prior.__repr__(self)

    def get_instantiation_dict(self):
        return Prior.get_instantiation_dict(self)


class ConditionalPriorException(PriorException):
    """ General base class for all conditional prior exceptions """


class IllegalRequiredVariablesException(ConditionalPriorException):
    """ Exception class for exceptions relating to handling the required variables. """
