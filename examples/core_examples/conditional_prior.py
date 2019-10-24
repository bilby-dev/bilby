import bilby
import numpy as np

# This tutorial demonstrates how we can sample a prior in the shape of a ball
# Note that this will not end up sampling uniformly in that space, constraint priors are more suitable for that.
# This implementation will draw a value for the x-coordinate from p(x), and given that draw a value for the
# y-coordinate from p(y|x), and given that draw a value for the z-coordinate from p(z|x,y).
# Only the x-coordinate will end up being uniform for this


class ZeroLikelihood(bilby.core.likelihood.Likelihood):
    """ Flat likelihood. This always returns 0.
    This way our posterior distribution is exactly the prior distribution."""
    def log_likelihood(self):
        return 0


def condition_func_y(reference_params, x):
    """ Condition function for our p(y|x) prior."""
    radius = 0.5 * (reference_params['maximum'] - reference_params['minimum'])
    y_max = np.sqrt(radius**2 - x**2)
    return dict(minimum=-y_max, maximum=y_max)


def condition_func_z(reference_params, x, y):
    """ Condition function for our p(z|x, y) prior."""
    radius = 0.5 * (reference_params['maximum'] - reference_params['minimum'])
    z_max = np.sqrt(radius**2 - x**2 - y**2)
    return dict(minimum=-z_max, maximum=z_max)


# Set up the conditional priors and the flat likelihood
priors = bilby.core.prior.ConditionalPriorDict()
priors['x'] = bilby.core.prior.Uniform(minimum=-1, maximum=1, latex_label="$x$")
priors['y'] = bilby.core.prior.ConditionalUniform(condition_func=condition_func_y, minimum=-1,
                                                  maximum=1, latex_label="$y$")
priors['z'] = bilby.core.prior.ConditionalUniform(condition_func=condition_func_z, minimum=-1,
                                                  maximum=1, latex_label="$z$")
likelihood = ZeroLikelihood(parameters=dict(x=0, y=0, z=0))

# Sample the prior distribution
res = bilby.run_sampler(likelihood=likelihood, priors=priors, sampler='dynesty', npoints=5000, walks=100,
                        label='conditional_prior', outdir='outdir', resume=False, clean=True)
res.plot_corner()
