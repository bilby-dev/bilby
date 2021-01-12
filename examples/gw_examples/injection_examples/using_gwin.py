#!/usr/bin/env python
"""
An example of how to use bilby with `gwin` (https://github.com/gwastro/gwin) to
perform CBC parameter estimation.

To run this example, it is sufficient to use the pip-installable pycbc package,
but the source installation of gwin. You can install this by cloning the
repository (https://github.com/gwastro/gwin) and running

$ python setup.py install

A practical difference between gwin and bilby is that while fixed parameters
are specified via the prior in bilby, in gwin, these are fixed at instantiation
of the model. So, in the following, we only create priors for the parameters
to be searched over.

"""
import numpy as np
import bilby

import gwin
from pycbc import psd as pypsd
from pycbc.waveform.generator import (FDomainDetFrameGenerator,
                                      FDomainCBCGenerator)

label = 'using_gwin'

# Search priors
priors = dict()
priors['distance'] = bilby.core.prior.Uniform(500, 2000, 'distance')
priors['polarization'] = bilby.core.prior.Uniform(0, np.pi, 'theta_jn')

# Data variables
seglen = 4
sample_rate = 2048
N = seglen * sample_rate / 2 + 1
fmin = 30.

# Injected signal variables
injection_parameters = dict(mass1=38.6, mass2=29.3, spin1z=0, spin2z=0,
                            tc=0, ra=3.1, dec=1.37, polarization=2.76,
                            distance=1500)

# These lines figure out which parameters are to be searched over
variable_parameters = priors.keys()
fixed_parameters = injection_parameters.copy()
for key in priors:
    fixed_parameters.pop(key)

# These lines generate the `model` object - see
# https://gwin.readthedocs.io/en/latest/api/gwin.models.gaussian_noise.html
generator = FDomainDetFrameGenerator(
    FDomainCBCGenerator, 0.,
    variable_args=variable_parameters, detectors=['H1', 'L1'],
    delta_f=1. / seglen, f_lower=fmin,
    approximant='IMRPhenomPv2', **fixed_parameters)
signal = generator.generate(**injection_parameters)
psd = pypsd.aLIGOZeroDetHighPower(int(N), 1. / seglen, 20.)
psds = {'H1': psd, 'L1': psd}
model = gwin.models.GaussianNoise(
    variable_parameters, signal, generator, fmin, psds=psds)
model.update(**injection_parameters)


# This create a dummy class to convert the model into a bilby.likelihood object
class GWINLikelihood(bilby.core.likelihood.Likelihood):

    def __init__(self, model):
        """ A likelihood to wrap around GWIN model objects

        Parameters
        ----------
        model: gwin.model.GaussianNoise
            A gwin model

        """
        self.model = model
        self.parameters = {x: None for x in self.model.variable_params}

    def log_likelihood(self):
        self.model.update(**self.parameters)
        return self.model.loglikelihood


likelihood = GWINLikelihood(model)


# Now run the inference
result = bilby.run_sampler(
    likelihood=likelihood, priors=priors, sampler='dynesty', npoints=500,
    label=label)
result.plot_corner()
