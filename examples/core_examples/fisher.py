#!/usr/bin/env python
"""
This example, building on the Gaussian example, demonstrates the use of the Fisher
tools within Bilby.

"""
import bilby
import numpy as np
from bilby.core.utils.random import rng, seed

# Sets seed of bilby's generator "rng" to "123" to ensure reproducibility
seed(123)

# A few simple setup steps
outdir = "outdir"
bilby.utils.check_directory_exists_and_if_not_mkdir(outdir)


# First, we define our "signal model", in this case a simple linear function
def model(time, m, c):
    return time * m + c


# Now we define the injection parameters which we make simulated data with
injection_parameters = dict(m=0.5, c=0.2)

# For this example, we'll use standard Gaussian noise

# These lines of code generate the fake data. Note the ** just unpacks the
# contents of the injection_parameters when calling the model function.
sampling_frequency = 10
time_duration = 10
time = np.arange(0, time_duration, 1 / sampling_frequency)
N = len(time)
sigma = rng.normal(1, 0.01, N)
data = model(time, **injection_parameters) + rng.normal(0, sigma, N)

# Now lets instantiate a version of our GaussianLikelihood, giving it
# the time, data and signal model
likelihood = bilby.likelihood.GaussianLikelihood(time, data, model, sigma)

# From hereon, the syntax is exactly equivalent to other bilby examples
# We make a prior
priors = dict()
priors["m"] = bilby.core.prior.Uniform(0, 5, "m")
priors["c"] = bilby.core.prior.Uniform(-2, 2, "c")

# Run the "fisher" sampler
result_fisher = bilby.run_sampler(
    likelihood=likelihood,
    priors=priors,
    sampler="fisher",
    injection_parameters=injection_parameters,
    outdir=outdir,
    label="example_fisher",
    nsamples=5000,
)


# Run dynesty as a comparison
result_dynesty = bilby.run_sampler(
    likelihood=likelihood,
    priors=priors,
    sampler="dynesty",
    nlive=500,
    injection_parameters=injection_parameters,
    outdir=outdir,
    label="example_dynesty",
)

# Plot a corner plot: all outputs are stored in outdir
bilby.result.plot_multiple(
    [result_fisher, result_dynesty],
    filename=f"{outdir}/comparison_fisher_dynesty.png",
    labels=[
        f"Fisher ({result_fisher.meta_data['run_statistics']['sampling_time_s']:0.2f} s)",
        f"Dynesty ({result_dynesty.meta_data['run_statistics']['sampling_time_s']:0.2f} s)",
    ],
    parameters=injection_parameters,
    truth_color="C3",
)

# Note that the `fisher` tools can also be accessed directly
fisher = bilby.core.fisher.FisherMatrixPosteriorEstimator(likelihood, priors)
samples = fisher.sample_dataframe(
    "maxL", 1000
)  # Draw a set of samples as a pandas dataframe
