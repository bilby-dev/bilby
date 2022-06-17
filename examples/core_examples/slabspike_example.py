#!/usr/bin/env python
"""
An example of how to use slab-and-spike priors in bilby.
In this example we look at a simple example with the sum
of two Gaussian distributions, and we try to fit with
up to three Gaussians.

We will use the `PyMultiNest` sampler which is fast but can be unreliable when
significant correlations exist in the likelihood.

To install `PyMultiNest` call

$ conda install -c conda-forge pymultinest
"""

import bilby
import matplotlib.pyplot as plt
import numpy as np

outdir = "outdir"
label = "slabspike"
bilby.utils.check_directory_exists_and_if_not_mkdir(outdir)


# Here we define our model. We want to inject two Gaussians and recover with up to three.
def gaussian(xs, amplitude, mu, sigma):
    return (
        amplitude
        / np.sqrt(2 * np.pi * sigma**2)
        * np.exp(-0.5 * (xs - mu) ** 2 / sigma**2)
    )


def triple_gaussian(
    xs,
    amplitude_0,
    amplitude_1,
    amplitude_2,
    mu_0,
    mu_1,
    mu_2,
    sigma_0,
    sigma_1,
    sigma_2,
    **kwargs,
):
    return (
        gaussian(xs, amplitude_0, mu_0, sigma_0)
        + gaussian(xs, amplitude_1, mu_1, sigma_1)
        + gaussian(xs, amplitude_2, mu_2, sigma_2)
    )


# Let's create our data set. We create 200 points on a grid.

xs = np.linspace(-5, 5, 200)
dx = xs[1] - xs[0]

# Note for our injection parameters we set the amplitude of the second component to 0.
injection_params = dict(
    amplitude_0=-3,
    mu_0=-4,
    sigma_0=4,
    amplitude_1=0,
    mu_1=0,
    sigma_1=1,
    amplitude_2=4,
    mu_2=3,
    sigma_2=3,
)

# We calculate the injected curve and add some Gaussian noise on the data points
sigma = 0.02
p = bilby.core.prior.Gaussian(mu=0, sigma=sigma)
ys = triple_gaussian(xs=xs, **injection_params) + p.sample(len(xs))

plt.errorbar(xs, ys, yerr=sigma, fmt=".k", capsize=0, label="Injected data")
plt.plot(xs, triple_gaussian(xs=xs, **injection_params), label="True signal")
plt.legend()
plt.savefig(f"{outdir}/{label}_injected_data")
plt.clf()


# Now we want to set up our priors.
priors = bilby.core.prior.PriorDict()
# For the slab-and-spike prior, we first need to define the 'slab' part, which is just a regular bilby prior.
amplitude_slab_0 = bilby.core.prior.Uniform(
    minimum=-10, maximum=10, name="amplitude_0", latex_label="$A_0$"
)
amplitude_slab_1 = bilby.core.prior.Uniform(
    minimum=-10, maximum=10, name="amplitude_1", latex_label="$A_1$"
)
amplitude_slab_2 = bilby.core.prior.Uniform(
    minimum=-10, maximum=10, name="amplitude_2", latex_label="$A_2$"
)
# We do the following to create the slab-and-spike prior. The spike height is somewhat arbitrary and can
# be corrected in post-processing.
priors["amplitude_0"] = bilby.core.prior.SlabSpikePrior(
    slab=amplitude_slab_0, spike_location=0, spike_height=0.1
)
priors["amplitude_1"] = bilby.core.prior.SlabSpikePrior(
    slab=amplitude_slab_1, spike_location=0, spike_height=0.1
)
priors["amplitude_2"] = bilby.core.prior.SlabSpikePrior(
    slab=amplitude_slab_2, spike_location=0, spike_height=0.1
)
# Our problem has a degeneracy in the ordering. In general, this problem is somewhat difficult to resolve properly.
# See e.g. https://github.com/GregoryAshton/kookaburra/blob/master/src/priors.py#L72 for an implementation.
# We resolve this by not letting the priors overlap in this case.
priors["mu_0"] = bilby.core.prior.Uniform(
    minimum=-5, maximum=-2, name="mu_0", latex_label=r"$\mu_0$"
)
priors["mu_1"] = bilby.core.prior.Uniform(
    minimum=-2, maximum=2, name="mu_1", latex_label=r"$\mu_1$"
)
priors["mu_2"] = bilby.core.prior.Uniform(
    minimum=2, maximum=5, name="mu_2", latex_label=r"$\mu_2$"
)
priors["sigma_0"] = bilby.core.prior.LogUniform(
    minimum=0.01, maximum=10, name="sigma_0", latex_label=r"$\sigma_0$"
)
priors["sigma_1"] = bilby.core.prior.LogUniform(
    minimum=0.01, maximum=10, name="sigma_1", latex_label=r"$\sigma_1$"
)
priors["sigma_2"] = bilby.core.prior.LogUniform(
    minimum=0.01, maximum=10, name="sigma_2", latex_label=r"$\sigma_2$"
)

# Setting up the likelihood and running the samplers works the same as elsewhere.
likelihood = bilby.core.likelihood.GaussianLikelihood(
    x=xs, y=ys, func=triple_gaussian, sigma=sigma
)
result = bilby.run_sampler(
    likelihood=likelihood,
    priors=priors,
    outdir=outdir,
    label=label,
    sampler="pymultinest",
    nlive=400,
)

result.plot_corner(truths=injection_params)


# Let's also plot the maximum likelihood fit along with the data.
max_like_params = result.posterior.iloc[-1]
plt.errorbar(xs, ys, yerr=sigma, fmt=".k", capsize=0, label="Injected data")
plt.plot(xs, triple_gaussian(xs=xs, **injection_params), label="True signal")
plt.plot(xs, triple_gaussian(xs=xs, **max_like_params), label="Max likelihood fit")
plt.legend()
plt.savefig(f"{outdir}/{label}_max_likelihood_recovery")
plt.clf()

# Finally, we can check what fraction of amplitude samples are exactly on the spike.
spike_samples_0 = len(np.where(result.posterior["amplitude_0"] == 0.0)[0]) / len(
    result.posterior
)
spike_samples_1 = len(np.where(result.posterior["amplitude_1"] == 0.0)[0]) / len(
    result.posterior
)
spike_samples_2 = len(np.where(result.posterior["amplitude_2"] == 0.0)[0]) / len(
    result.posterior
)
print(f"{spike_samples_0 * 100:.2f}% of amplitude_0 samples are exactly 0.0")
print(f"{spike_samples_1 * 100:.2f}% of amplitude_1 samples are exactly 0.0")
print(f"{spike_samples_2 * 100:.2f}% of amplitude_2 samples are exactly 0.0")
