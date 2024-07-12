#!/usr/bin/env python
"""
An example of how to use bilby with a (multi-modal) multivariate
Gaussian prior distribution.
"""

import bilby
import matplotlib as mpl
import numpy as np
from bilby.core.likelihood import GaussianLikelihood
from scipy import linalg, stats

# A few simple setup steps
label = "multivariate_gaussian_prior"
outdir = "outdir"
bilby.utils.check_directory_exists_and_if_not_mkdir(outdir)


# First, we define our "signal model", in this case a simple linear function
def model(time, m, c):
    return time * m + c


# For this example assume a very broad Gaussian likelihood, so that the
# posterior is completely dominated by the prior (and we can check the
# prior sampling looks correct!)
sigma = 300.0

# These lines of code generate the fake (noise-free) data
sampling_frequency = 1
time_duration = 10
time = np.arange(0, time_duration, 1 / sampling_frequency)
N = len(time)
data = model(time, 0.0, 0.0)  # noiseless data


# instantiate the GaussianLikelihood
likelihood = GaussianLikelihood(time, data, model, sigma=sigma)

# Create a Multivariate Gaussian prior distribution with two modes
names = ["m", "c"]
mus = [[-5.0, -5.0], [5.0, 5.0]]  # means of the two modes
corrcoefs = [
    [[1.0, -0.7], [-0.7, 1.0]],
    [[1.0, 0.7], [0.7, 1.0]],
]  # correlation coefficients of the two modes
sigmas = [[1.5, 1.5], [2.1, 2.1]]  # standard deviations of the two modes
weights = [1.0, 3.0]  # relative weights of each mode
nmodes = 2
mvg = bilby.core.prior.MultivariateGaussianDist(
    names, nmodes=2, mus=mus, corrcoefs=corrcoefs, sigmas=sigmas, weights=weights
)
priors = dict()
priors["m"] = bilby.core.prior.MultivariateGaussian(mvg, "m")
priors["c"] = bilby.core.prior.MultivariateGaussian(mvg, "c")

result = bilby.run_sampler(
    likelihood=likelihood,
    priors=priors,
    sampler="dynesty",
    nlive=4000,
    outdir=outdir,
    label=label,
)

fig = result.plot_corner(save=False)

# plot the priors (to show that they look correct)
axs = fig.get_axes()

# plot the 1d marginal distributions
x = np.linspace(-12, 12, 5000)
aidx = [0, 3]
for j in range(2):  # loop over parameters
    gp = np.zeros(len(x))
    for i in range(nmodes):  # loop over modes
        gp += weights[i] * stats.norm.pdf(x, loc=mus[i][j], scale=mvg.sigmas[i][j])
    gp = gp / np.trapz(gp, x)  # renormalise

    axs[aidx[j]].plot(x, gp, "k--", lw=2)

# plot the 2d distribution
for i in range(nmodes):
    v, w = linalg.eigh(mvg.covs[i])
    v = 2.0 * np.sqrt(2.0) * np.sqrt(v)
    u = w[0] / linalg.norm(w[0])
    angle = np.arctan(u[1] / u[0])
    angle = 180.0 * angle / np.pi  # convert to degrees
    ell = mpl.patches.Ellipse(
        xy=mus[i],
        width=v[0],
        height=v[1],
        angle=180.0 + angle,
        edgecolor="black",
        facecolor="none",
        lw=2,
        ls="--",
    )
    axs[2].add_artist(ell)

fig.savefig("{}/{}_corner.png".format(outdir, label), dpi=300)
