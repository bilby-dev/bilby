""" Script used to generate the samples for the bilby logo """
import bilby
import numpy as np
import scipy.interpolate as si
from skimage import io


class Likelihood(bilby.Likelihood):
    def __init__(self, interp):
        self.interp = interp
        super().__init__(parameters=dict(x=None, y=None))

    def log_likelihood(self):
        return -1 / (self.interp(self.parameters["x"], self.parameters["y"])[0])


for letter in ["B", "I", "L", "Y"]:
    img = 1 - io.imread("{}.png".format(letter), as_gray=True)[::-1, :]
    x = np.arange(img.shape[0])
    y = np.arange(img.shape[1])
    interp = si.RectBivariateSpline(x, y, img, kx=1, ky=1)

    likelihood = Likelihood(interp)

    priors = {}
    priors["x"] = bilby.prior.Uniform(0, max(x), "x")
    priors["y"] = bilby.prior.Uniform(0, max(y), "y")

    result = bilby.run_sampler(
        likelihood=likelihood,
        priors=priors,
        sampler="nestle",
        npoints=5000,
        label=letter,
    )
    fig = result.plot_corner(quantiles=None)
