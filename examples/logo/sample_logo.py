""" Script used to generate the samples for the tupak logo """
import tupak
import numpy as np
import scipy.interpolate as si
from skimage import io


class Likelihood(tupak.Likelihood):
    def __init__(self, interp):
        self.interp = interp
        self.parameters = dict(x=None, y=None)

    def log_likelihood(self):
        return -1/(self.interp(self.parameters['x'], self.parameters['y'])[0])


for letter in ['t', 'u', 'p', 'a', 'k']:
    img = 1-io.imread('{}.jpg'.format(letter), as_grey=True)[::-1, :]
    x = np.arange(img.shape[0])
    y = np.arange(img.shape[1])
    interp = si.interpolate.interp2d(x, y, img.T)

    likelihood = Likelihood(interp)

    priors = {}
    priors['x'] = tupak.prior.Uniform(0, max(x), 'x')
    priors['y'] = tupak.prior.Uniform(0, max(y), 'y')

    result = tupak.run_sampler(
        likelihood=likelihood, priors=priors, sampler='nestle', npoints=5000,
        label=letter)
    fig = result.plot_corner(quantiles=None, smooth=1)
