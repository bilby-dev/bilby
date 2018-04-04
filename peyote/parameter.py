from __future__ import division, print_function
import numpy as np
from peyote import prior as prior


class Parameter:

    def __init__(self, name, prior=None, default=None, latex_label=None):
        self.name = name
        self.default = default
        self.prior = prior
        self.is_fixed = False
        self.value = self.default
        if latex_label is None:
            self.latex_label = name
        else:
            self.latex_label = latex_label

    def fix(self, value=None):
        """
        Specify parameter as fixed, this will not be sampled.
        """
        self.is_fixed = True
        if value is not None: self.default = value
        self.prior = prior.DeltaFunction(self.default)
        return None

    def set_value(self, value):
        """Set a value for the parameter"""
        self.value = value
        return None


# Default prior parameters

# Component masses
mass1 = Parameter(name='mass1', prior=prior.PowerLaw(alpha=0, bounds=(5, 100)),
                  latex_label='$m_1$')
mass2 = Parameter(name='mass2', prior=prior.PowerLaw(alpha=0, bounds=(5, 100)),
                  latex_label='$m_2$')
chirp_mass = Parameter(name='mchirp', prior=prior.PowerLaw(alpha=0, bounds=(5, 100)),
                       latex_label='$\mathcal{M}$')
mass_ratio = Parameter(name='q', prior=prior.PowerLaw(alpha=0, bounds=(0, 1)))

# spins
a1 = Parameter(name='a1', prior=prior.PowerLaw(alpha=0, bounds=(0, 1)), default=0)
a2 = Parameter(name='a2', prior=prior.PowerLaw(alpha=0, bounds=(0, 1)), default=0)
tilt1 = Parameter(name='tilt1', prior=prior.Sine(), default=0)
tilt2 = Parameter(name='tilt2', prior=prior.Sine(), default=0)
phi1 = Parameter(name='phi1', prior=prior.PowerLaw(alpha=0, bounds=(0, 2*np.pi)), default=0)
phi2 = Parameter(name='phi2', prior=prior.PowerLaw(alpha=0, bounds=(0, 2*np.pi)), default=0)

# extrinsic
luminosity_distance = Parameter(
    name='luminosity_distance',
    prior=prior.PowerLaw(alpha=2, bounds=(1e2, 5e3)), default=400)
# zz = Parameter(name='z', prior=prior.FromFile('SFR_redshift_prior.txt'))  # FIXME: This file doesn't exist
dec = Parameter(name='dec', prior=prior.Cosine(), default=0)
ra = Parameter(name='ra', prior=prior.PowerLaw(alpha=0, bounds=(0, 2*np.pi)), default=0)
inclination = Parameter(name='iota', prior=prior.Sine(), default=0)
psi = Parameter(name='psi', prior=prior.PowerLaw(alpha=0, bounds=(0, 2*np.pi)), default=0)
phase = Parameter(name='phase', prior=prior.PowerLaw(alpha=0, bounds=(0, 2*np.pi)), default=0)

time_at_coalescence = Parameter(name='tc', default=1126259642.413)
