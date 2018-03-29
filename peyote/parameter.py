#!/bin/python

import numpy as np
import prior


class Parameter:
    instances = []

    def __init__(self, name, prior=None, default=None):
        self.name = name
        self.default = default
        self.prior = prior
        self.is_fixed = False
        self.value = self.default
        Parameter.instances.append(self)

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

# define a bunch of parameter objects for CBCs

# component masses
mass1 = Parameter(name='mass1', prior=prior.PowerLaw(alpha=0, bounds=(5, 100)))
mass2 = Parameter(name='mass2', prior=prior.PowerLaw(alpha=0, bounds=(5, 100)))
chirp_mass = Parameter(name='mchirp', prior=prior.PowerLaw(alpha=0, bounds=(5, 100)))
mass_ratio = Parameter(name='q', prior=prior.PowerLaw(alpha=0, bounds=(0, 1)))

# spins
a1 = Parameter(name='a1', prior=prior.PowerLaw(alpha=0, bounds=(0, 1)), default=0)
a2 = Parameter(name='a2', prior=prior.PowerLaw(alpha=0, bounds=(0, 1)), default=0)
tilt1 = Parameter(name='tilt1', prior=prior.Sine(), default=0)
tilt2 = Parameter(name='tilt2', prior=prior.Sine(), default=0)
phi1 = Parameter(name='phi1', prior=prior.PowerLaw(alpha=0, bounds=(0, 2*np.pi)), default=0)
phi2 = Parameter(name='phi2', prior=prior.PowerLaw(alpha=0, bounds=(0, 2*np.pi)), default=0)

# extrinsic
distance = Parameter(name='distance', prior=prior.PowerLaw(alpha=2, bounds=(1e2, 5e3)), default=400)
#zz = Parameter(name='z', prior=prior.FromFile('SFR_redshift_prior.txt'))  # FIXME: This file doesn't exist
latitude = Parameter(name='latitude', prior=prior.Cosine(), default=0)
longitude = Parameter(name='longitude', prior=prior.PowerLaw(alpha=0, bounds=(0, 2*np.pi)), default=0)
inclination = Parameter(name='inclination', prior=prior.Sine(), default=0)
polarization = Parameter(name='polarization', prior=prior.PowerLaw(alpha=0, bounds=(0, 2*np.pi)), default=0)
phase = Parameter(name='phase', prior=prior.PowerLaw(alpha=0, bounds=(0, 2*np.pi)), default=0)

time_at_coalescence = Parameter(name='tc', default=1126259642.413)

# parameters for a sine wave

frequency = Parameter(name='frequency', prior=prior.PowerLaw(alpha=-1, bounds=(1e2, 2e3)))
amplitude = Parameter(name='amplitude', prior=prior.PowerLaw(alpha=0, bounds=(1*1e-24, 5*1e-24)))
phase = Parameter(name='phase', prior=prior.PowerLaw(alpha=1, bounds=(0, 2*np.pi)))
