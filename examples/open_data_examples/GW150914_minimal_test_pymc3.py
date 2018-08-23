#!/bin/python
"""
Tutorial to demonstrate the minimum number of steps required to run parameter
stimation on GW150914 using open data.

"""
import tupak
from tupak.core.prior import Uniform, DeltaFunction, Sine, Cosine
from collections import OrderedDict
import numpy as np

prior = OrderedDict()
prior['mass_1'] = Uniform(name='mass_1', minimum=30, maximum=50)
prior['mass_2'] = Uniform(name='mass_2', minimum=20, maximum=40)
prior['a_1'] = Uniform(name='a_1', minimum=0, maximum=0.8)
prior['a_2'] = Uniform(name='a_2', minimum=0, maximum=0.8)
prior['tilt_1'] = Sine(name='tilt_1')
prior['tilt_2'] = Sine(name='tilt_2')
prior['phi_12'] =  Uniform(name='phi_12', minimum=0, maximum=2 * np.pi)
prior['phi_jl'] =  Uniform(name='phi_jl', minimum=0, maximum=2 * np.pi)
#prior['luminosity_distance'] = DeltaFunction(1e3, name='luminosity_distance')
prior['luminosity_distance'] = tupak.gw.prior.UniformComovingVolume(name='luminosity_distance', minimum=1e2, maximum=1e3)
prior['dec'] = Cosine(name='dec')
prior['ra'] = Uniform(name='ra', minimum=0., maximum=2.*np.pi)
prior['iota'] = Sine(name='iota')
prior['psi'] = Uniform(name='psi', minimum=0, maximum=2 * np.pi)
prior['phase'] = Uniform(name='phase', minimum=0, maximum=2 * np.pi)
prior['geocent_time'] = Uniform(1126259462.322, 1126259462.522, name='geocent_time')

interferometers = tupak.gw.detector.get_event_data("GW150914")
likelihood = tupak.gw.likelihood.get_binary_black_hole_likelihood(interferometers)
result = tupak.run_sampler(likelihood, prior, label='GW150914', sampler='pymc3', draws=2000)
result.plot_corner()
