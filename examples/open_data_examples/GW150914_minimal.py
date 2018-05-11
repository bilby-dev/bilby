#!/bin/python
"""
Tutorial to demonstrate the minimum number of steps required to run parameter
stimation on GW150914 using open data.

"""
import tupak

t0 = tupak.utils.get_event_time("GW150914")
prior = dict(geocent_time=tupak.prior.Uniform(t0-0.1, t0+0.1, name='geocent_time'))
interferometers = tupak.detector.get_event_data("GW150914")
likelihood = tupak.likelihood.get_binary_black_hole_likelihood(interferometers)
result = tupak.sampler.run_sampler(likelihood, prior, label='GW150914')
result.plot_corner()
