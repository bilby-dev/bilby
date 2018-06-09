#!/bin/python
"""
Tutorial to demonstrate the minimum number of steps required to run parameter
stimation on GW150914 using open data.

"""
import tupak

t0 = tupak.gw.utils.get_event_time("GW150914")
prior = dict(geocent_time=tupak.core.prior.Uniform(t0 - 0.1, t0 + 0.1, name='geocent_time'))
interferometers = tupak.gw.detector.get_event_data("GW150914")
likelihood = tupak.gw.likelihood.get_binary_black_hole_likelihood(interferometers)
result = tupak.run_sampler(likelihood, prior, label='GW150914')
result.plot_corner()
