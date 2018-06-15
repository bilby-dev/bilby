#!/bin/python
"""
Tutorial to demonstrate the minimum number of steps required to run parameter
stimation on GW150914 using open data.

"""
import tupak

prior = tupak.gw.prior.BBHPriorSet(filename='GW150914.prior')
interferometers = tupak.gw.detector.get_event_data("GW150914")
likelihood = tupak.gw.likelihood.get_binary_black_hole_likelihood(interferometers)
result = tupak.run_sampler(likelihood, prior, label='GW150914')
result.plot_corner()
