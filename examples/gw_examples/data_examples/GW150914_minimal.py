#!/usr/bin/env python
"""
Tutorial to demonstrate the minimum number of steps required to run parameter
stimation on GW150914 using open data.

"""
import bilby

prior = bilby.gw.prior.BBHPriorDict(filename='GW150914.prior')
interferometers = bilby.gw.detector.get_event_data("GW150914")
likelihood = bilby.gw.likelihood.get_binary_black_hole_likelihood(interferometers)
result = bilby.run_sampler(likelihood, prior, label='GW150914')
result.plot_corner()
