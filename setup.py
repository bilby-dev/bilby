#!/usr/bin/env python

from distutils.core import setup

setup(name='tupak',
      version='0.1',
      packages=['tupak'],
      package_dir={'tupak': 'tupak'},
      package_data={'tupak': ['noise_curves/*.txt', 'prior_files/*.txt']}
      )
