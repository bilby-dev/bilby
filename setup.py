#!/usr/bin/env python

from distutils.core import setup

setup(name='peyote',
      version='0.1',
      packages=['peyote'],
      package_dir={'peyote': 'peyote'},
      package_data={'peyote': ['noise_curves/*.txt']}
      )
