#!/usr/bin/env python

from setuptools import setup
import subprocess
from os import path

version = '0.2.1'

# Write a version file containing the git hash and info
try:
    git_log = subprocess.check_output(
        ['git', 'log', '-1', '--pretty=%h %ai']).decode('utf-8')
    git_diff = (subprocess.check_output(['git', 'diff', '.'])
                + subprocess.check_output(
                    ['git', 'diff', '--cached', '.'])).decode('utf-8')
    if git_diff == '':
        status = '(CLEAN) ' + git_log
    else:
        status = '(UNCLEAN) ' + git_log
except subprocess.CalledProcessError:
    status = ''

version_file = '.version'
if path.isfile(version_file) is False:
    with open('tupak/' + version_file, 'w+') as f:
        f.write('{} - {}'.format(version, status))


here = path.abspath(path.dirname(__file__))
with open(path.join(here, 'README.rst')) as f:
        long_description = f.read()

setup(name='tupak',
      description='The User friendly Parameter estimAtion Kode',
      long_description=long_description,
      url='https://git.ligo.org/Monash/tupak',
      author='Greg Ashton, Moritz Huebner, Paul Lasky, Colm Talbot',
      author_email='paul.lasky@monash.edu',
      license="MIT",
      version=version,
      packages=['tupak', 'tupak.core', 'tupak.gw', 'tupak.hyper', 'cli_tupak'],
      package_dir={'tupak': 'tupak'},
      package_data={'tupak.gw': ['prior_files/*', 'noise_curves/*.txt', 'detectors/*'],
                    'tupak': [version_file]},
      install_requires=[
          'future',
          'dynesty',
          'corner',
          'numpy>=1.9',
          'matplotlib>=2.0',
          'deepdish',
          'pandas',
          'scipy',
          ],
      entry_points={'console_scripts':
                    ['tupak_plot=cli_tupak.plot_multiple_posteriors:main']
                    })

