#!/usr/bin/env python

from distutils.core import setup
import subprocess
from os import path

version = '0.1.4'

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
    status = "NO VERSION INFORMATION"

version_file = '.version'
with open('tupak/' + version_file, 'w+') as f:
    f.write('{} - {}'.format(version, status))


here = path.abspath(path.dirname(__file__))
with open(path.join(here, 'README.rst')) as f:
        long_description = f.read()

setup(name='tupak',
      description='The User friendly Parameter estimAtion Kode',
      long_description=long_description,
      url='https://git.ligo.org/Monash/tupak',
      author='Greg Ashton, Moritz Hubner, Paul Lasky, Colm Talbot',
      author_email='paul.lasky@monash.edu',
      license="MIT",
      version=version,
      packages=['tupak', 'tupak.core', 'tupak.gw'],
      package_dir={'tupak': 'tupak'},
      package_data={'tupak.gw': ['prior_files/*', 'noise_curves/*.txt'],
                    'tupak': [version_file]}
      )
