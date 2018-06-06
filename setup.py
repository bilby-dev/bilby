#!/usr/bin/env python

from distutils.core import setup
import subprocess

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
    f.write(status)


setup(name='tupak',
      version='0.1',
      packages=['tupak', 'tupak.core', 'tupak.gw'],
      package_dir={'tupak': 'tupak'},
      package_data={'tupak.core': ['noise_curves/*.txt'],
                    'tupak.gw': ['prior_files/*.txt'],
                    'tupak': [version_file]}
      )
