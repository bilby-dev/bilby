#!/usr/bin/env python

from setuptools import setup
import subprocess
from os import path


def write_version_file(version):
    """ Writes a file with version information to be used at run time

    Parameters
    ----------
    version: str
        A string containing the current version information

    Returns
    -------
    version_file: str
        A path to the version file

    """
    try:
        git_log = subprocess.check_output(
            ['git', 'log', '-1', '--pretty=%h %ai']).decode('utf-8')
        git_diff = (subprocess.check_output(['git', 'diff', '.'])
                    + subprocess.check_output(
                        ['git', 'diff', '--cached', '.'])).decode('utf-8')
        if git_diff == '':
            git_status = '(CLEAN) ' + git_log
        else:
            git_status = '(UNCLEAN) ' + git_log
    except Exception as e:
        print("Unable to obtain git version information, exception: {}"
              .format(e))
        git_status = ''

    version_file = '.version'
    if path.isfile(version_file) is False:
        with open('tupak/' + version_file, 'w+') as f:
            f.write('{}: {}'.format(version, git_status))

    return version_file


def get_long_description():
    """ Finds the README and reads in the description """
    here = path.abspath(path.dirname(__file__))
    with open(path.join(here, 'README.rst')) as f:
            long_description = f.read()
    return long_description


version = '0.2.2'
version_file = write_version_file(version)
long_description = get_long_description()

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
      package_data={'tupak.gw': ['prior_files/*', 'noise_curves/*.txt',
                                 'detectors/*'],
                    'tupak': [version_file]},
      install_requires=[
          'future',
          'dynesty',
          'corner',
          'numpy>=1.9',
          'matplotlib>=2.0',
          'deepdish',
          'pandas',
          'scipy'],
      entry_points={'console_scripts':
                    ['tupak_plot=cli_tupak.plot_multiple_posteriors:main']
                    })
