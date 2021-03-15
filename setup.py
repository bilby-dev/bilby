#!/usr/bin/env python

from setuptools import setup
import subprocess
import sys
import os

# check that python version is 3.5 or above
python_version = sys.version_info
if python_version < (3, 5):
    sys.exit("Python < 3.5 is not supported, aborting setup")
print("Confirmed Python version {}.{}.{} >= 3.5.0".format(*python_version[:3]))


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
        git_diff = (subprocess.check_output(['git', 'diff', '.']) +
                    subprocess.check_output(
                        ['git', 'diff', '--cached', '.'])).decode('utf-8')
        if git_diff == '':
            git_status = '(CLEAN) ' + git_log
        else:
            git_status = '(UNCLEAN) ' + git_log
    except Exception as e:
        print("Unable to obtain git version information, exception: {}"
              .format(e))
        git_status = 'release'

    version_file = '.version'
    if os.path.isfile(version_file) is False:
        with open('bilby/' + version_file, 'w+') as f:
            f.write('{}: {}'.format(version, git_status))

    return version_file


def get_long_description():
    """ Finds the README and reads in the description """
    here = os.path.abspath(os.path.dirname(__file__))
    with open(os.path.join(here, 'README.rst')) as f:
        long_description = f.read()
    return long_description


def get_requirements():
    with open("requirements.txt", "r") as ff:
        requirements = ff.readlines()
    return requirements


# get version info from __init__.py
def readfile(filename):
    with open(filename) as fp:
        filecontents = fp.read()
    return filecontents


VERSION = '1.1.1'
version_file = write_version_file(VERSION)
long_description = get_long_description()

setup(name='bilby',
      description='A user-friendly Bayesian inference library',
      long_description=long_description,
      long_description_content_type="text/x-rst",
      url='https://git.ligo.org/lscsoft/bilby',
      author='Greg Ashton, Moritz Huebner, Paul Lasky, Colm Talbot',
      author_email='paul.lasky@monash.edu',
      license="MIT",
      version=VERSION,
      packages=['bilby', 'bilby.core', 'bilby.core.prior', 'bilby.core.sampler',
                'bilby.gw', 'bilby.gw.detector', 'bilby.gw.sampler',
                'bilby.hyper', 'bilby.gw.eos', 'cli_bilby'],
      package_dir={'bilby': 'bilby', 'cli_bilby': 'cli_bilby'},
      package_data={'bilby.gw': ['prior_files/*'],
                    'bilby.gw.detector': ['noise_curves/*.txt', 'detectors/*'],
                    'bilby.gw.eos': ['eos_tables/*.dat'],
                    'bilby': [version_file]},
      python_requires='>=3.5',
      install_requires=get_requirements(),
      entry_points={'console_scripts':
                    ['bilby_plot=cli_bilby.plot_multiple_posteriors:main',
                     'bilby_result=cli_bilby.bilby_result:main']
                    },
      classifiers=[
          "Programming Language :: Python :: 3.6",
          "Programming Language :: Python :: 3.7",
          "License :: OSI Approved :: MIT License",
          "Operating System :: OS Independent"])
