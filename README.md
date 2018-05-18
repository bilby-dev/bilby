[![pipeline status](https://git.ligo.org/Monash/tupak/badges/master/pipeline.svg)](https://git.ligo.org/Monash/tupak/commits/master)
[![coverage report](https://monash.docs.ligo.org/tupak/coverage_report.svg)](
https://monash.docs.ligo.org/tupak/)

# Tupak

Fulfilling all your gravitational wave dreams.

## Example

To get started with `tupak`, we have a few examples and tutorials:

1. [Examples of injecting and recovering data](https://git.ligo.org/Monash/tupak/tree/master/examples/injection_examples)
    * [A basic tutorial](https://git.ligo.org/Monash/tupak/blob/master/examples/injection_examples/basic_tutorial.py)
    * [Create your own source model](https://git.ligo.org/Monash/tupak/blob/master/examples/injection_examples/create_your_own_source_model.py)
    * [Create your own time-domain source model](https://git.ligo.org/Monash/tupak/blob/master/examples/injection_examples/create_your_own_time_domain_source_model.py)
    * [How to specify the prior](https://git.ligo.org/Monash/tupak/blob/master/examples/injection_examples/how_to_specify_the_prior.py)
    * [Using a partially marginalized likelihood](https://git.ligo.org/Monash/tupak/blob/master/examples/injection_examples/marginalized_likelihood.py)

2. [Examples using open data](https://git.ligo.org/Monash/tupak/tree/master/examples/open_data_examples)
    * [Analysing the first Binary Black hole detection, GW150914](https://git.ligo.org/Monash/tupak/blob/master/examples/open_data_examples/GW150914.py)

3. [Notebook-style tutorials](https://git.ligo.org/Monash/tupak/tree/master/examples/tutorials)
    * [Comparing different samplers](https://git.ligo.org/Monash/tupak/blob/master/examples/tutorials/compare_samplers.ipynb)
    * [Visualising the output](https://git.ligo.org/Monash/tupak/blob/master/examples/tutorials/visualising_the_results.ipynb)


## Installation

`tupak` is developed to work with both Python 2.7+ and Python 3+. In the
following, we assume you have a working python installation, [python
pip](https://packaging.python.org/tutorials/installing-packages/#use-pip-for-installing),
and [git](https://git-scm.com/).

### Install tupak
Clone the repository, install the requirements, and then install `tupak`.
```bash
$ git clone git@git.ligo.org:Monash/tupak.git
$ cd tupak/
$ pip install -r requirements.txt
$ python setup.py install
```

Once you have run these steps, you have `tupak` installed. You can now try to
run the examples. However, for many gravitational-wave related examples you'll
also need to install `lalsuite`.

### Install lalsuite
We recommend you install `lalsuite` the simple way:

```bash
$ pip install lalsuite.
```

If this doesn't work, or you prefer to, you can also install from source.
Head to
[https://git.ligo.org/lscsoft/lalsuite](https://git.ligo.org/lscsoft/lalsuite)
to check you have an account and SSH keys set up. Then,

```bash
$ git lfs install # you may need to install git-lfs first
$ git clone git@git.ligo.org:lscsoft/lalsuite.git
$ cd lalsuite
$ ./00boot
$ ./configure --prefix=${HOME}/lalsuite-install --disable-all-lal --enable-swig  --enable-lalsimulation
$ make; make install
```

Warning: in the configure line here, we have disabled everything except
lalsimulation. If you need other modules, see `./configure --help`.

### Install pymultinest

If you want to use the `pymultinest` sampler, you first need the
MultiNest library to be installed to work properly. The full instructions can
be found [here](https://johannesbuchner.github.io/PyMultiNest/install.html). We
have also written [a shortened tl;dr here](./TLDR_MULTINEST.md).


## Tests and coverage

We have a variety of tests which can be found in the `tests` directory.  You
can find a [current coverage report for master
here.](https://monash.docs.ligo.org/tupak/).


