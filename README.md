[![pipeline status](https://git.ligo.org/Monash/peyote/badges/master/pipeline.svg)](https://git.ligo.org/Monash/peyote/commits/master)
[![coverage report](https://monash.docs.ligo.org/peyote/coverage.svg)](
https://monash.docs.ligo.org/peyote/)

# PEYOte

Fulfilling all your GW dreams.

In the following, we assume you have installed
[pip](https://pip.pypa.io/en/stable/installing/) and [git](https://git-scm.com/).

First, clone the repository, install the requirements, and then install `Peyote`.
```bash
$ git clone git@git.ligo.org:Monash/peyote.git
$ pip install -r requirements.txt
$ python setup.py install
```

Once you have run these three steps, you have `Peyote installed`. However, you
aren't quite yet ready to run anything. First,
`pymultinest` needs the MultiNest library to be installed to work properly. The
full instructions can be found
[here](https://johannesbuchner.github.io/PyMultiNest/install.html). We have
also written [a shortened tl;dr here](./TLDR_MULTINEST.md).

Second, you need `lalsimulation` from `lalsuite` installed. To do this head
to [https://git.ligo.org/lscsoft/lalsuite](https://git.ligo.org/lscsoft/lalsuite)
to check you have an account and SSH keys set up. Then,

```bash
$ git lfs install
$ git clone git@git.ligo.org:lscsoft/lalsuite.git
$ cd lalsuite
$ ./00boot
$ ./configure --prefix=${HOME}/lalsuite-install --disable-all-lal --enable-swig  --enable-lalsimulation
$ make; make install
```

** warning **: in the configure line here, we have disabled everything except lalsimulation. If you need other modules, see `./configure --help`.

You could also `pip install lal, lalsuite`.

## Tests and coverage

To locally test the code

```bash
$ python tests.py
```

To locally generate a coverage report

```bash
$ pip install coverage
$ coverage run tests.py
$ coverage html
```

This will generate a directory `htmlcov`, to see detailed coverage navigate
from your browser to the file `peyote/htmlcov/index.html`.

The coverage report for master can be seen here:
[https://monash.docs.ligo.org/peyote/](https://monash.docs.ligo.org/peyote/).


