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