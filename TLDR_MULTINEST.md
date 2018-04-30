Here is a short version, if this fails refer to the full instructions.

First, install the dependencies (for Ubuntu/Linux):

```bash
$ sudo apt-get install python-{scipy,numpy,matplotlib,progressbar} ipython libblas{3,-dev} liblapack{3,-dev} libatlas{3-base,-dev} cmake build-essential git gfortran
```

For Mac, the advice in the instructions are "If you google for “MultiNest Mac OSX” or “PyMultiNest Mac OSX” you will find installation instructions".

The following will place a directory `MultiNest` in your `$HOME` directory, if you want
to place it somewhere, adjust the instructions as such.

```bash
git clone https://github.com/JohannesBuchner/MultiNest $HOME
cd $HOME/MultiNest/build
cmake ..
make
```

Finally, add the libraries to you path. Add this to the `.bashrc` file
```
export LD_LIBRARY_PATH=$HOME/Downloads/MultiNest/lib:
```

(you'll need to resource your `.bashrc` after this, i.e. run `bash`.

