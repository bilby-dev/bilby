|pipeline status| |coverage report| |pypi| |conda| |version|

=====
Bilby
=====

A user-friendly Bayesian inference library.
Fulfilling all your Bayesian dreams.

Online material to help you get started:

-  `Installation instructions <https://lscsoft.docs.ligo.org/bilby/installation.html>`__
-  `Documentation <https://lscsoft.docs.ligo.org/bilby/index.html>`__

If you need help, find an issue, or just have a question/suggestion you can

- Email our support desk: contact+lscsoft-bilby-1846-issue-@support.ligo.org
- Join our `Slack workspace <https://bilby-code.slack.com/>`__
- Ask questions (or search through other users questions and answers) on `StackOverflow <https://stackoverflow.com/questions/tagged/bilby>`__ using the bilby tag
- For www.git.ligo.org users, submit issues directly through `the issue tracker <https://git.ligo.org/lscsoft/bilby/issues>`__

We encourage you to contribute to the development of bilby. This is done via a merge request.  For
help in creating a merge request, see `this page
<https://docs.gitlab.com/ee/gitlab-basics/add-merge-request.html>`__ or contact
us directly. For advice on contributing, see `this help page <https://git.ligo.org/lscsoft/bilby/blob/master/CONTRIBUTING.md>`__.


--------------
Citation guide
--------------

If you use :code:`bilby` in a scientific publication, please cite

* `Bilby: A user-friendly Bayesian inference library for gravitational-wave
  astronomy
  <https://ui.adsabs.harvard.edu/#abs/2018arXiv181102042A/abstract>`__

Additionally, :code:`bilby` builds on a number of open-source packages. If you
make use of this functionality in your publications, we recommend you cite them
as requested in their associated documentation.

**Samplers**

* `dynesty <https://github.com/joshspeagle/dynesty>`__
* `nestle <https://github.com/kbarbary/nestle>`__
* `pymultinest <https://github.com/JohannesBuchner/PyMultiNest>`__
* `cpnest <https://github.com/johnveitch/cpnest>`__
* `emcee <https://github.com/dfm/emcee>`__
* `ptemcee <https://github.com/willvousden/ptemcee>`__
* `ptmcmcsampler <https://github.com/jellis18/PTMCMCSampler>`__
* `pypolychord <https://github.com/PolyChord/PolyChordLite>`__
* `PyMC3 <https://github.com/pymc-devs/pymc3>`_


**Gravitational-wave tools**

* `gwpy <https://github.com/gwpy/gwpy>`__
* `lalsuite <https://git.ligo.org/lscsoft/lalsuite>`__
* `astropy <https://github.com/astropy/astropy>`__

**Plotting**

* `corner <https://github.com/dfm/corner.py>`__ for generating corner plot
* `matplotlib <https://github.com/matplotlib/matplotlib>`__ for general plotting routines


.. |pipeline status| image:: https://git.ligo.org/lscsoft/bilby/badges/master/pipeline.svg
   :target: https://git.ligo.org/lscsoft/bilby/commits/master
.. |coverage report| image:: https://lscsoft.docs.ligo.org/bilby/coverage_badge.svg
   :target: https://lscsoft.docs.ligo.org/bilby/htmlcov/
.. |pypi| image:: https://badge.fury.io/py/bilby.svg
   :target: https://pypi.org/project/bilby/
.. |conda| image:: https://img.shields.io/conda/vn/conda-forge/bilby.svg
   :target: https://anaconda.org/conda-forge/bilby
.. |version| image:: https://img.shields.io/pypi/pyversions/bilby.svg
   :target: https://pypi.org/project/bilby/
