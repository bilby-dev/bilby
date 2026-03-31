

=====
Bilby
=====

A bilby fork to develope features relevant for A# and/or 3G detectors:

- Inject glitches to the data:
   - An example script is at ``examples/gw_examples/injection_examples/glitch.py``
   - An example script to inject `blip glitches <https://git.ligo.org/melissa.lopez/gengli>`_ is at: ``examples/gw_examples/injection_examples/inject_glitch_from_gengli.py``. 
   - This feature is implemented by adding an ``inject_glitch`` method to the ``interferometer.py``. 
   - Useful to estimate the impact of glitches on PE or glitch mitigation related simulations.

- Earth rotation:
   - An example script is at ``examples/gw_examples/injection_examples/rotation.py``
   - This feature is implemented by adding a bool (``earth_rotation``) argument to the ``inject_signal`` method.
   - Also added the same bool to the base likelihood object, to control whether both (injection/recovery) should account for Earth rotation.
   - Currently only supports ``geocent_time`` as the reference time.
   - Does not support time marginalisation.
   
