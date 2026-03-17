

=====
Bilby
=====

Added features: 

- Inject glitches to the data:
   - An example script is at ``examples/gw_examples/injection_examples/glitch.py``
   - Adds an ``inject_glitch`` method to the ``interferometer.py``. 
   - Useful to estimate the impact of glitches on PE or glitch mitigation related simulations.

- Earth rotation:
   - An example script is at ``examples/gw_examples/injection_examples/rotation.py``
   - Added a bool (``earth_rotation``) argument to the ``inject_signal`` method.
   - Also added the same bool to the base likelihood object, to control whether both (injection/recovery) should account for Earth rotation.
   - Currently only supports ``geocent_time`` as the reference time.
   - Does not support time marginalisation.
   
