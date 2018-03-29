"""
Basic tutorial to get PEYOte running
"""
import numpy as np

import peyote.source as src
import peyote.parameter as par

time = np.linspace(0, 100, 10000)

params = [par.amplitude, par.frequency, par.time_at_coalescence]

foo = src.SimpleSinusoidSource('test', params)
ht = foo.waveform(time)

plt.plot(time, ht)
plt.show()
