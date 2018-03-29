import numpy as np

class likelihood:

 def __init__(self, Interferometers):

  self.Interferometers = Interferometers

 def logL_cbc(self, source, params):

  logL = 0

  waveform_polarizations = source.model(params)


  for Interferometer in self.Interferometers:


   for mode in source.params['modes'].keys() :

    det_response = Interometer.response(parmams['ra'], parmams['dec'], parmams['psi'], mode)
    if signal is None:
     signal = waveform_polarizations['%s'%mode] * det_response
    else:
     signal += waveform_polarizations['%s'%mode] * det_response

			#time_shift = Interferometer.time_shift(source.params['geocent_time'])
			#signal *= np.exp(-1j*2*np.pi*time_shift) # This is just here as a reminder that a tc shift needs to be performed
			                                          # on frequency-domain GWs

   logL += 4 * Interferometer.deltaF * np.vdot( Interferometer.data - signal, (Interferometer.data - signal) / Interferometer.data.psd )

  return logL	
