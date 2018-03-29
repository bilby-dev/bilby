import numpy as np
import pdb

class likelihood:
	def __init__(self, Interferometers, source):
		self.Interferometers = Interferometers
		self.source = source

	def logL(self, params):
		logL = 0
		waveform_polarizations = self.source.model(params)
		for Interferometer in self.Interferometers:
			for mode in params['modes']:

				#det_response = Interferometer.antenna_response(
                                #        params['ra'], params['dec'],
                                #        params['geocent_time'], params['psi'],
                                #        mode )
				det_response = 1

				waveform_polarizations[mode] *= det_response

			signal_IFO = np.sum( waveform_polarizations.values() )
			print(waveform_polarizations.values()[0])
			pdb.set_trace()
			#time_shift = Interferometer.time_shift(source.params['geocent_time'])
			#signal *= np.exp(-1j*2*np.pi*time_shift) # This is just here as a reminder that a tc shift needs to be performed
			                                          # on frequency-domain GWs
			logL += 4. * params['deltaF'] * np.vdot( Interferometer.data - signal_IFO, ( Interferometer.data - signal_IFO ) / Interferometer.psd )

		return logL
