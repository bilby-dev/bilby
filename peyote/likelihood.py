import numpy as np

class likelihood:

	def __init__( self, Interferometers ):

		self.Interferometers = Interferometers

	def logL_cbc( self, source, params ):

		logL = 0

		waveform_polarizations = source.model( params )


		for Interferometer in self.Interferometers:

			for mode in source.params['modes'].keys() :

				det_response = Interferometer.response( params['ra'], params['dec'], params['geocent_time'], params['psi'], mode )

				waveform_polarizations[mode] *= det_response

			signal_IFO = np.sum( waveform_polarizations.values() )

			#time_shift = Interferometer.time_shift(source.params['geocent_time'])
			#signal *= np.exp(-1j*2*np.pi*time_shift) # This is just here as a reminder that a tc shift needs to be performed
			                                          # on frequency-domain GWs
 			logL += 4. * Interferometer.deltaF * np.vdot( Interferometer.data - signal_IFO, ( Interferometer.data - signal_IFO ) / Interferometer.psd )

		return logL
