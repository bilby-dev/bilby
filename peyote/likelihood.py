import numpy as np

class likelihood:

	def __init__(self, detectors):

		self.detectors = detectors
		
	def logL_cbc(self, source):
		
		logL = 0 

		waveform_polarizations = source.model(source.params)

		

		for detector in self.detectors:
				

			det_response = detector.response(source.params)
			time_shift = detector.time_shift(source.params['geocent_time'])

			signal = np.dot(waveform_polarizations, det_response) 
				
			signal *= np.exp(-1j*2*np.pi*time_shift)

	
			logL += 4 * detector.deltaF * np.vdot( detector.data - signal, (detector.data - signal) / detector.psd )
		
		return logL	
				
			

			

	
		

		 
