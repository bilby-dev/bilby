import numpy as np

class likelihood:

	def __init__(self, detectors):

		self.detector = detectors
		
	def logL_cbc(self, source):
		
		self.logL = 0 

		waveform_polarizations = source.model(source.params)

		

		for detector in self.detectors:
				

			det_response = detector.response(source.params)

			signal = np.dot(waveform_polarizations, det_response) 
			
			self.logL += 4 * detector.deltaF * np.vdot( detector.data - signal, (detector.data - signal) / detector.psd )
		
			
				
			

			

	
		

		 
