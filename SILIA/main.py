import numpy as np
import sys
from .reference_signal import *
from .helper import *

class Amplifier:
	"""
	A software Lock-in Amplifier
	"""

	def __init__(self, cutoff):
		"""
		Takes in a cutoff frequency (float) as an input
		"""
		self.cutoff = cutoff

	def update_cutoff(self, new_cutoff):
		"""
		Changes cutoff frequency to new value and returns
		new value
		"""
		self.cutoff = new_cutoff
		return new_cutoff

	def amplify(self, references, signal_input):

		"""
		Performs simultaneous lock-in.
		"""

		#Fits the reference signals to sine waves.

		ref_vals = setUp(references)

		magnitudes = []
		angles = []
		references = {'frequencies' : [], 'phase' : []}
		for fit_params in ref_vals:
			est_freq, est_phase, est_offset, est_amp = fit_params[0], fit_params[1], fit_params[2], fit_params[3]
			references['frequencies'].append(est_freq)
			references['phase'].append(est_phase)
			#Timestamps
			time = np.asarray(signal_input['time'])
			signal = np.asarray(signal_input['signal'])
			size = signal.shape
			dim = len(signal.shape)

			#Reshaping the n-dimensional input into a 1D array for each timestamp
			arr_len = 1
			for i in range(1, dim):
				arr_len *= size[i]
			signal = np.reshape(signal, (len(time), arr_len))

			#Mixes the intensity signal with the normal and phase shifted reference signals.
			mixed, mixed_phaseShift = mix(signal, time, est_freq, est_phase)

			#Applies lowpass filter
			curr_magnitudes, curr_angles = apply_lowpass(mixed, mixed_phaseShift, time, self.cutoff)
			magnitudes.append(curr_magnitudes)
			angles.append(curr_angles)

		i = 0
		out = {'references' : references}
		while i < len(magnitudes):
			label = 'reference ' + str(i + 1)
			#reshaping output into their original form without the time dependence
			mags = np.reshape(magnitudes[i], size[1: dim])
			phases = np.reshape(angles[i], size[1: dim])
			out[label] = {'magnitudes' : magnitudes[i], 'phases' : phases}
			i += 1
		return out