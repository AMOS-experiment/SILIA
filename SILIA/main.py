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

	def amplify(self, channels, references, signal_input):

		"""
		Performs simultaneous lock-in.
		"""

		#Fits the reference signals to sine waves.

		ref_vals = setUp(references)

		magnitudes = []
		angles = []
		reference_frequencies = []
		for fit_params in ref_vals:
			est_freq, est_phase, est_offset, est_amp = fit_params[0], fit_params[1], fit_params[2], fit_params[3]
			reference_frequencies.append(est_freq)
			#Timestamps
			time = signal_input['time']
			#Mixes the intensity signal with the normal and phase shifted reference signals.
			mixed, mixed_phaseShift = mix(signal_input, est_freq, est_phase)

			#Applies lowpass filter
			curr_magnitudes, curr_angles = apply_lowpass(mixed, mixed_phaseShift, time, self.cutoff)
			magnitudes.append(curr_magnitudes)
			angles.append(curr_angles)

		i = 0
		out = {'channels' : channels, 'reference frequencies' : reference_frequencies}
		while i < len(magnitudes):
			label = 'reference ' + str(i + 1)
			out[label] = {'magnitudes' : magnitudes[i], 'phase' : angles[i]}
			i += 1
		return out