import numpy as np
import sys
from .reference_signal import *
from .helper import *

class Amplifier:
	"""
	A software Lock-in Amplifier
	"""

	def __init__(self, cutoff, pbar = True):
		"""
		Takes in a cutoff frequency (float) as an input
		as well as whether or not to display the progress
		bar.
		"""
		self.cutoff = cutoff
		self.pbar = pbar

	def update_cutoff(self, new_cutoff):
		"""
		Changes cutoff frequency to new value and returns
		new value
		"""
		self.cutoff = new_cutoff
		return new_cutoff

	def amplify(self, references, signal_input, fit_ref = True,
	 num_windows = 1, window_size = 1, interpolate = False):

		"""
		Performs simultaneous lock-in. See the docstrings in helper.py and 
		the tutorial example for a more detailed description of the input
		parameters and outputs. The docstring for the lock_in function in
		helper.py might be helpful. 
		"""

		#Fits the reference signals to sine waves.

		if fit_ref:

			ref_vals = fit(references)

			magnitudes = []
			angles = []
			mag_errors = []
			ang_errors = []
			fit_vals = {'frequencies' : [], 'phases' : []}
			for fit_params in ref_vals:
				est_freq, est_phase, est_offset, est_amp = fit_params[0],\
				 fit_params[1], fit_params[2], fit_params[3]
				fit_vals['frequencies'].append(est_freq)
				fit_vals['phases'].append(est_phase)
				#Timestamps
				time = np.asarray(signal_input['time'])
				signal = np.asarray(signal_input['signal'])
				size = signal.shape
				dim = len(signal.shape)

				#Reshaping the n-dimensional input into a 1D array for each timestamp
				arr_len = 1
				for i in range(1, dim):
					arr_len *= size[i]
				signal = np.reshape(signal, (size[0], arr_len))

				#Applies lock-in with errorbars
				curr_magnitudes, curr_angles, curr_mag_err, curr_phase_err, indices = lock_in(self, signal,
				 time, est_freq, est_phase, num_windows, window_size, interpolate)

				#Applies lock-in for results - only necessary if there is more than one window.
				if num_windows != 1:
					curr_magnitudes, curr_angles, _, _, _ = lock_in(self,signal,
					 time, est_freq, est_phase, num_windows = 1, window_size = 1,
					  interpolate = interpolate)

				magnitudes.append(curr_magnitudes)
				angles.append(curr_angles)
				mag_errors.append(curr_mag_err)
				ang_errors.append(curr_phase_err)

			i = 0
			out = {'ref. fit params' : fit_vals}
			if num_windows != 1:
				out['indices'] = indices
			while i < len(magnitudes):
				label = 'reference ' + str(i + 1)
				#reshaping output into their original form without the time dependence
				mags = np.reshape(magnitudes[i], size[1: dim])
				phases = np.reshape(angles[i], size[1: dim])
				out[label] = {'magnitudes' : mags.tolist(), 'phases' : phases.tolist()}
				if num_windows != 1:
					magnitude_stds = np.reshape(mag_errors[i], size[1: dim])
					phase_stds = np.reshape(ang_errors[i], size[1: dim])
					out[label]['magnitude stds'] = magnitude_stds.tolist()
					out[label]['phase stds'] = phase_stds.tolist()
				
				i += 1
		else:
			magnitudes = []
			angles = []
			mag_errors = []
			ang_errors = []
			for ref in references:
				ref_time = np.asarray(ref['time'])
				ref_sig = np.asarray(ref['signal'])
				sig_time = np.asarray(signal_input['time'])
				signal = np.asarray(signal_input['signal'])
				size = signal.shape
				dim = len(signal.shape)

				#Reshaping the n-dimensional input into a 1D array for each timestamp
				arr_len = 1
				for i in range(1, dim):
					arr_len *= size[i]
				signal = np.reshape(signal, (size[0], arr_len))

				#Applies lock-in with errorbars
				curr_magnitudes, curr_mag_err, indices = lock_in_no_fit(self, signal, sig_time, ref_sig, 
					ref_time, num_windows, window_size, interpolate)

				#Applies lock-in for results - only necessary if there is more than one window.
				if num_windows != 1:
					curr_magnitudes, _, _ = lock_in_no_fit(self,signal, sig_time, ref_sig,
						ref_time, num_windows = 1, window_size = 1,
						 interpolate = interpolate)

				magnitudes.append(curr_magnitudes)
				mag_errors.append(curr_mag_err)

			i = 0
			out = {}
			if num_windows != 1:
				out['indices'] = indices
			while i < len(magnitudes):
				label = 'reference ' + str(i + 1)
				#reshaping output into their original form without the time dependence
				mags = np.reshape(magnitudes[i], size[1: dim])
				out[label] = {'magnitudes' : mags.tolist()}
				if num_windows != 1:
					magnitude_stds = np.reshape(mag_errors[i], size[1: dim])
					out[label]['magnitude stds'] = magnitude_stds.tolist()				
				i += 1

		return out