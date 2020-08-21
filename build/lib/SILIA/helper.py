import numpy as np
from numpy.fft import rfft, irfft
from tqdm import tqdm
from tqdm import trange
import scipy.interpolate


def find_nearest(array, value):
	"""
	Returns the index of the the array element closest 
	to a desired value in an array. 
	"""
	array = np.asarray(array)
	idx = (np.abs(array - value)).argmin()
	return idx


def refValue (t, est_freq, est_phase):
	"""
	Returns the value of the fitted reference signal 
	at a given timestamp, t. 
	"""
	return 2 * np.sin(est_freq*t*2*np.pi + est_phase)


def refValue_phaseShift (t, est_freq, est_phase):
	"""
	Returns the value of the fitted reference signal 
	phase shifted by pi/2 at a given timestamp, t. 
	"""
	return 2 * np.cos(est_freq*t*2*np.pi + est_phase)



def mix(signal, time, est_freq, est_phase):
	"""
	Performs the signal mixing step of a lock in amplifier.
	Mixes, or multiplies, the intensity signal for all channels
	by the fitted reference signal as well as its pi/2 phase shift.
	Also performs cubic interpolation on the input so the mixed signal has
	consistent timesteps and applies the Hanning window to 
	mitigate sidelobes. 
	Parameters
	----------
	signal : 2D array of floats
		Timestamp and intensity values for each channel.
		Each row is a set of intensity values for each channel with a timestamp.
	est_freq : float
		Estimated frequency of the reference signal.
	est_phase : float
		Estimated phase of the reference signal.
	Returns
	-------
	mixed : 2D array of floats
		signal multiplied by reference signal.
		Each row is a set of mixed values for each channel with a timestamp.
	mixed_phaseShift : 2D array of floats
		signal multiplied by phase shifted reference signal.
		Same formatting as mixed. 
	"""
	#Shifts intensity values so each signal is centered at 0.
	print("Mixing...", flush = True)
	interpolated = scipy.interpolate.interp1d(time, signal, bounds_error=False, kind='cubic', axis = 0)
	min_time = min(time)
	max_time = max(time)
	len_time = len(time)
	timestep = (max(time) - min(time))/len(time)
	even_time = np.arange(min_time, max_time, timestep)
	signal = interpolated(even_time)
	num_rows = len(signal)
	num_cols = len(signal[0])
	ref_vals = refValue(even_time, est_freq, est_phase)
	ref_vals_phaseShift = refValue_phaseShift(even_time, est_freq, est_phase)
	mixed = np.multiply(signal, np.array([ref_vals]).T)
	mixed_phaseShift = np.multiply(signal, np.array([ref_vals_phaseShift]).T)

	window = np.hanning(num_rows)
	mixed = mixed * window.reshape((window.size, 1))
	mixed_phaseShift = mixed_phaseShift * window.reshape((window.size, 1))
	return mixed, mixed_phaseShift

def fft_lowpass(data, cutoff, f_s, time):
	"""
	Lowpass filter using the numpy fft algorithm.
	Used to filter the mixed signal for single channels.
	zero pads the input data
	Parameters
	----------
	data : 1D array of floats
	    Signal data over time. 
	cutoff : float
		Cutoff frequency for lowpass filter.
	f_s : float
	    Sampling frequency of the intensity values. 
	time : 1D array of floats
		The timestamps for the signal data.
	
	Returns
	-------
	filtered_signal : 1D array of floats
		Signal after being filtered. 
	"""
	n = len(data)
	fourier = rfft(data)
	frequencies = np.fft.rfftfreq(len(time)) * f_s
	index_upper = 0
	for index, freq in enumerate(frequencies):
		if freq > cutoff:
			index_upper = index
			break

	for index, freq in enumerate(frequencies):
		if index >= index_upper:
			fourier[index] = 0
	fourier *= 2
	filtered_signal = irfft(fourier)
	return filtered_signal


def apply_lowpass(mixed, mixed_phaseShift, time, cutoff):
	"""
	Applies lowpass filter to the mixed signals to get cartesian lock in values
	for each measured channel.
	Parameters
	----------
	mixed : 2D array of floats
		signal multiplied by reference signal.
		Each row is a set of mixed values for each channel.
	mixed_phaseShift : 2D array of floats
		signal multiplied by phase shifted reference signal.
		Same formatting as mixed. 
	time : 1D array of floats
		Timestamps for each signal measurement.
	cutoff : float
		Cutoff frequency for lowpass filter.
	"""

	print("Applying Lowpass on each Channel", flush = True)

	timeSteps = len(time)
	totalTime = time[timeSteps - 1] - time[0]
	timePerSample = totalTime/timeSteps

	sample_rate = 1/timePerSample
	num_channels = len(mixed[0])

	r = []
	theta = []
	for i in trange(num_channels, position= 0, leave = True):
		data = mixed[:,i]
		data_phaseShift = mixed_phaseShift[:,i]
		filteredColumn = fft_lowpass(data, cutoff, sample_rate, time)
		filteredColumn_phaseShift = fft_lowpass(data_phaseShift, cutoff, sample_rate, time)
		values = np.sqrt(np.power(np.absolute(filteredColumn), 2) + np.power(np.absolute(filteredColumn_phaseShift), 2))
		angles = np.arctan(np.absolute(filteredColumn_phaseShift)/np.absolute(filteredColumn))
		angle = np.mean(angles)
		value = np.mean(values)
		r.append(value)
		theta.append(angle)
	return r, theta

