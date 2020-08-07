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



def mix(signal_input, est_freq, est_phase):
	"""
	Performs the signal mixing step of a lock in amplifier.
	Mixes, or multiplies, the intensity signal for all channels
	by the fitted reference signal as well as its pi/2 phase shift.
	Also linearly interpolates the input so the mixed signal has
	consistent timesteps and applies the kaiser window to 
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
	time = signal_input['time']
	signal = np.array(signal_input['signal'])
	interpolated = scipy.interpolate.interp1d(time, signal, bounds_error=False, kind='cubic', axis = 0)
	signal = interpolated(time)
	print("Started Mixing", flush = True)
	num_rows = len(signal)
	num_cols = len(signal[0])
	mixed = []
	mixed_phaseShift = []
	for i in trange(num_rows):
		j = 0
		timeStamp = time[i] #timestamp

		ref_value = refValue(timeStamp, est_freq, est_phase)
		ref_value_phaseShift = refValue_phaseShift(timeStamp, est_freq, est_phase)

		mixed.append([])
		mixed_phaseShift.append([])
		while (j < num_cols):
			mixed[i].append(ref_value * (signal[i, j]))
			mixed_phaseShift[i].append(ref_value_phaseShift * (signal[i, j]))
			j += 1

	mixed = np.asarray(mixed)
	mixed_phaseShift = np.asarray(mixed_phaseShift)
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
	index_lower = len(fourier) - 1  
	for index, freq in enumerate(frequencies):
		if freq > cutoff:
			index_upper = index
			break

	for index, freq in enumerate(frequencies):
		if (freq < 0 and freq > -1 * cutoff):
			index_lower = index - 1 #Subtracting 1 from index to ensure upper and lower frequencies are the same
			break
	for index, freq in enumerate(frequencies):
		if index < index_lower and index > index_upper:
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
	for i in trange(num_channels):
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

