import numpy as np
from numpy.fft import rfft, irfft, rfftfreq
from tqdm import trange
import scipy.interpolate
import scipy.stats as sp
import sys


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
	return np.sin(est_freq*t*2*np.pi + est_phase)


def refValue_phaseShift (t, est_freq, est_phase):
	"""
	Returns the value of the fitted reference signal 
	phase shifted by pi/2 at a given timestamp, t. 
	"""
	return np.cos(est_freq*t*2*np.pi + est_phase)



def mix(signal, time, est_freq, est_phase, interpolate):
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
		Intensity values for each channel over time.
	time : 1D array of floats
		Timestamps for the data
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
	print("Mixing...", flush = True)
	if interpolate:
		min_time = min(time)
		max_time = max(time)
		len_time = len(time)
		timestep = (max(time) - min(time))/len(time)
		even_time = np.arange(min_time, max_time, timestep)
		interpolated = scipy.interpolate.interp1d(time, signal, bounds_error=False,
		 kind='cubic', axis = 0, fill_value = "extrapolate")
		signal = interpolated(even_time)
		ref_vals = refValue(even_time, est_freq, est_phase)
		ref_vals_phaseShift = refValue_phaseShift(even_time, est_freq, est_phase)
	
	num_rows = len(signal)
	num_cols = len(signal[0])
	ref_vals = refValue(time, est_freq, est_phase)
	ref_vals_phaseShift = refValue_phaseShift(time, est_freq, est_phase)
	mixed = np.multiply(signal, np.array([ref_vals]).T) * 2 #The 2 is a scaling factor
	mixed_phaseShift = np.multiply(signal, np.array([ref_vals_phaseShift]).T) * 2 #The 2 is a scaling factor

	window = np.hanning(num_rows)
	mixed = mixed * window.reshape((window.size, 1))
	mixed_phaseShift = mixed_phaseShift * window.reshape((window.size, 1))
	if interpolate:
		return mixed, mixed_phaseShift, even_time
	else:
		return mixed, mixed_phaseShift, time



def fft_lowpass(data, cutoff, f_s, timesteps):
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
	timesteps : float
		Number of timesteps for signal data
	
	Returns
	-------
	filtered_signal : 1D array of floats
		Signal after being filtered. 
	"""
	n = len(data)
	fourier = rfft(data)

	frequencies = rfftfreq(timesteps) * f_s
	index_upper = int(cutoff * timesteps/f_s)
	mask = np.zeros(fourier.size)
	mask[range(index_upper + 1)] = 2
	fourier *= mask
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
		filteredColumn = fft_lowpass(data, cutoff, sample_rate, timeSteps)
		filteredColumn_phaseShift = fft_lowpass(data_phaseShift, cutoff, sample_rate, timeSteps)
		values = np.sqrt(np.power(np.absolute(filteredColumn), 2) + np.power(np.absolute(filteredColumn_phaseShift), 2))
		angles = np.arctan2(filteredColumn_phaseShift, filteredColumn)
		angle = np.mean(angles)
		value = np.mean(values)
		r.append(value)
		theta.append(angle)
	return r, theta

def split(sample_len, num_windows, window_size):  
	"""
	Returns a list of approximate indices to split an array into 
	'windows' windows where each window overlaps with
	each other window by a proportion given by 
	'overlap'. The sample_len is the length of the sample. 
	"""
	window_size = int(sample_len * window_size)
	if num_windows == 1:
		return [(0, window_size)]
	step_size = int((sample_len - window_size)/(num_windows - 1)) 
	leftover = leftover = sample_len - step_size * (num_windows-1) - window_size
	
	sizes = (window_size  + int(leftover/num_windows))* np.ones(num_windows)
	step_sizes = (step_size + int(leftover/num_windows)) * np.ones(num_windows)
	leftover = leftover % num_windows
	for i in range(leftover):
		sizes[i] += 1
		step_sizes[i + 1] += 1
	indices = []
	for w in range(num_windows):
		if w == 0:
			start = 0
			end = sizes[0]
			prev = (start, end)
		else:
			start = prev[0] + step_sizes[w]
			end = start + sizes[w]
			prev = (start, end)
		indices.append((int(start), int(end)))

	return indices

def lock_in(signal, time, est_freq, est_phase, cutoff, num_windows, window_size, interpolate):

	"""
	Applies lock-in to the data by performing the signal mixing and 
	calling the lowpass filter. Also splits the data to get errorbars
	as specified by the window and overlap parameters.
	
	Parameters
	----------
	signal : 2D array of floats
		Intensity values for each channel over time.
	time : 1D array of floats
		Timestamps for the data
	est_freq : float
		Estimated frequency of the reference signal.
	est_phase : float
		Estimated phase of the reference signal
	cutoff : float
		Cutoff frequency for lowpass filter.
	num_windows : int
		Number of windows to split the data into
	window_size : float
		Value between 0 and 1, the size of each window as a 
		percentage of total input size.

	Returns
	-------
	magnitudes : 1D array of floats
		Lock-in output magnitudes for each channel
	phases : 1D array of floats
		Lock-in output phases for each channel
	mag_errors : 1D array of floats
		Standard deviation for each Lock-in magnitude output
	phase_errors : 1D array of floats
		Standard deviation for each Lock-in phase output
	indices : 1D array of tuples
		List of indices for the window splitting
	"""

	if num_windows == 1:
		#Splitting here in case user wanted to throw away some of the data
		indices = split(len(signal), num_windows, window_size)
		signal = signal[indices[0][0]:indices[0][1]]
		time = time[indices[0][0]:indices[0][1]]
		#mixing the signal
		mixed, mixed_phaseShift, even_time = mix(signal, time, est_freq, est_phase, interpolate)
		#Applying the lowpass filter
		magnitudes, phases = apply_lowpass(mixed, mixed_phaseShift, even_time, cutoff)
		return magnitudes, phases, 0, 0, indices
	
	print("Splitting Input...", flush = True)
	indices = split(len(signal), num_windows, window_size)
	mags_list = []
	phases_list = []
	for index in indices:
		tmpSig = signal[index[0] : index[1]]
		tmpTime = time[index[0] : index[1]]
		#Mixes the intensity signal with the normal and phase shifted reference signals.
		mixed, mixed_phaseShift, even_time = mix(tmpSig, tmpTime, est_freq, est_phase, interpolate)
		#Applies lowpass filter
		tmpMags, tmpPhases  = apply_lowpass(mixed, mixed_phaseShift, even_time, cutoff)
		mags_list.append(np.asarray(tmpMags))
		phases_list.append(np.asarray(tmpPhases))

	magnitudes, var_mags = sp.describe(mags_list)[2:4]
	mag_errors = np.sqrt(var_mags)
	phases, var_phases = sp.describe(phases_list)[2:4]
	phase_errors = np.sqrt(var_phases)

	return magnitudes, phases, mag_errors, phase_errors, indices



# No fit functions

def mix_no_fit(signal, sig_time, reference, ref_time, interpolate):
	"""
	Performs the signal mixing step of a lock in amplifier.
	Mixes, or multiplies, the intensity signal for all channels
	by the provided reference signal. 
	Also performs cubic interpolation on the input so the mixed signal has
	consistent timesteps and applies the Hanning window to 
	mitigate sidelobes. This version cannot compute phase since phase is 
	essentially meaningless for an arbitrarily complex reference input. 
	Parameters
	----------
	signal : 2D array of floats
		Intensity values for each channel over time.
	sig_time : 1D array of floats
		Timestamps for the data
	reference : 1D array of floats
		Reference signal over time. 
	ref_time : 1D array of floats
		Timestamps for reference signal
	Returns
	-------
	mixed : 2D array of floats
		signal multiplied by reference signal.
		Each row is a set of mixed values for each channel with a timestamp.
	"""
	print("Mixing...", flush = True)
	if interpolate:
		interpolated_sig = scipy.interpolate.interp1d(sig_time, signal, bounds_error=False,
		 kind='cubic', axis = 0, fill_value = "extrapolate")
		interpolated_ref = scipy.interpolate.interp1d(ref_time, reference, bounds_error=False,
		 kind='cubic', axis = 0, fill_value = "extrapolate")
		min_time = min(sig_time)
		max_time = max(sig_time)
		len_time = len(sig_time)
		timestep = (max(sig_time) - min(sig_time))/len(sig_time)
		even_time = np.arange(min_time, max_time, timestep)
		signal = interpolated_sig(even_time)
		reference = interpolated_ref(even_time)
	num_rows = len(signal)
	num_cols = len(signal[0])
	mixed = np.multiply(signal, np.array([reference]).T) * 2 #The 2 is a scaling factor.

	window = np.hanning(num_rows)
	mixed = mixed * window.reshape((window.size, 1))
	if interpolate:
		return mixed, even_time
	else:
		return mixed, sig_time

def apply_lowpass_no_fit(mixed, time, cutoff):
	"""
	Applies lowpass filter to the mixed signal to get cartesian lock in values
	for each measured channel.
	Parameters
	----------
	mixed : 2D array of floats
		signal multiplied by reference signal.
		Each row is a set of mixed values for each channel. 
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
		filteredColumn = fft_lowpass(data, cutoff, sample_rate, timeSteps)
		values = np.absolute(filteredColumn)
		value = np.mean(values)
		r.append(value)
	return r

def lock_in_no_fit(signal, sig_time, reference, ref_time, cutoff, num_windows, window_size, interpolate):

	"""
	Applies lock-in to the data by performing the signal mixing and 
	calling the lowpass filter. Also can split the data to get errorbars
	as specified by the window and overlap parameters.
	
	Parameters
	----------
	signal : 2D array of floats
		Intensity values for each channel over time.
	sig_time : 1D array of floats
		Timestamps for the signal data
	reference : 1D array of floats
		Reference signal over time. 
	ref_time : 1D array of floats
		Timestamps for reference signal
	cutoff : float
		Cutoff frequency for lowpass filter.
	num_windows : int
		Number of windows to split the data into
	window_size : float
		Value between 0 and 1, the size of each window as a 
		percentage of total input size.

	Returns
	-------
	magnitudes : 1D array of floats
		Lock-in output magnitudes for each channel
	mag_errors : 1D array of floats
		Standard deviation for each Lock-in magnitude output
	indices : 1D array of tuples
		List of indices for the window splitting
	"""

	if num_windows == 1:
		#Splitting here in case user wanted to throw away some of the data
		indices = split(len(signal), num_windows, window_size)
		signal = signal[indices[0][0]:indices[0][1]]
		sig_time = sig_time[indices[0][0]:indices[0][1]]
		reference = reference[indices[0][0]:indices[0][1]]
		ref_time = ref_time[indices[0][0]:indices[0][1]]
		#mixing the signal
		mixed, even_time = mix_no_fit(signal, sig_time, reference, ref_time, interpolate)
		#Applying the lowpass filter
		magnitudes = apply_lowpass_no_fit(mixed, even_time, cutoff)
		return magnitudes, 0, indices
	
	print("Splitting Input...", flush = True)
	indices = split(len(signal), num_windows, window_size)
	mags_list = []
	phases_list = []
	for index in indices:
		tmpSig = signal[index[0] : index[1]]
		tmpTime = sig_time[index[0] : index[1]]
		tmpRef = reference[index[0] : index[1]]
		tmpRefTime = ref_time[index[0] : index[1]]
		#Mixes the intensity signal with the normal and phase shifted reference signals.
		mixed, even_time = mix_no_fit(tmpSig, tmpTime, tmpRef, tmpRefTime, interpolate)
		#Applies lowpass filter
		tmpMags = apply_lowpass_no_fit(mixed, even_time, cutoff)
		mags_list.append(np.asarray(tmpMags))

	magnitudes, var_mags = sp.describe(mags_list)[2:4]
	mag_errors = np.sqrt(var_mags)

	return magnitudes, mag_errors, indices