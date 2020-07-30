import numpy as np
import os
import reference_signal as rf
from numpy.fft import fft, ifft
import matplotlib.pyplot as plt
import simulate_signal
import csv


"""
Cutoff frequency for low pass filter (Hz)
"""
cutoff = 0

"""
Frequencies to lock into
"""
frequencies = [75, 125]
"""
Sampling rate of the reference signal DaQ
"""
refFreq = simulate_signal.refFreq()

"""
The relative path to the directory where the data is being stored.
"""
data_dir = '../Lock-in_sim_results/Two-Freq/new/'

"""
1D Array of floats representing each channel.
"""
channels =simulate_signal.channels()


"""
2D Array of timestamp and intensities for each channel.
Each row is a set of intensity values for a timestamp.
"""
times = np.arange(0, 1.024 * 8, 0.001)
intensities = simulate_signal.signal(frequencies, channels, times)
os.mkdir(data_dir)
with open(data_dir + 'intensities.csv', 'w', newline = '') as csvfile:
    writer = csv.writer(csvfile, delimiter = ',')
    writer.writerows(intensities)
print("Saved Intensities")
with open(data_dir + 'wavelengths.csv', 'w', newline = '') as csvfile:
    writer = csv.writer(csvfile, delimiter = ',')
    writer.writerow(channels)


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

def mix(intensities, est_freq, est_phase):
	"""
	Performs the signal mixing step of a lock in amplifier.
	Mixes, or multiplies, the intensity signal for all channels
	by the fitted reference signal as well as its pi/2 phase shift. 
	Parameters
	----------
	intensities : 2D array of floats
		Timestamp and intensity values for each channel.
		Each row is a set of intensity values for each channel with a timestamp.
	est_freq : float
		Estimated frequency of the reference signal.
	est_phase : float
		Estimated phase of the reference signal.
	Returns
	-------
	mixed : 2D array of floats
		Intensities signals multiplied by reference signal.
		Each row is a set of mixed values for each channel with a timestamp.
	mixed_phaseShift : 2D array of floats
		Intensities signals multiplied by phase shifted reference signal.
		Same formatting as mixed. 
	"""
	#Shifts intensity values so each signal is centered at 0.
	k = 1
	intensities = np.array(intensities)
	while (k < len(intensities[0])):
		avg = np.mean(intensities[:, k])
		j = 0
		while (j < len(intensities)):
			intensities[j][k] -= avg
			j += 1
		k += 1
	i = 0
	mixed = []
	mixed_phaseShift = []
	while (i < len(intensities)):
		j = 1
		timeStamp = intensities[i][0] #timestamp in seconds

		ref_value = refValue(timeStamp, est_freq, est_phase)
		ref_value_phaseShift = refValue_phaseShift(timeStamp, est_freq, est_phase)

		mixed.append([timeStamp])
		mixed_phaseShift.append([timeStamp])
		while (j < len(intensities[i])):
			mixed[i].append(ref_value * (intensities[i][j]))
			mixed_phaseShift[i].append(ref_value_phaseShift * (intensities[i][j]))
			j += 1
		i += 1

	mixed = np.asarray(mixed)
	mixed_phaseShift = np.asarray(mixed_phaseShift)
	print("Finished Mixing")

	return mixed, mixed_phaseShift

def fft_lowpass(data, cutoff, f_s, time):
	"""
	Lowpass filter using the numpy fft algorithm.
	Used to filter the mixed signal for single channels.
	
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
	fourier = fft(data)
	frequencies = np.fft.fftfreq(len(time)) * f_s
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
	filtered_signal = ifft(fourier)
	return filtered_signal


def apply_lowpass(mixed, mixed_phaseShift, time, cutoff):
	"""
	Applies lowpass filter to the mixed signals to get cartesian lock in values
	for each measured channel.
	Parameters
	----------
	mixed : 2D array of floats
		Intensities signals multiplied by reference signal.
		Each row is a set of mixed values for eachw channel with a timestamp.
	mixed_phaseShift : 2D array of floats
		Intensities signals multiplied by phase shifted reference signal.
		Same formatting as mixed. 
	time : 1D array of floats
		Timestamps in milliseconds for each signal measurement.
	cutoff : float
		Cutoff frequency for lowpass filter.
	"""

	timeSteps = len(time)
	totalTime = time[timeSteps - 1] - time[0]
	timePerSample = totalTime/timeSteps

	sample_rate = 1/timePerSample
	num_channels = len(mixed[0])

	i = 1
	r = []
	theta = []
	while (i < num_channels):
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
		i += 1
	return r, theta


def polarOutput(values, values_phaseShift, channels, times, freq):
	"""
	Converts lock in output to polar and writes it to csv 
	"lock_in_values_r.csv" and "lock_in_values_theta.csv".
	Writes the first row as the channels being measured if the csv is empty.
	Adds a row of lock in values to the .csv files as: 
	chunk timeStamp, r0, r1... \n, and the same for theta.
	Parameters
	----------
	values : 1D array
		An array of lock in values for each channel.
	values_phaseShift : 1D array
		An array of the phase shifted lock in values for each channel. 
	channels : 1D array
		A list of the channels being measured.
	times : 1D array
		The timestamps for each signal measurement.
	"""
	file1 = open(data_dir + "lock_in_values_r.csv", "a+")
	file2 = open(data_dir + "lock_in_values_theta.csv", "a+")
	if (os.stat(data_dir + "lock_in_values_r.csv").st_size == 0):
		for channel in channels:
			file1.write("," + str(channel))
			file2.write("," + str(channel))
		file1.write("\n")
		file2.write("\n")
	file1.write(str(freq) + ",")
	file2.write(str(freq) + ",")
	n = len(values)
	i = 0
	while (i < n):
		x = values[i]
		y = values_phaseShift[i]
		r = np.sqrt(x**2 + y**2)
		theta = np.arctan2(y, x)
		file1.write(str(r))
		file2.write(str(theta))
		i += 1
		if (i < n):
			file1.write(",")
			file2.write(",")

	file1.write("\n")
	file2.write("\n")
	file1.close()
	file2.close()



def main():
	"""
	Performs simultaneous lock-in on a chunk of data.
	"""

	#Fits the reference signal to a sine wave.

	ref_vals = rf.setUp(times, frequencies, refFreq)

	magnitudes = []
	angles = []
	for fit_params in ref_vals:
		est_freq, est_phase, est_offset, est_amp = fit_params[0], fit_params[1], fit_params[2], fit_params[3]

		#Mixes the intensity signal with the normal and phase shifted reference signals.
		mixed, mixed_phaseShift = mix(intensities, est_freq, est_phase)

		#Timestamps in seconds
		time = mixed[:,0]

		curr_magnitudes, curr_angles = apply_lowpass(mixed, mixed_phaseShift, time, cutoff)
		magnitudes.append(curr_magnitudes)
		angles.append(curr_angles)

	i = 0
	while i < len(filtered):
		freq = frequencies[i]
		polarOutput(magnitudes[i], angles[i], channels, time, freq)
		i += 1

if __name__ == "__main__":
	main()