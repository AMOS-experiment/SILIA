import numpy as np
from scipy.signal import square
import csv

def noise(noise_amplitude, noise_type):
	if noise_type == 'None':
		return 0
	elif noise_type == 'Gaussian':
		return np.random.normal(0, noise_amplitude)
	elif noise_type == 'Uniform':
		return np.random.uniform(-1 * noise_amplitude, noise_amplitude)
def references(frequencies, time, refFreq):

	"""Returns the reference signals for the at the different frequencies that should be analyzed"""
	references = []
	for frequency in frequencies:
		references += [square(2 * np.pi * frequency * time)]
	return references
		

def refFreq():
	"""Sampling rate of reference signal"""
	return 1000

def channels():
	"""Returns the channels being read by the lock in"""
	return np.arange(0, 100, 1)

def signal(frequencies, channels, time):
	"""Returns signal with noise at the frequencies"""
	signal = []
	for t in time:
		row = [t]
		i =0
		while i < len(channels):
			if i < len(channels)/5:
				row += [noise(1, 'Gaussian')]
			elif i >= len(channels)/5 and i < 2 * len(channels)/5:
				row += [np.sin(2 * np.pi * frequencies[0] * t) + 10 * i]
			elif i >= 2 * len(channels)/5 and i < 3 * len(channels)/5:
				row += [noise(1, 'Gaussian')]
			elif i >= 3 * len(channels)/5 and i < 4 * len(channels)/5:
				row += [np.sin(2 * np.pi * frequencies[1] * t) + i * noise(1, 'Gaussian')]
			else:
				row += [noise(1, 'Gaussian')]
			i += 1
		signal += [row] 
	return signal