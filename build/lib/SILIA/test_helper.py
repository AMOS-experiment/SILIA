import pytest
import numpy as np
import numpy.testing as nptest
from .helper import *
import scipy.signal


def test_find_nearest():
	#Testing function to find index of closest value in array
	assert find_nearest(np.arange(0, 3, 1), .6) == 1
	assert find_nearest(np.arange(0, 3, .1), .6) == 6


def test_mix_no_interp():
	#Testing the mixing of signal with reference without interpolation
	time = np.arange(0, 1, 1/2000)
	signal = np.ones((time.size, 1))
	freq = 100
	phase = np.pi/4
	mixed = mix(signal, time, freq, phase, interpolate = False, pbar = False)
	window = np.hanning(time.size)

	nptest.assert_allclose(mixed[0], 2 * np.transpose([np.sin(2 * np.pi * freq * time + phase)])\
	 * window.reshape((window.size, 1)))


def test_mix_interp():
	#Testing the mixing of signal with reference with interpolation
	time = np.arange(0, 1, 1/2000)
	even_time = np.arange(min(time), max(time), (max(time) - min(time))/len(time))
	signal_input = np.ones((time.size, 1))
	freq = 100
	phase = np.pi/4
	mixed = mix(signal_input, time, freq, phase, interpolate = True, pbar = False)
	window = np.hanning(even_time.size)

	nptest.assert_allclose(mixed[0], 2 * np.transpose([np.sin(2 * np.pi * freq * even_time + phase)])\
	 * window.reshape((window.size, 1)))



def test_fft_lowpass():
	#Testing the lowpass filter
	time = np.arange(0, 1, 1/2000)
	signal = np.sin(2 * np.pi * 100 * time)
	filtered = fft_lowpass(signal, cutoff = 0, f_s = 2000, timesteps = time.size)
	nptest.assert_allclose(filtered, np.zeros(time.size), atol = 10**(-5))

def test_split():
	#Testing function to split signal into windows
	assert split(100, 5, .2) == [(0, 20), (20, 40), (40, 60), (60, 80), (80, 100)]
	assert split(100, 3, .5) == [(0, 50), (25, 75), (50, 100)]


def test_mix_no_fit_no_interp():
	#Testing mixing function without fitting reference, no interpolation
	time = np.arange(0, 1, 1/2000)
	signal = np.ones((time.size, 1))
	freq = 100
	phase = np.pi/4
	reference = scipy.signal.square(2 * np.pi * freq * time + phase)
	mixed = mix_no_fit(signal, time, reference, time, interpolate = False, pbar = False)
	window = np.hanning(time.size)

	nptest.assert_allclose(mixed[0], 2 * np.transpose([reference])\
	 * window.reshape((window.size, 1)))
