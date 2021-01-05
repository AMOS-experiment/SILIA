import pytest
import numpy as np
import numpy.testing as nptest
import scipy.signal
from .reference_signal import *


def test_sin_fitting():

	#Testing fit for a sine reference
	time = np.arange(0, 1, 1/1000)
	freq = 100
	reference = [{'time' : time, 'signal' : 10 * np.sin(2 * np.pi * freq * time - 3 * np.pi/4) + 5}]
	fit_vals = fit(reference)[0]
	nptest.assert_allclose(fit_vals, [freq, -3 * np.pi/4, 5, 10])



def test_square_fitting():
	#Testing fit for a square reference
	time = np.arange(0, 1, 1/1000)
	freq = 75
	reference = [{'time' : time, 'signal' : 2 * scipy.signal.square(2 * np.pi * freq * time + np.pi/4)}]
	fit_vals = fit(reference)[0]
	nptest.assert_allclose(fit_vals[:3], [freq, np.pi/4, 0], atol = .1)


def test_triangle_fitting():
	#Testing fit for a triangle wave reference
	time = np.arange(0, 1, 1/1000)
	freq = 55
	reference = [{'time' : time, 'signal' : 2 * scipy.signal.square(2 * np.pi * freq * time + np.pi/2)}]
	fit_vals = fit(reference)[0]
	nptest.assert_allclose(fit_vals[:3], [freq, np.pi/2, 0], atol = .1)

def test_multiple_fits():
	#Testing fit for multiple references
	time = np.arange(0, 1, 1/2000)
	frequencies = np.arange(10, 200, 5)
	references = []
	for freq in frequencies:
		references.append({'time' : time, 'signal' : 2 * np.sin(2 * np.pi * freq * time)})
	fit_vals = np.asarray(fit(references))
	fitted_frequencies = fit_vals[:, 0]
	fitted_phases = fit_vals[:, 1]
	nptest.assert_allclose(fitted_frequencies, frequencies, rtol = .1)

