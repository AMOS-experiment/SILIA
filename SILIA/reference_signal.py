import numpy as np
from scipy.optimize import leastsq
import os
from scipy import square

def fit(references):

    """
    Fits the measured reference signal to a sine wave and returns
    the fit parameters. Starts by guessing the fit parameters, 
    then uses least squares optimization.
    Parameters
    ----------
    references : array of dictionary 
        Array of reference signals, where each reference signal is a dictionary
        which consists of an array of timestamps labeled by 'time' and an array 
        of signal values labeled by 'signal'.
    """

    ref_values = []

    for ref in references:
        rawInput = ref['signal']
        time = ref['time']
        N = len(rawInput)
        samplingRate = len(time)/(max(time) - min(time))
        maxValue = max(rawInput)
        minValue = min(rawInput)
        guess_offset = np.mean(rawInput)

        #Guesses amplitude of the reference signal.
        greater_than_offset = []
        less_than_offset = []
        for value in rawInput:
            if value > guess_offset:
                greater_than_offset += [value]
            else:
                less_than_offset += [value]
        guess_amp = (np.mean(greater_than_offset) - np.mean(less_than_offset))/2


        #Guesses frequency of the reference signal.
        num_switches = 0
        length = len(rawInput)
        try:
            i = int(N/2)
            startingSign = (rawInput[i] - guess_offset > 0)
            while ((rawInput[i] - guess_offset > 0) == startingSign):
                i += 1
            startIndex = i
            prev_sign = (rawInput[i] - guess_offset > 0)
            endIndex = i
            while i < length:
                if (rawInput[i] - guess_offset > 0) != prev_sign:
                    prev_sign = (rawInput[i] - guess_offset > 0)
                    num_switches += 1
                    endIndex = i
                i += 1
        except IndexError as e:
            print("Bad Reference Signal (Either too few cycles, or no clear oscillations)")
            raise e

        guess_freq = 0.5 * num_switches * samplingRate/(endIndex - startIndex)

        #Guesses phase of the reference signal.
        try:
            i = 0
            while ((rawInput[i] - guess_offset > 0) == False):
                i += 1
            phaseIndex = i
        except IndexError as e:
            print("Bad reference signal")
            raise e
        guess_phase = np.pi*(phaseIndex * num_switches/(endIndex-startIndex))

        optimize_func = lambda x: x[3]*np.sin(x[0]*time*2*np.pi + x[1]) + x[2] - rawInput
        est_freq, est_phase, est_offset, est_amp = leastsq(optimize_func, [guess_freq, guess_phase, guess_offset, guess_amp])[0]
        est_phase = est_phase % (2 * np.pi)
        if est_phase > np.pi:
        	est_phase -= 2 * np.pi
        ref_values += [[est_freq, est_phase, est_offset, est_amp]]
    return ref_values