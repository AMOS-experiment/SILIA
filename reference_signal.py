import numpy as np
from scipy.optimize import leastsq
import os
import simulate_signal
from scipy import square


def setUp(time, frequencies, refFreq):

    """
    Fits the measured reference signal to a sine wave and returns
    the fit parameters. Starts by guessing the fit parameters, 
    then uses least squares optimization.
    Parameters
    ----------
    startTime : float   
        Timestamp at which measurement of the reference signal began.
    refFreq : float
        Sampling rate of the reference signal.
    data_dir : string
        Relative path to the folder containing the measured reference signal in
        "RefData.csv". 
    """
    references = simulate_signal.references(frequencies, time, refFreq)
    samplingRate =  refFreq  
    ref_values = []

    for rawInput in references:
        N = len(rawInput)
        #time = np.linspace(startTime, startTime + (N/samplingRate) * 1000, N) 
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
        ref_values += [[est_freq, est_phase, est_offset, est_amp]]
    return ref_values