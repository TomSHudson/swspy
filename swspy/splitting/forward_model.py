#-----------------------------------------------------------------------------------------------------------------------------------------

# Script Description:

# Input variables:

# Output variables:

# Created by Tom Hudson, 4th April 2022

#-----------------------------------------------------------------------------------------------------------------------------------------

# Import neccessary modules:
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy import signal, interpolate
import obspy
import sys, os
import gc
from swspy.splitting import split 

class CustomError(Exception):
    pass


def create_src_time_func(dur, fs, src_pol_from_N=0, src_pol_from_up=0, src_freq=10., wavelet="ricker", t_src=1.):
    """
    Function to create synthetic source time function.
    
        Parameters
    ----------
    dur : float
        Duration of trace, in seconds.

    fs : float
        Sampling rate, in Hz.

    src_pol_from_N : float
        Source polarisation from North, in degrees. Optional. Default = 0 degrees.
    
    src_pol_from_up : float
        Source polarisation from vertical up, in degrees. Optional. Default = 0 degrees.

    src_freq : float
        Dominant source frequency, in Hz. Optional. Default is 10 Hz. 

    wavelet : str
        Type of wavelet to use for the source-time function. Optional. Default is ricker.
        Other options not currently implemented.

    t_src : float
        Time, in seconds, from the beginning of the trace when the source-time function peaks.
        Optional. Default = 1 s.
    """
    # Perform some initial checks:
    if wavelet != "ricker":
        print("Error: wavelet = "+wavelet+"not supported. Exiting")
        sys.exit()
    
    # Create signal:
    n_samp = int(dur*fs)
    wavelet_width = (1/0.66) * fs / src_freq
    t_shift_from_centre_samps = -1 * int( (n_samp/2) - (t_src*fs) )
    sig = np.roll(signal.ricker(n_samp, wavelet_width), t_shift_from_centre_samps)

    # Create signal in LQT coordinates:
    # (assumes source polarisation in Q-axis (i.e. bazi = src pol arbitarily here for rotation))
    LQT_st = obspy.Stream()
    # L:
    tr = obspy.Trace()
    tr.stats.network = "00"
    tr.stats.station = "synth"
    tr.stats.channel = "HHL"
    tr.stats.sampling_rate = fs 
    tr.data = np.zeros(n_samp)
    LQT_st.append(tr)
    # Q:
    tr = obspy.Trace()
    tr.stats.network = "00"
    tr.stats.station = "synth"
    tr.stats.channel = "HHQ"
    tr.stats.sampling_rate = fs 
    tr.data = sig
    LQT_st.append(tr)
    # T:
    tr = obspy.Trace()
    tr.stats.network = "00"
    tr.stats.station = "synth"
    tr.stats.channel = "HHT"
    tr.stats.sampling_rate = fs
    tr.data = np.zeros(n_samp) 
    LQT_st.append(tr)

    return LQT_st




    
