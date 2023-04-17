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
    wavelet_width = (1/0.66**2) * fs / (src_freq**2) # (Note squared as Ricker is second derivitive of Gaussian)
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

    # And rotate into ZNE:
    ZNE_st = LQT_st.rotate('LQT->ZNE', back_azimuth=src_pol_from_N, inclination=src_pol_from_up)
    del LQT_st
    gc.collect()

    return ZNE_st


def _convert_phi_from_NZ_to_Q_coords(back_azi, phi_from_N):
    """Function to convert phi and all assocated data structures from phi relative to clockwise 
    from N,U to phi clockwise Q.
    Returns phi_from_Q. All angles are in degrees."""
    # Calculate horizontal angle from N:
    phi_from_Q = phi_from_N - back_azi 
    if phi_from_Q < 0:
        phi_from_Q = phi_from_Q + 360 # Convert to 0-360 deg.
    if phi_from_Q > 90:
        phi_from_Q = phi_from_Q - 180 # Convert to -90 to 90 deg
        if phi_from_Q > 90:
            phi_from_Q = phi_from_Q - 180 # Convert to -90 to 90 deg again if needed
    return phi_from_Q


def add_splitting(ZNE_st, phi_from_N, dt, back_azi, event_inclin_angle_at_station, snr=None):
    """
    Function to add splitting to waveforms in ZNE notation.
    
        Parameters
    ----------
    ZNE_st : obspy stream
        Obspy stream with Z, N and E traces.

    phi_from_N : float
        Angle of fast direction, in degrees from N.

    dt : float
        Delay time between fast and slow S-waves, in seconds.

    back_azi : float
        Back-azimuth from source to receiver, in degrees from N.

    event_inclin_angle_at_station : float
        Inclination of ray at station, in degrees from vertical up.

    phi_from_up : float
        Angle of fast direction, in degrees from vertical up. Optional. Default = 0 degrees.
    
    snr : float
        Signal-to-noise-ratio of output source-time function. Optional. If not None, then will 
        add white gaussian noise to data.

    """
    # Perform initial checks:
    if len(ZNE_st.select(channel="??N")) != 1:
        print("Error: st_LQT_uncorr has more or less than 1 N component. Exiting.")
        sys.exit()
    if len(ZNE_st.select(channel="??E")) != 1:
        print("Error: st_LQT_uncorr has more or less than 1 E component. Exiting.")
        sys.exit()

    # Calculate phi from Q:
    phi_from_Q = _convert_phi_from_NZ_to_Q_coords(back_azi, phi_from_N)
    
    # Add splitting:
    # Rotate ZNE stream into LQT then BPA propagation coords:
    LQT_st = split._rotate_ZNE_to_LQT(ZNE_st, back_azi, event_inclin_angle_at_station)
    st_BPA = split._rotate_LQT_to_BPA(LQT_st, 0)
    # Apply SWS:
    x_in, y_in = st_BPA.select(channel="??P")[0].data, st_BPA.select(channel="??A")[0].data
    fs = st_BPA.select(channel="??A")[0].stats.sampling_rate
    # 1. Rotate data into splitting coordinates:
    x, y = split._rotate_QT_comps(x_in, y_in, phi_from_Q * np.pi/180)
    # And write fast and slow directions out as F and S channels:
    chan_prefixes = ZNE_st.select(channel="??Z")[0].stats.channel[0:2]
    st_BPA_with_splitting = st_BPA.copy()
    # 2. Apply reverse time shift to slow data:
    y = np.roll(y, int((dt) * fs))
    # 3. And rotate back to QT (PA) coordinates:
    x, y = split._rotate_QT_comps(x, y, -phi_from_Q * np.pi/180)
    # And put data back in stream form:
    st_BPA_with_splitting.select(channel="??P")[0].data = x 
    st_BPA_with_splitting.select(channel="??A")[0].data = y
    # And rotate back into ZNE coords:
    st_LQT_with_splitting = split._rotate_BPA_to_LQT(st_BPA_with_splitting.copy(), 0)
    ZNE_st_with_splitting = split._rotate_LQT_to_ZNE(st_LQT_with_splitting, back_azi, event_inclin_angle_at_station)

    # And tidy:
    del LQT_st, st_BPA, st_BPA_with_splitting, st_LQT_with_splitting
    gc.collect()

    # Add Gaussian noise to data, if specified:
    # (white noise)
    if snr:
        sig_max = np.max( np.array( [np.max(np.abs(ZNE_st_with_splitting[0].data)), np.max(np.abs(ZNE_st_with_splitting[1].data)), 
                                                                        np.max(np.abs(ZNE_st_with_splitting[2].data))] ) )
        for i in range(len(ZNE_st_with_splitting)):
            noise = np.random.normal(0, sig_max / snr, len(ZNE_st_with_splitting[i].data)) # (mu, sigma, size)
            ZNE_st_with_splitting[i].data += noise
        del noise 
        gc.collect()

    return ZNE_st_with_splitting

    
