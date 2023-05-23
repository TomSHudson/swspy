#-----------------------------------------------------------------------------------------------------------------------------------------

# Script Description:

# Input variables:

# Output variables:

# Created by Tom Hudson, 30th July 2021

#-----------------------------------------------------------------------------------------------------------------------------------------

# Import neccessary modules:
import swspy
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import pandas as pd 
import numba 
from numba import jit, njit, types, set_num_threads, prange, float64, int64
from scipy import stats, interpolate
from sklearn import cluster
from sklearn.cluster import DBSCAN, KMeans, AgglomerativeClustering
import obspy
from obspy import UTCDateTime as UTCDateTime
import sys, os
import glob
import subprocess
import gc
import time 


class CustomError(Exception):
    pass


def find_nearest(array,value):
    idx = (np.abs(array-value)).argmin()
    return array[idx], idx


def _rotate_ZNE_to_LQT(st_ZNE, back_azi, event_inclin_angle_at_station):
    """Function to rotate ZRT traces into LQT and save to outdir.
    Requires: tr_z,tr_r,tr_t - traces for z,r,t components; back_azi - back azimuth angle from reciever to event in degrees from north; 
    event_inclin_angle_at_station - inclination angle of arrival at receiver, in degrees from vertical down."""
    # Rotate to LQT:
    st_LQT = st_ZNE.copy()
    st_LQT.rotate(method='ZNE->LQT', back_azimuth=back_azi, inclination=event_inclin_angle_at_station)
    return st_LQT


def _rotate_LQT_to_BPA(st_LQT, back_azi):
    """Function to rotate LQT coords into BPA propagation coords, as in Walsh et al. (2013). Requires: st_LQT - stream with traces for 
    z,r,t components; back_azi - back azimuth angle from reciever to event in degrees from north.
    Note: L is equivient to B. 
    Note: Only works for a single station, as only supply a single back azimuth.
    Note: Artifically orients P to North. Actual source polarisation is found later in the script.
    """
    # Rotate LQT to BPA:
    st_BPA = st_LQT.copy()
    # And rename channels:
    chan_prefixes = st_LQT.select(channel="??L")[0].stats.channel[0:2]
    st_BPA.select(channel="??L")[0].stats.channel = ''.join((chan_prefixes,"B"))
    st_BPA.select(channel="??Q")[0].stats.channel = ''.join((chan_prefixes,"P"))
    st_BPA.select(channel="??T")[0].stats.channel = ''.join((chan_prefixes,"A"))
    # Convert back-azimuth to radians:
    back_azi_rad = back_azi * np.pi / 180.
    ###???# Rotate clockwise rather than anti-clockwise:
    ###???back_azi_rad = -back_azi_rad
    # Define rotation matrix (for counter-clockwise rotation):
    rot_matrix = np.array([[np.cos(back_azi_rad), -np.sin(back_azi_rad)], [np.sin(back_azi_rad), np.cos(back_azi_rad)]])
    # And perform the rotation (y = Q, P in this case, x = T, A in this case):
    try:
        vec = np.vstack((st_LQT.select(channel="??T")[0].data, st_LQT.select(channel="??Q")[0].data))
    except IndexError:
        raise ValueError("Q and/or T component in <st_LQT> doesn't exist.")
    vec_rot = np.dot(rot_matrix, vec)
    # And write out data:
    st_BPA.select(channel="??A")[0].data = np.array(vec_rot[0,:])
    st_BPA.select(channel="??P")[0].data = np.array(vec_rot[1,:])
    return st_BPA


def _rot_phi_by_90_deg(phi):
    """Rotate phi by cyclic 90 degrees."""
    phi_rot = phi+90.
    if phi_rot > 90.:
        phi_rot = -90. + (phi_rot - 90.)
    return phi_rot


def _rotate_LQT_to_ZNE(st_LQT, back_azi, event_inclin_angle_at_station):
    """Function to rotate LQT traces into ZNE and save to outdir.
    Requires: tr_z,tr_r,tr_t - traces for z,r,t components; back_azi - back azimuth angle from reciever to event in degrees from north; 
    event_inclin_angle_at_station - inclination angle of arrival at receiver, in degrees from vertical down."""
    # Rotate to ZNE:
    st_ZNE = st_LQT.copy()
    st_ZNE.rotate(method='LQT->ZNE', back_azimuth=back_azi, inclination=event_inclin_angle_at_station)
    return st_ZNE


def _rotate_BPA_to_LQT(st_BPA, back_azi):
    """Function to rotate BPA propagation coords into LQT coords, as in Walsh et al. (2013). Requires: st_LQT - stream with traces for 
    z,r,t components; back_azi - back azimuth angle from reciever to event in degrees from north.
    Note: L is equivient to B. 
    Note: Only works for a single station, as only supply a single back azimuth."""
    # reasign channel labels:
    chan_prefixes = st_BPA.select(channel="??B")[0].stats.channel[0:2]
    st_BPA.select(channel="??B")[0].stats.channel = ''.join((chan_prefixes,"L"))
    st_BPA.select(channel="??P")[0].stats.channel = ''.join((chan_prefixes,"Q"))
    st_BPA.select(channel="??A")[0].stats.channel = ''.join((chan_prefixes,"T"))
    # Rotate BPA to LQT (just inverse angle of LQT -> BPA):
    st_LQT = _rotate_LQT_to_BPA(st_BPA, -back_azi)
    # And reasign channel labels again:
    chan_prefixes = st_LQT.select(channel="??B")[0].stats.channel[0:2]
    st_LQT.select(channel="??B")[0].stats.channel = ''.join((chan_prefixes,"L"))
    st_LQT.select(channel="??P")[0].stats.channel = ''.join((chan_prefixes,"Q"))
    st_LQT.select(channel="??A")[0].stats.channel = ''.join((chan_prefixes,"T"))
    return st_LQT


def _rotate_QT_comps(data_arr_Q, data_arr_T, rot_angle_rad):
    """
    Function to rotate Q (SV polarisation) and T (SH polarisation) components by a rotation angle <rot_angle_rad>,
    in radians. Output is a set of rotated traces in a rotated coordinate system about 
    degrees clockwise from Q.
    """
    # Convert angle from counter-clockwise from x to clockwise:
    theta_rot = -rot_angle_rad #-1. * (rot_angle_rad + (np.pi / 2. ))
    ###!!!theta_rot = -1. * (rot_angle_rad + (np.pi / 2. ))

    # Define rotation matrix:
    rot_matrix = np.array([[np.cos(theta_rot), -np.sin(theta_rot)], [np.sin(theta_rot), np.cos(theta_rot)]])
    # And perform the rotation:
    vec = np.vstack((data_arr_T, data_arr_Q))
    vec_rot = np.dot(rot_matrix, vec)
    data_arr_Q_rot = np.array(vec_rot[1,:])
    data_arr_T_rot = np.array(vec_rot[0,:])

    return data_arr_Q_rot, data_arr_T_rot


def _find_pol(x,y):
    """Function to return polarity of a 2-vector."""
    # 1. Calculate eigenvalues and vectors:
    xy_arr = np.vstack((x, y))
    lambdas_unsort, eigvecs_unsort = np.linalg.eig(np.cov(xy_arr))

    # 2. Find angle associated with max. eigenvector:
    max_eigvec = eigvecs_unsort[:,np.argmax(lambdas_unsort)]
    pol_deg = np.rad2deg( np.arctan2(max_eigvec[0], max_eigvec[1]) )

    # 3. And calculate approx. error in result:
    # (Based on ratio of orthogonal eigenvalue magnitudes)
    # (Approximately a maximum, hence the 180 degree term)
    pol_deg_err = 180. * ( np.min(lambdas_unsort) / np.max(lambdas_unsort) )

    return pol_deg, pol_deg_err


def _find_src_pol(x, y, z):
    """Function to find source polarisation angle relative to clockwise from y 
    and clockwise from z.
    Returns angles in degrees and associated errors."""
    # Find horizontal polarisation:
    h_src_pol_deg, h_src_pol_deg_err = _find_pol(x,y)
    # 4. And limit to be between 0 to 180:
    if h_src_pol_deg < 0:
        h_src_pol_deg = h_src_pol_deg + 180.
    elif h_src_pol_deg >= 180.:
        h_src_pol_deg = h_src_pol_deg - 180.

    # And find vertical polarisation:
    v_src_pol_deg, v_src_pol_deg_err = _find_pol(np.sqrt((x**2) + (y**2)),z)
    # 4. And limit to be between 0 to 180:
    if v_src_pol_deg < 0:
        v_src_pol_deg = v_src_pol_deg + 180.
    elif v_src_pol_deg >= 180.:
        v_src_pol_deg = v_src_pol_deg - 180.

    # And combine out:
    src_pol_deg = np.array([h_src_pol_deg, v_src_pol_deg])
    src_pol_deg_err = np.array([h_src_pol_deg_err, v_src_pol_deg_err])

    return src_pol_deg, src_pol_deg_err


def _get_ray_back_azi_and_inc_from_nonlinloc(nonlinloc_hyp_data, station):
    """Function to get ray back azimuth and inclination from nonlinloc hyp data object."""
    ray_back_azi = nonlinloc_hyp_data.phase_data[station]['S']['SAzim'] + 180.
    if ray_back_azi >= 360.:
        ray_back_azi = ray_back_azi - 360.
    ray_inc_at_station = nonlinloc_hyp_data.phase_data[station]['S']['RDip']
    return ray_back_azi, ray_inc_at_station


def remove_splitting(st_ZNE_uncorr, phi, dt, back_azi, event_inclin_angle_at_station, return_BPA=False,
                        src_pol=0., return_FS=True):
    """
    Function to remove SWS from ZNE data for a single station.
    Note: Consistency in this function with sws measurement. Uses T in x-direction and Q in y direction 
    convention.

    Parameters
    ----------
    st_LQT_uncorr : obspy stream object
        Stream data for station corresponding to splitting parameters.
    phi : float
        Splitting angle, for LQT coordinates.
    dt : float
        Fast-slow delay time (lag) for splitting.
    back_azi: float
        Back azimuth angle from reciever to event in degrees from North.
    event_inclin_angle_at_station : float
        Inclination angle of arrival at receiver, in degrees from vertical down.
    return_BPA : bool
        If True, will return obspy stream with Z,N,E and B,P,A channels (as in 
        Walsh (2013)). Optional. Default = False.
    src_pol : float
        If <return_BPA> = True, then uses src_pol to calculate the polarisiation and 
        null (P,A) vectors. Units are degrees clockwise from North.
    return_FS : bool
        If True, will return obspy stream with F (fast) and S (slow) channels 
        also included. Optional. Default = True.

    Returns
    -------
    st_ZNE_corr : obspy stream object
        Corrected data, in ZNE coordinates (unless <return_BPA> = True, then will 
        also output BPA channels too).
    """
    # Perform inital data checks:
    if len(st_ZNE_uncorr.select(channel="??N")) != 1:
        print("Error: st_LQT_uncorr has more or less than 1 N component. Exiting.")
        sys.exit()
    if len(st_ZNE_uncorr.select(channel="??E")) != 1:
        print("Error: st_LQT_uncorr has more or less than 1 E component. Exiting.")
        sys.exit()
    # Rotate ZNE stream into LQT then BPA propagation coords:
    st_LQT_uncorr = _rotate_ZNE_to_LQT(st_ZNE_uncorr, back_azi, event_inclin_angle_at_station)
    ###!!!st_BPA_uncorr = _rotate_LQT_to_BPA(st_LQT_uncorr, back_azi)
    st_BPA_uncorr = _rotate_LQT_to_BPA(st_LQT_uncorr, 0)
    # Perform SWS correction:
    x_in, y_in = st_BPA_uncorr.select(channel="??P")[0].data, st_BPA_uncorr.select(channel="??A")[0].data
    fs = st_BPA_uncorr.select(channel="??A")[0].stats.sampling_rate
    # 1. Rotate data into splitting coordinates:
    x, y = _rotate_QT_comps(x_in, y_in, phi * np.pi/180)
    # And write fast and slow directions out as F and S channels:
    chan_prefixes = st_ZNE_uncorr.select(channel="??Z")[0].stats.channel[0:2]
    st_BPA_corr = st_BPA_uncorr.copy()
    # For fast:
    tr_tmp = st_BPA_corr.select(channel="??P")[0].copy()
    tr_tmp.data = x
    tr_tmp.stats.channel = "".join((chan_prefixes, "F"))
    st_BPA_corr.append(tr_tmp)
    # For slow:
    tr_tmp = st_BPA_corr.select(channel="??A")[0].copy()
    tr_tmp.data = y
    tr_tmp.stats.channel = "".join((chan_prefixes, "S"))
    st_BPA_corr.append(tr_tmp)
    # 2. Apply reverse time shift to Q and T data:
    x = np.roll(x, int((dt / 2) * fs))
    y = np.roll(y, -int((dt / 2) * fs))
    # 3. And rotate back to QT (PA) coordinates:
    x, y = _rotate_QT_comps(x, y, -phi * np.pi/180)
    # And put data back in stream form:
    st_BPA_corr.select(channel="??P")[0].data = x 
    st_BPA_corr.select(channel="??A")[0].data = y
    # And rotate back into ZNE coords:
    ###!!!st_LQT_corr = _rotate_BPA_to_LQT(st_BPA_corr, back_azi)
    st_LQT_corr = _rotate_BPA_to_LQT(st_BPA_corr.copy(), 0)
    st_ZNE_corr = _rotate_LQT_to_ZNE(st_LQT_corr, back_azi, event_inclin_angle_at_station)
    # And append BPA channels if specified:
    if return_BPA:
        # Append B channel:
        # (Note: L equivilent to B)
        tr_tmp = st_BPA_corr.select(channel="??B")[0]
        st_ZNE_corr.append(tr_tmp)
        # Calculate P and A channels:
        # (Note: P is actually not P, but oriented to N if using current ZNE projection,
        # therefore need to rotate P,A clockwise by src_pol deg with respect to N)
        # (Note: Uses src_pol in horizontal direction, as calc. P and A from horizontal dir at the moment)
        # Rotate by src pol
        # st_BPA_uncorr = _rotate_LQT_to_BPA(st_LQT_uncorr, back_azi)
        tr_tmp_P = st_BPA_corr.select(channel="??P")[0] # (note that P=Q in st_BPA_corr)
        tr_tmp_A = st_BPA_corr.select(channel="??A")[0] # (note that A=T in st_BPA_corr)
        #!tr_tmp_P.data, tr_tmp_A.data = _rotate_QT_comps(tr_tmp_P.data, tr_tmp_A.data, np.deg2rad(src_pol))
        ###!!!tr_tmp_P.data, tr_tmp_A.data = _rotate_QT_comps(st_ZNE_corr.select(channel="??N")[0].data, -st_ZNE_corr.select(channel="??E")[0].data, 
        #                                                       np.deg2rad(src_pol))
        tr_tmp_P.data, tr_tmp_A.data = _rotate_QT_comps(st_LQT_corr.select(channel="??Q")[0].data, st_LQT_corr.select(channel="??T")[0].data, np.deg2rad(src_pol - back_azi))
        # Append P channel:
        st_ZNE_corr.append(tr_tmp_P)
        # Append A channel:
        st_ZNE_corr.append(tr_tmp_A)
    # And remove fast and slow channels, if not wanted:
    if not return_FS:
        st_ZNE_corr.remove(st_ZNE_corr.select(channel="??F")[0])
        st_ZNE_corr.remove(st_ZNE_corr.select(channel="??S")[0])

    # And tidy:
    del st_LQT_uncorr, st_BPA_uncorr, st_LQT_corr, st_BPA_corr
    gc.collect()
    return st_ZNE_corr


# def get_cov_eigen_values(x,y):
#     """
#     Get the covariance matrix eigenvalues.
#     Returns lambda2, lambda1
#     """ 
#     data = np.vstack((x,y))
#     return np.sort(np.linalg.eigvalsh(np.cov(data)))

def calc_dof(y):
    """
    Finds the number of degrees of freedom using a noise trace, y. Uses 
    definition as in Walsh2013.
    Note: Doesn't apply any form of smoothing filter (f(t,h)).
    """
    # Take fft of noise:
    Y = np.fft.fft(y)
    amp = np.absolute(Y)
    # estimate E2 and E4 (from Walsh2013):
    a = np.ones(len(Y))
    a[0] = 0.5
    a[-1] = 0.5
    E2 = np.sum( a * amp**2) # Eq. 25, Walsh2013
    E4 = np.sum( (4 * a**2 / 3) * amp**4) # Eq. 26, Walsh2013
    # And find dof based on E2 and E4 estimates:
    dof = 2 * ( (2 * E2**2 / E4) - 1 )  # Eq. 31, Walsh2013 
    return dof
    
def ftest(data, dof, alpha=0.05, k=2, min_max='min'):
    """
    Finds the confidence bounds value associated with data.
    Note that this version uses the minumum of the data by 
    default.
    Parameters
    ----------
    data : np array
        Data to process.
    dof : int
        Number of degrees of freedom.
    alpha : float
        Confidence level (e.g. if alpha = 0.05, then 95pc confidence level found).
    k : int
        Number of parameters (e.g. phi, dt).
    min_max : specific str
        Whether performs ftest on min or max of data.

    Returns
    -------
    conf_bound : float
        Value of the confidence bounds for the specified confidence 
        level, alpha.
    """
    # Check that dof within acceptable limits:
    if dof < 3:
        raise Exception('Degrees of freedom < 3. Window length is likely too short to perform analysis.')
    # And perform ftest:
    if min_max == 'max':
        data_minmax = np.max(data)
    else:
        data_minmax = np.min(data)
    F = stats.f.ppf(1-alpha, k, dof)
    conf_bound = data_minmax * ( 1 + ( k / (dof - k) ) * F)
    return conf_bound


# @jit((float64[:], float64[:], int64[:], int64[:], int64, int64, int64, float64, float64, float64[:,:,:], float64[:,:,:]), nopython=True, parallel=True)
#@njit((types.float64[:], types.float64[:], types.int64[:], types.int64[:], types.int64, types.int64, types.int64, types.float64, types.float64, types.float64[:,:,:], types.float64[:,:,:]), parallel=True)
# @jit(nopython=True, parallel=True)
#@njit()#parallel=True)
@jit((float64[:], float64[:], int64[:], int64[:], int64, int64, int64, float64, float64, float64[:,:,:], float64[:,:,:]), nopython=True, parallel=True)
def _phi_dt_grid_search(data_arr_Q, data_arr_T, win_start_idxs, win_end_idxs, n_t_steps, n_angle_steps, n_win, fs, rotate_step_deg, 
                                    grid_search_results_all_win_EV, grid_search_results_all_win_XC):
    """Function to do numba accelerated grid search of phis and dts.
    Calculates splitting via eigenvalue (EV) (and also cross-correlation (XC) methods 
    if <grid_search_results_all_win_XC> is specified).
    Note: Currently takes absolute values of rotated, time-shifted waveforms to 
    cross-correlate."""
    # Loop over start and end windows:
    for a in range(n_win):
        start_win_idx = win_start_idxs[a]
        for b in range(n_win):
            end_win_idx = win_end_idxs[b]
            grid_search_idx = int(n_win*a + b)

            # Loop over angles:
            for j in prange(n_angle_steps):
                angle_shift_rad_curr = ((float(j) * rotate_step_deg) - 90.) * np.pi / 180. # (Note: -90 as should loop between -90 and 90 (see phi_labels))

                # Rotate QT waveforms by angle:
                # (Note: Explicit rotation specification as wrapped in numba):
                # Convert angle from counter-clockwise from x to clockwise:
                theta_rot = -angle_shift_rad_curr
                # Perform the rotation explicitely (avoiding creating additional arrays):
                # (Q = y, T = x)
                rot_T_curr = (data_arr_T[start_win_idx:end_win_idx] * np.cos(theta_rot)) - (data_arr_Q[start_win_idx:end_win_idx] * np.sin(theta_rot))
                rot_Q_curr = (data_arr_T[start_win_idx:end_win_idx] * np.sin(theta_rot)) + (data_arr_Q[start_win_idx:end_win_idx] * np.cos(theta_rot))

                # Loop over time shifts:
                for i in range(n_t_steps):
                    t_samp_shift_curr = int( i ) # (note: the minus sign as lag so need to shift back, if assume slow direction aligned with T)
                    # Time-shift data (note: + dt for fast dir (rot Q), and -dt for slow dir (rot T)):
                    rolled_rot_Q_curr = np.roll(rot_Q_curr, +int(t_samp_shift_curr/2.))
                    rolled_rot_T_curr = np.roll(rot_T_curr, -int(t_samp_shift_curr/2.))

                    # ================== Calculate splitting parameters via. EV method ==================
                    # Calculate eigenvalues:
                    xy_arr = np.vstack((rolled_rot_T_curr, rolled_rot_Q_curr))
                    lambdas_unsort = np.linalg.eigvalsh(np.cov(xy_arr))
                    lambdas = np.sort(lambdas_unsort)
                    #lambdas[lambdas==0] = 1e-20

                    # And save eigenvalue results to datastores:
                    # Note: Use lambda2 divided by lambda1 as in Wuestefeld2010 (most stable):
                    grid_search_results_all_win_EV[grid_search_idx,i,j] = lambdas[0] / lambdas[1] 

                    # ================== Calculate splitting parameters via. XC method ==================
                    # if len(grid_search_results_all_win_XC) > 0:
                        # Calculate XC coeffecient and save to array:
                        # if np.std(rolled_rot_Q_curr)  * np.std(rolled_rot_T_curr) > 0. and len(rolled_rot_T_curr) > 0:
                    grid_search_results_all_win_XC[grid_search_idx,i,j] = np.sum( np.abs(rolled_rot_Q_curr * rolled_rot_T_curr) / (np.std(rolled_rot_Q_curr) * np.std(rolled_rot_T_curr))) / float(len(rolled_rot_T_curr))

    return grid_search_results_all_win_EV, grid_search_results_all_win_XC 


@jit((float64[:], float64[:], int64[:], int64[:], int64, int64, int64, float64, float64, float64[:,:,:,:,:], float64[:,:,:,:,:]), nopython=True, parallel=True)
def _phi_dt_grid_search_direct_multi_layer(data_arr_Q, data_arr_T, win_start_idxs, win_end_idxs, n_t_steps, n_angle_steps, n_win, fs, rotate_step_deg, 
                                    grid_search_results_all_win_EV, grid_search_results_all_win_XC):
    """Function to do numba accelerated grid search of phis and dts for a multi-layer 
    inversion directly.
    Calculates splitting via eigenvalue (EV) (and also cross-correlation (XC) methods 
    if <grid_search_results_all_win_XC> is specified).
    Note: Currently takes absolute values of rotated, time-shifted waveforms to 
    cross-correlate.
    Shapes of <grid_search_results_all_win_EV> and <grid_search_results_all_win_XC> 
    must be of:
    <n_win> x <n_t_steps> x <n_angle_steps> x <n_t_steps> x <n_angle_steps>, 
    or equivilently:
    <n_win> x (<n_t_steps> x <n_angle_steps>) ^ <n_layers>.
    CURRENTLY ONLY APPLIED FOR TWO LAYERS."""
    # Loop over start and end windows:
    for a in range(n_win):
        start_win_idx = win_start_idxs[a]
        for b in range(n_win):
            end_win_idx = win_end_idxs[b]
            grid_search_idx = int(n_win*a + b)

            # Apply layers:
            # (n loops = n layers)
            # (Note: Currently only implemented for 2 layers)

            #----------------------- Apply initial rotation and time shift for first layer -----------------------
            # Loop over angles:
            for j in prange(n_angle_steps):
                angle_shift_rad_curr = ((float(j) * rotate_step_deg) - 90.) * np.pi / 180. # (Note: -90 as should loop between -90 and 90 (see phi_labels))

                # Rotate QT waveforms by angle:
                # (Note: Explicit rotation specification as wrapped in numba):
                # Convert angle from counter-clockwise from x to clockwise:
                theta_rot = -angle_shift_rad_curr
                # Perform the rotation explicitely (avoiding creating additional arrays):
                # (Q = y, T = x)
                rot_T_curr = (data_arr_T[start_win_idx:end_win_idx] * np.cos(theta_rot)) - (data_arr_Q[start_win_idx:end_win_idx] * np.sin(theta_rot))
                rot_Q_curr = (data_arr_T[start_win_idx:end_win_idx] * np.sin(theta_rot)) + (data_arr_Q[start_win_idx:end_win_idx] * np.cos(theta_rot))

                # Loop over time shifts:
                for i in range(n_t_steps):
                    t_samp_shift_curr = int( i ) # (note: the minus sign as lag so need to shift back, if assume slow direction aligned with T)
                    # Time-shift data (note: + dt for fast dir (rot Q), and -dt for slow dir (rot T)):
                    rolled_rot_Q_curr = np.roll(rot_Q_curr, +int(t_samp_shift_curr/2.))
                    rolled_rot_T_curr = np.roll(rot_T_curr, -int(t_samp_shift_curr/2.))

                    #----------------------- Apply rotation and time shift for second layer -----------------------
                    # Loop over angles:
                    for l in range(n_angle_steps):
                        angle_shift_rad_curr = ((l * rotate_step_deg) - 90.) * np.pi / 180. # (Note: -90 as should loop between -90 and 90 (see phi_labels))

                        # Rotate QT waveforms by angle:
                        # (Note: Explicit rotation specification as wrapped in numba):
                        # Convert angle from counter-clockwise from x to clockwise:
                        theta_rot = -angle_shift_rad_curr
                        # Perform the rotation explicitely (avoiding creating additional arrays):
                        # (Q = y, T = x)
                        rot_T_curr = (rolled_rot_T_curr[start_win_idx:end_win_idx] * np.cos(theta_rot)) - (rolled_rot_Q_curr[start_win_idx:end_win_idx] * np.sin(theta_rot))
                        rot_Q_curr = (rolled_rot_T_curr[start_win_idx:end_win_idx] * np.sin(theta_rot)) + (rolled_rot_Q_curr[start_win_idx:end_win_idx] * np.cos(theta_rot))

                        # Loop over time shifts:
                        for k in range(n_t_steps):
                            t_samp_shift_curr = int( k ) # (note: the minus sign as lag so need to shift back, if assume slow direction aligned with T)
                            # Time-shift data (note: + dt for fast dir (rot Q), and -dt for slow dir (rot T)):
                            rolled_rot_Q_curr = np.roll(rot_Q_curr, +int(t_samp_shift_curr/2.))
                            rolled_rot_T_curr = np.roll(rot_T_curr, -int(t_samp_shift_curr/2.))

                            #----------------------- And calculate output for current multi-layer -----------------------
                            # ================== Calculate splitting parameters via. EV method ==================
                            # Calculate eigenvalues:
                            xy_arr = np.vstack((rolled_rot_T_curr, rolled_rot_Q_curr))
                            lambdas_unsort = np.linalg.eigvalsh(np.cov(xy_arr))
                            lambdas = np.sort(lambdas_unsort)
                            #lambdas[lambdas==0] = 1e-20

                            # And save eigenvalue results to datastores:
                            # Note: Use lambda2 divided by lambda1 as in Wuestefeld2010 (most stable):
                            grid_search_results_all_win_EV[grid_search_idx,i,j,k,l] = lambdas[0] / lambdas[1] 

                            # ================== Calculate splitting parameters via. XC method ==================
                            # if len(grid_search_results_all_win_XC) > 0:
                                # Calculate XC coeffecient and save to array:
                                # if np.std(rolled_rot_Q_curr)  * np.std(rolled_rot_T_curr) > 0. and len(rolled_rot_T_curr) > 0:
                            #grid_search_results_all_win_XC[grid_search_idx,i,j,k,l] = np.sum( np.abs(rolled_rot_Q_curr * rolled_rot_T_curr) / (np.std(rolled_rot_Q_curr) * 
                            #                                            np.std(rolled_rot_T_curr))) / len(rolled_rot_T_curr)
                    

    return grid_search_results_all_win_EV, grid_search_results_all_win_XC 



class create_splitting_object:
    """
    Class to create splitting object to perform shear wave splitting on.

    Parameters
    ----------
    sts : obspy Stream object
        Obspy stream containing all the waveform data for all the stations and 
        channels to perform the splitting on.

    nonlinloc_event_path : str
        Path to NonLinLoc .grid0.loc.hyp file for the event.

    Attributes
    ----------
    overall_win_start_pre_fast_S_pick : float (default = 0.1 s)
        Overall window start time in seconds before S pick.


    overall_win_start_post_fast_S_pick : float (default = 0.2 s)
        Overall window start time in seconds after S pick.
    
    win_S_pick_tolerance : float (default = 0.1 s)
        Time before and after S pick to not allow windows to start within (in seconds). 
        For example, start windows start at:
        S arrival time - (<overall_win_start_pre_fast_S_pick> + <win_S_pick_tolerance>)
        And end times windows start at:
        S arrival time + <win_S_pick_tolerance> + <overall_win_start_post_fast_S_pick>

    rotate_step_deg : float (default = 2.0 degrees)
        Rotation step size of phi in degrees for the grid search in phi-delay-time space.

    max_t_shift_s : float (default = 0.1 s)
        The maximum time shift the data by in seconds.
    
    n_win : int (default = 10)
        The number of window start and end times to pick. Currently implemented as 
        constant window step sizes within the specified range, as defined by 
        <overall_win_start_pre_fast_S_pick> amd <win_S_pick_tolerance>. Therefore, 
        will calculate splitting for n_win^2 windows in total.

    Methods
    -------
    perform_sws_analysis : Function to perform shear-wave splitting analysis.
    plot : Function to plot the shear-wave splitting results.
    save_result : Function to save sws results to file.
    save_wfs : Function to save uncorrected and corrected waveforms to file.


    """

    def __init__(self, st, nonlinloc_event_path=None, event_uid=None, stations_in=[], S_phase_arrival_times=[], back_azis_all_stations=[], receiver_inc_angles_all_stations=[]):
        """Initiate the class object.

        Parameters
        ----------
        st : obspy Stream object
            Obspy stream containing all the waveform data for all the stations and 
            channels to perform the splitting on.

        nonlinloc_event_path : str
            Path to NonLinLoc .grid0.loc.hyp file for the event. Optional. If supplied, 
            then will use this data for the event rather than <stations>, 
            <back_azis_all_stations> and <receiver_inc_angles_all_stations>. 

        event_uid : str
            Event unique identifying string. For saving data.

        stations_in : list of strs
            List of stations to process, corresponding to order of values in the lists 
            <back_azis_all_stations> and <receiver_inc_angles_all_stations>. All station 
            observations must be constained in <st>. Optional. If not specified, or if 
            <nonlinloc_event_path> specifed then will use <nonlinloc_event_path> instead.

        S_phase_arrival_times : list of obspy UTCDateTime objects
            List of S arrival times as UTCDateTime objects corresponding to order of 
            <stations_in>. 

        back_azis_all_stations : list of floats
            List of back-azimuths for observations at all stations, corresponding to order 
            of values in the list <stations_in>. Optional. If not specified, or 
            if <nonlinloc_event_path> specifed then will use <nonlinloc_event_path> instead.
            Automatically sets all back azimuths to zero if not specified but <stations_in> 
            is specified. Optional. If not specified, or if <nonlinloc_event_path> specifed 
            then will use <nonlinloc_event_path> values instead.

        receiver_inc_angles_all_stations : list of floats
            List of ray inclination angles for observations at all stations, corresponding to 
            order of values in the list <stations_in>. Optional. If not specified, 
            or if <nonlinloc_event_path> specifed then will use <nonlinloc_event_path> instead.
            Automatically sets all inclinations to zero if not specified but <stations_in> 
            is specified.

        """
        # Perform initial checks:
        if not nonlinloc_event_path:
            if len(stations_in) == 0:
                print("Error: <stations_in>, <back_azis_all_stations> and <receiver_inc_angles_all_stations> must be specified if <nonlinloc_event_path> is not specified.")
                raise 
        if len(stations_in) > 0:
            if len(back_azis_all_stations) == 0:
                back_azis_all_stations = np.zeros(len(stations_in))
                receiver_inc_angles_all_stations = np.zeros(len(stations_in))
        if nonlinloc_event_path:
            self.stations_in = []
            self.back_azis_all_stations = []
            self.receiver_inc_angles_all_stations = []
            self.S_phase_arrival_times = []
        else:
            self.stations_in = stations_in
            self.back_azis_all_stations = back_azis_all_stations
            self.receiver_inc_angles_all_stations = receiver_inc_angles_all_stations
            self.S_phase_arrival_times = S_phase_arrival_times
        # Define parameters:
        self.st = st 
        self.nonlinloc_event_path = nonlinloc_event_path
        if self.nonlinloc_event_path:
            self.nonlinloc_hyp_data = swspy.io.read_nonlinloc.read_hyp_file(nonlinloc_event_path)
        if self.nonlinloc_event_path:
            self.origin_time = self.nonlinloc_hyp_data.origin_time
        else:
            self.origin_time = self.st[0].stats.starttime 
        if event_uid:
            self.event_uid = event_uid
        else:
            if nonlinloc_event_path:
                nonlinloc_fname_tmp = os.path.split(nonlinloc_event_path)[-1]
                self.event_uid = ''.join((nonlinloc_fname_tmp.split('.')[2], nonlinloc_fname_tmp.split('.')[3]))
            else:
                self.event_uid = st[0].stats.starttime.strftime("%Y%m%d%H%M%S")
        # Define attributes:
        self.overall_win_start_pre_fast_S_pick = 0.1
        self.overall_win_start_post_fast_S_pick = 0.2
        self.win_S_pick_tolerance = 0.1
        self.rotate_step_deg = 2.0
        self.max_t_shift_s = 0.1
        self.n_win = 10
        # Testing params:
        self.plot_once = True
        # Define datastores:
        self.sws_result_df = None 
        self.sws_multi_layer_result_df = None 


    def _select_windows(self):
        """
        Function to specify window start and end time indices.
        """
        start_idxs_range = int((self.overall_win_start_pre_fast_S_pick - self.win_S_pick_tolerance) * self.fs) 
        win_start_idxs = np.linspace(0, start_idxs_range, num=self.n_win, dtype=int) #np.random.randint(start_idxs_range, size=self.n_win)
        end_idxs_start = int((self.overall_win_start_pre_fast_S_pick + self.overall_win_start_post_fast_S_pick) * self.fs) 
        win_end_idxs = np.linspace(end_idxs_start, end_idxs_start + start_idxs_range, num=self.n_win, dtype=int) #np.random.randint(end_idxs_start, end_idxs_start + start_idxs_range, size=self.n_win)
        return win_start_idxs, win_end_idxs
    

    def _calc_splitting_eig_val_method(self, data_arr_Q, data_arr_T, win_start_idxs, win_end_idxs, sws_method="EV", n_layers=1, num_threads=numba.config.NUMBA_DEFAULT_NUM_THREADS):
        """
        Function to calculate splitting via eigenvalue method.
        sws_method can be EV (eigenvalue) or EV_and_XC (eigenvalue and cross-correlation). EV_and_XC is for automation as in Wustefeld et al. (2010).
        Note: 
        """
        # Perform initial checks:
        if ( sws_method != "EV" ) and ( sws_method != "EV_and_XC" ):
            print("Error: sws_method = ", sws_method, "not recognised. Exiting.")
            sys.exit()
        win_start_idxs = win_start_idxs.astype('int64')
        win_end_idxs = win_end_idxs.astype('int64')
            
        # Setup paramters:
        n_t_steps = int(self.max_t_shift_s * self.fs)
        n_angle_steps = int(180. / self.rotate_step_deg) + 1
        n_win = int(self.n_win)
        fs = float(self.fs)
        rotate_step_deg = float(self.rotate_step_deg)

        # Setup datastores:
        lags_labels = np.arange(0., n_t_steps, 1) / fs 
        phis_labels = np.arange(-90, 90 + rotate_step_deg, rotate_step_deg)
        if n_layers == 1:
            grid_search_results_all_win_EV = np.zeros((n_win**2, n_t_steps, n_angle_steps), dtype=float)
            grid_search_results_all_win_XC = np.zeros((n_win**2, n_t_steps, n_angle_steps), dtype=float)
        elif n_layers == 2:
            grid_search_results_all_win_EV = np.zeros((n_win**2, n_t_steps, n_angle_steps, n_t_steps, n_angle_steps), dtype=float)
            grid_search_results_all_win_XC = np.zeros((n_win**2, n_t_steps, n_angle_steps, n_t_steps, n_angle_steps), dtype=float)
        else:
            print("Error: n_layers = ", n_layers, "not supported (n_layers = 1 or 2 currently). Exiting.")
            sys.exit()

        # Perform grid search:
        if n_layers == 1:
            grid_search_results_all_win_EV, grid_search_results_all_win_XC = _phi_dt_grid_search(data_arr_Q, data_arr_T, win_start_idxs, win_end_idxs, n_t_steps, n_angle_steps, n_win, fs, rotate_step_deg, grid_search_results_all_win_EV, grid_search_results_all_win_XC)
        elif n_layers == 2:
            set_num_threads(int(num_threads))
            grid_search_results_all_win_EV, grid_search_results_all_win_XC = _phi_dt_grid_search_direct_multi_layer(data_arr_Q, data_arr_T, win_start_idxs, win_end_idxs, 
                                                                                                        n_t_steps, n_angle_steps, n_win, fs, rotate_step_deg, 
                                                                                                        grid_search_results_all_win_EV, grid_search_results_all_win_XC)

        # And return results:
        if sws_method == "EV":
            return grid_search_results_all_win_EV, lags_labels, phis_labels
        elif sws_method == "EV_and_XC":
            return grid_search_results_all_win_EV, grid_search_results_all_win_XC, lags_labels, phis_labels


    def _get_phi_and_lag_errors_single_win(self, phi_dt_single_win, lags_labels, phis_labels, tr_for_dof, interp_fac=1):
        """
        Finds the error associated with phi and lag for a given grid search window result.
        Returns errors in phi and lag.
        Calculates errors based on the Silver and Chan (1991) method using the 95% confidence 
        interval, found using an f-test. (Or 90% ?!)
        """
        # Define the error surface to work with:
        # (Done explicitely simply so that can change easily)
        if interp_fac > 1:
            x_tmp = lags_labels 
            y_tmp = phis_labels 
            interp_spline = interpolate.RectBivariateSpline(lags_labels, phis_labels, phi_dt_single_win)
            error_surf = interp_spline(np.linspace(x_tmp[0], x_tmp[-1], interp_fac*len(x_tmp)), np.linspace(y_tmp[0], y_tmp[-1], interp_fac*len(y_tmp)))
            # And update dt and phi labels to correspond to interpolated error surface:
            lags_labels = np.linspace(x_tmp[0], x_tmp[-1], interp_fac*len(x_tmp))
            phis_labels = np.linspace(y_tmp[0], y_tmp[-1], interp_fac*len(y_tmp))
        else:
            error_surf = phi_dt_single_win

        # Find grid search array points where within confidence interval:
        # Use transverse component to calculate dof
        dof = calc_dof(tr_for_dof.data)
        #conf_bound = ftest(error_surf, dof, alpha=0.05, k=2) # (2 sigma)
        conf_bound = ftest(error_surf, dof, alpha=0.003, k=2) # (3 sigma)
        #conf_bound = ftest(error_surf, dof, alpha=0.32, k=2) # (1 sigma)
        conf_mask = error_surf <= conf_bound

        # Find lag dt error:
        # (= 1/2 (not 1/4) width of confidence box (see Silver1991))
        lag_mask = conf_mask.any(axis=1) #(axis=0) # Condense axis1 down along lag (axis0)
        true_idxs = np.where(lag_mask)[0]
        if len(true_idxs) == 0:
            # Artifically set error to zero, as cannot calculte error:
            lag_err = 0.0 
        else:
            # Else calculate error, if possible:
            lag_step_s = lags_labels[1] - lags_labels[0]
            lag_err = (true_idxs[-1] - true_idxs[0] + 1) * lag_step_s * 0.5 #0.25

        # Find fast direction phi error:
        # (= 1/2 (not 1/4) width of confidence box (see Silver1991))
        # (Note: This must deal with angle symetry > 90 or < -90)
        phi_mask = conf_mask.any(axis=0) #(axis=1) # Condense axis0 down along phi (axis1)
        phi_mask_with_pos_neg_overlap = np.hstack((phi_mask, phi_mask, phi_mask))
        if len(np.where(phi_mask_with_pos_neg_overlap)[0]) > 0:
            # Calculate phi error:
            max_false_len = np.diff(np.where(phi_mask_with_pos_neg_overlap)).max() - 1
            # shortest line that contains ALL true values is then:
            max_true_len = len(phi_mask) - max_false_len
            ###max_true_len = np.diff(np.where(phi_mask_with_pos_neg_overlap)).max() + 1
            phi_step_deg = phis_labels[1] - phis_labels[0]
            phi_err = max_true_len * phi_step_deg * 0.5 #0.25
        else:
            # Set phi error equal to zero, as not possible to calculate:
            phi_err = 0.

        return phi_err, lag_err 

    def _get_phi_and_lag_errors(self, grid_search_results_all_win, tr_for_dof, interp_fac=1):
        """Finds the error associated with phi and lag for all search window results.
        Returns errors in phi and lag.
        Calculates errors based on the Silver and Chan (1991) method using the 95% confidence 
        interval, found using an f-test."""
        lags = np.zeros(grid_search_results_all_win.shape[0])
        phis = np.zeros(grid_search_results_all_win.shape[0])
        lag_errs = np.zeros(grid_search_results_all_win.shape[0])
        phi_errs = np.zeros(grid_search_results_all_win.shape[0])
        min_eig_ratios = np.zeros(grid_search_results_all_win.shape[0])
        # Loop over windows:
        for i in range(grid_search_results_all_win.shape[0]):
            grid_search_result_curr_win = grid_search_results_all_win[i,:,:]
            # Get lag and phi:
            min_idxs = np.where(grid_search_result_curr_win == np.min(grid_search_result_curr_win)) 
            lags[i] = self.lags_labels[min_idxs[0][0]] 
            phis[i] = self.phis_labels[min_idxs[1][0]]
            # Get associated error (from f-test with 95% confidence interval):
            # (Note: Uses transverse trace for dof estimation (see Silver and Chan 1991))
            phi_errs[i], lag_errs[i] = self._get_phi_and_lag_errors_single_win(grid_search_result_curr_win, self.lags_labels, self.phis_labels, tr_for_dof,
                                                                                 interp_fac=interp_fac)
            # And get min_eig_ratio:
            min_eig_ratios[i] = np.min(grid_search_result_curr_win)

        return phis, lags, phi_errs, lag_errs, min_eig_ratios


    def _sws_win_clustering(self, lags, phis, lag_errs, phi_errs, min_eig_ratios=None, method="dbscan", return_clusters_data=False):
        """Function to perform sws clustering of phis and lags. This clustering is based on the method of 
        Teanby2004, except that this function uses new coordinate system to deal with the cyclic nature  
        of phi about -90,90, and therefore uses a different clustering algorithm (dbscan) to perform 
        optimal clustering within this new space.
        Note: Performs analysis on normallised lag data.
        """
        # Do initial check on lags to make sure not exactly zero (as coord. transform doesn't work):
        if np.max(lags) == 0.:
            lags = np.ones(len(lags)) * self.fs

        # Weight samples by their error variances:
        # samples_weights = 1. - ((lag_errs/lags)**2 + (phi_errs/phis)**2) # (= 1 - (var_lag_norm + var_phi_norm))
        
        # Convert phis and lags into new coordinate system:
        samples_new_coords =  np.dstack(( ( lags / np.max(lags) ) * np.cos(2 * np.deg2rad(phis)), ( lags / np.max(lags) ) * np.sin(2 * np.deg2rad(phis)) ))[0,:,:]

        # And perform clustering:
        # ward = AgglomerativeClustering(n_clusters=None, linkage='ward',distance_threshold=0.25)
        # ward.fit(samples_new_coords)#, sample_weight=samples_weights)
        db = DBSCAN(eps=0.25, min_samples=int(np.sqrt(len(lags))))
        clustering = db.fit(samples_new_coords)#, sample_weight=samples_weights)
        # Separate samples into clusters:
        n_clusters = len(set(clustering.labels_)) - (1 if -1 in clustering.labels_ else 0) # Note: -1 are noise coords
        if n_clusters > 0:
            clusters_dict = {}
            for i in range(n_clusters):
                curr_cluster_idxs = np.where(clustering.labels_ == i)[0]
                clusters_dict[str(i)] = {}
                clusters_dict[str(i)]['lags'] = lags[curr_cluster_idxs]
                clusters_dict[str(i)]['lag_errs'] = lag_errs[curr_cluster_idxs]
                clusters_dict[str(i)]['phis'] = phis[curr_cluster_idxs]
                clusters_dict[str(i)]['phi_errs'] = phi_errs[curr_cluster_idxs]
                if not min_eig_ratios is None:
                    clusters_dict[str(i)]['min_eig_ratios'] = min_eig_ratios[curr_cluster_idxs]
            # And find smallest variance cluster and smallest variance observation within that cluster:
            # (Note: Variances as in Teanby2004, Eq. 13, 14)
            cluster_vars = np.zeros(n_clusters)
            data_vars = np.zeros(n_clusters)
            for i in range(n_clusters):
                # Calculate cluster variances (Eq. 13, Teanby2004):
                cluster_vars[i] = np.sum( ( clusters_dict[str(i)]['lags'] - np.mean(clusters_dict[str(i)]['lags']) )**2 + ( clusters_dict[str(i)]['phis'] - np.mean(clusters_dict[str(i)]['phis']) )**2 ) / len(clusters_dict[str(i)]['lags'])
                # And calculate cluster variances (Eq. 14, Teanby2004):
                data_vars[i] = ( 1 / np.sum( 1 / ( clusters_dict[str(i)]['lag_errs']**2 ) ) ) + ( 1 / np.sum( 1 / ( clusters_dict[str(i)]['phi_errs']**2 ) ) )
            # And calculate representitive variance of each cluster (= max(cluster_vars, data_vars)):
            cluster_var_0s = np.maximum(cluster_vars, data_vars)
            # And find smallest overall variance cluster:
            min_var_idx = np.argmin(cluster_var_0s)
            smallest_var_cluster = clusters_dict[str(min_var_idx)]
            # And calculate combined dt + phi variance for each point in cluster, to find best overall observation:
            n_obs = len(smallest_var_cluster['lags'])
            cluster_vars_tmp = smallest_var_cluster['lag_errs']**2 + smallest_var_cluster['phi_errs']**2
            opt_obs_idx = np.argmin(cluster_vars_tmp)
            opt_lag = smallest_var_cluster['lags'][opt_obs_idx]
            opt_phi = smallest_var_cluster['phis'][opt_obs_idx]
            opt_lag_err = smallest_var_cluster['lag_errs'][opt_obs_idx]
            opt_phi_err = smallest_var_cluster['phi_errs'][opt_obs_idx]
            if not min_eig_ratios is None:
                opt_eig_ratio = smallest_var_cluster['min_eig_ratios'][opt_obs_idx]

            if return_clusters_data:
                if not min_eig_ratios is None:
                    return opt_phi, opt_lag, opt_phi_err, opt_lag_err, opt_eig_ratio, clusters_dict, min_var_idx
                else:
                    return opt_phi, opt_lag, opt_phi_err, opt_lag_err, clusters_dict, min_var_idx

            else:
                if not min_eig_ratios is None:
                    return opt_phi, opt_lag, opt_phi_err, opt_lag_err, opt_eig_ratio
                else:
                    return opt_phi, opt_lag, opt_phi_err, opt_lag_err

        else:
            print("Warning: Failed to cluster for current receiver of current event. Skipping receiver.")
            if return_clusters_data:
                if not min_eig_ratios is None:
                    return None, None, None, None, None, None, None
                else:
                    return None, None, None, None, None, None
            else:
                if not min_eig_ratios is None:
                    return None, None, None, None, None
                else:
                    return None, None, None, None


    def _convert_phi_from_Q_to_NZ_coords(self, back_azi, event_inclin_angle_at_station, opt_phi):
        """Function to convert phi and all assocated data structures from phi relative to clockwise 
        from Q to phi clockwise from N and phi from up.
        Returns opt_phi_vec as a 2-vector of angle from N in degrees and angle from vertical up in 
        degrees. All angles are in degrees."""
        # Define new phi output as a vector:
        opt_phi_vec = np.zeros(2)
        # Calculate horizontal angle from N:
        opt_phi_vec[0] = opt_phi + back_azi 
        if opt_phi_vec[0] > 360:
            opt_phi_vec[0] = opt_phi_vec[0] - 360 # Convert to 0-360 deg.
        if opt_phi_vec[0] > 90:
            opt_phi_vec[0] = opt_phi_vec[0] - 180 # Convert to -90 to 90 deg
            if opt_phi_vec[0] > 90:
                opt_phi_vec[0] = opt_phi_vec[0] - 180 # Convert to -90 to 90 deg again if needed
        # Calculate vertical angle from up:
        opt_phi_vec[1] = 90 - event_inclin_angle_at_station 
        return opt_phi_vec


    def _calc_Q_w(self, opt_phi_EV, opt_lag_EV, grid_search_results_all_win_XC, tr_for_dof, method="dbscan"):
        """Function to calculate Q_w from Wustefeld et al. (2010), for automated analysis."""
        # 1. Calculate optimal lags for XC method (as already passed values from EV method):
        # (Note: Uses same error and window analysis as XC method)
        # Convert XC to all negative so that universal with mimimum method used in EV analysis:
        grid_search_results_all_win_XC = 1. / grid_search_results_all_win_XC
        # And calculate optimal lags for XC method:
        phis_tmp, lags_tmp, phi_errs_tmp, lag_errs_tmp, min_eig_ratios_tmp = self._get_phi_and_lag_errors(grid_search_results_all_win_XC, tr_for_dof)
        opt_phi_XC, opt_lag_XC, opt_phi_err_XC, opt_lag_err_XC = self._sws_win_clustering(lags_tmp, phis_tmp, lag_errs_tmp, phi_errs_tmp, method=method)

        # 2. Calculate ratios between EV and XC methods:
        dt_diff = opt_lag_XC / opt_lag_EV
        phi_diff = (opt_phi_EV - opt_phi_XC) / 45.
        d_null = np.sqrt( ( (dt_diff**2) + (( phi_diff - 1 )**2) ) / 2.)
        d_good = np.sqrt( ( ( (dt_diff - 1)**2 ) + (phi_diff**2) ) / 2.)
        if d_null > 1:
            d_null = 1
        if d_good > 1:
            d_good = 1

        # 3. Calculate Q_w (Eq. 4, Wustefeld2010):
        if d_null < d_good:
            Q_w = -1 * (1 - d_null)
        else:
            Q_w = 1 - d_good

        return Q_w 

    
    def _rot_phi_from_sws_coords_to_deg_from_N(self, phi_pre_rot_deg, back_azi):
        """Function to rotate phi from LQT splitting to degrees from N."""
        phi_rot_deg = phi_pre_rot_deg + back_azi
        if phi_rot_deg > 360.:
            phi_rot_deg = phi_rot_deg - 360.
        return phi_rot_deg 


    def _get_uncorr_and_corr_waveforms(self, station):
        """Function to return uncorrected and corrected waveforms, if specified."""
        # 1. Get uncorrected waveforms:
        st_ZNE_curr = self.st.select(station=station).copy()
        try:
            back_azi = self.nonlinloc_hyp_data.phase_data[station]['S']['SAzim'] + 180.
            if back_azi >= 360.:
                back_azi = back_azi - 360.
            if self.coord_system == "LQT":
                event_inclin_angle_at_station = self.nonlinloc_hyp_data.phase_data[station]['S']['RDip']
                print("Warning: LQT coord. system not yet fully tested. \n Might produce spurious results...")
            elif self.coord_system == "ZNE":
                event_inclin_angle_at_station = 0. # Rotates ray to arrive at vertical incidence, simulating NE components.
            else:
                print("Error: coord_system =", self.coord_system, "not supported. Exiting.")
                sys.exit()
        except (KeyError, AttributeError) as e:
            if len(self.stations_in) > 0:
                station_idx_tmp = self.stations_in.index(station)
                back_azi = self.back_azis_all_stations[station_idx_tmp]
                event_inclin_angle_at_station = self.receiver_inc_angles_all_stations[station_idx_tmp]
            else:
                print("No S phase pick for station:", station, "therefore skipping this station.")
                raise CustomError("No S phase pick for station:", station, "therefore skipping this station.")

        # And trim data:
        if self.nonlinloc_event_path:
            arrival_time_curr = self.nonlinloc_hyp_data.phase_data[station]['S']['arrival_time']
        else:
            arrival_time_curr = self.S_phase_arrival_times[station_idx_tmp]
        st_ZNE_curr.trim(starttime=arrival_time_curr - self.overall_win_start_pre_fast_S_pick,
                            endtime=arrival_time_curr + self.overall_win_start_post_fast_S_pick + self.max_t_shift_s)

        # 2. And remove splitting to get corrected waveforms:
        try:
            phi_curr = float(self.sws_result_df.loc[self.sws_result_df['station'] == station]['phi_from_Q'].iloc[0])
            dt_curr = float(self.sws_result_df.loc[self.sws_result_df['station'] == station]['dt'].iloc[0])
            phi_err_curr = float(self.sws_result_df.loc[self.sws_result_df['station'] == station]['phi_err'].iloc[0])
            dt_err_curr = float(self.sws_result_df.loc[self.sws_result_df['station'] == station]['dt_err'].iloc[0])
            src_pol_curr = float(self.sws_result_df.loc[self.sws_result_df['station'] == station]['src_pol_from_N'].iloc[0])
            src_pol_err_curr = float(self.sws_result_df.loc[self.sws_result_df['station'] == station]['src_pol_from_N_err'].iloc[0])
            Q_w_curr = float(self.sws_result_df.loc[self.sws_result_df['station'] == station]['Q_w'].iloc[0])
        except TypeError:
            # If cannot get parameters becuase splitting clustering failed, skip station:
            raise CustomError("Cannot get splitting parameters because splitting clustering failed.")
        st_ZNE_curr_sws_corrected = remove_splitting(st_ZNE_curr, phi_curr, dt_curr, back_azi, event_inclin_angle_at_station,
                                                    return_BPA=True, src_pol=src_pol_curr) # (Note: Uses src_pol in horizontal direction, as calc. P and A from horizontal dir at the moment)

        return st_ZNE_curr, st_ZNE_curr_sws_corrected
    

    def _get_uncorr_and_corr_waveforms_multi_layer(self, station):
        """Function to return uncorrected and corrected waveforms for a multi-layer solution, if specified."""
        # 1. Get uncorrected waveforms:
        st_ZNE_curr = self.st.select(station=station).copy()
        try:
            back_azi = self.nonlinloc_hyp_data.phase_data[station]['S']['SAzim'] + 180.
            if back_azi >= 360.:
                back_azi = back_azi - 360.
            if self.coord_system == "LQT":
                event_inclin_angle_at_station = self.nonlinloc_hyp_data.phase_data[station]['S']['RDip']
                print("Warning: LQT coord. system not yet fully tested. \n Might produce spurious results...")
            elif self.coord_system == "ZNE":
                event_inclin_angle_at_station = 0. # Rotates ray to arrive at vertical incidence, simulating NE components.
            else:
                print("Error: coord_system =", self.coord_system, "not supported. Exiting.")
                sys.exit()
        except (KeyError, AttributeError) as e:
            if len(self.stations_in) > 0:
                station_idx_tmp = self.stations_in.index(station)
                back_azi = self.back_azis_all_stations[station_idx_tmp]
                event_inclin_angle_at_station = self.receiver_inc_angles_all_stations[station_idx_tmp]
            else:
                print("No S phase pick for station:", station, "therefore skipping this station.")
                raise CustomError("No S phase pick for station:", station, "therefore skipping this station.")
        # And trim data:
        if self.nonlinloc_event_path:
            arrival_time_curr = self.nonlinloc_hyp_data.phase_data[station]['S']['arrival_time']
        else:
            arrival_time_curr = self.S_phase_arrival_times[station_idx_tmp]
        st_ZNE_curr.trim(starttime=arrival_time_curr - self.overall_win_start_pre_fast_S_pick,
                            endtime=arrival_time_curr + self.overall_win_start_post_fast_S_pick + self.max_t_shift_s)

        # 2. And remove splitting to get corrected waveforms, post layer 2 correction:
        try:
            phi_curr = float(self.sws_multi_layer_result_df.loc[self.sws_multi_layer_result_df['station'] == station]['phi2_from_Q'].iloc[0])
            dt_curr = float(self.sws_multi_layer_result_df.loc[self.sws_multi_layer_result_df['station'] == station]['dt2'].iloc[0])
            src_pol_curr = float(self.sws_multi_layer_result_df.loc[self.sws_multi_layer_result_df['station'] == station]['src_pol_from_N'].iloc[0])
        except TypeError:
            # If cannot get parameters becuase splitting clustering failed, skip station:
            raise CustomError("Cannot get splitting parameters because splitting clustering failed.")
        st_ZNE_curr_sws_corrected_layer_2 = remove_splitting(st_ZNE_curr, phi_curr, dt_curr, back_azi, event_inclin_angle_at_station,
                                                    return_BPA=True, src_pol=src_pol_curr) # (Note: Uses src_pol in horizontal direction, as calc. P and A from horizontal dir at the moment)
        
        # 3. And remove splitting to get corrected waveforms, post layer 1 correction:
        try:
            phi_curr = float(self.sws_multi_layer_result_df.loc[self.sws_multi_layer_result_df['station'] == station]['phi1_from_Q'].iloc[0])
            dt_curr = float(self.sws_multi_layer_result_df.loc[self.sws_multi_layer_result_df['station'] == station]['dt1'].iloc[0])
            src_pol_curr = float(self.sws_multi_layer_result_df.loc[self.sws_multi_layer_result_df['station'] == station]['src_pol_from_N'].iloc[0])
        except TypeError:
            # If cannot get parameters becuase splitting clustering failed, skip station:
            raise CustomError("Cannot get splitting parameters because splitting clustering failed.")
        st_ZNE_curr_sws_corrected_layer_2_ZNE_only = obspy.Stream() # (Note: So that don't duplicate B,P,A,F,S components)
        st_ZNE_curr_sws_corrected_layer_2_ZNE_only.append(st_ZNE_curr_sws_corrected_layer_2.select(channel="??Z")[0])
        st_ZNE_curr_sws_corrected_layer_2_ZNE_only.append(st_ZNE_curr_sws_corrected_layer_2.select(channel="??N")[0])
        st_ZNE_curr_sws_corrected_layer_2_ZNE_only.append(st_ZNE_curr_sws_corrected_layer_2.select(channel="??E")[0])
        st_ZNE_curr_sws_corrected_layer_1_and_2 = remove_splitting(st_ZNE_curr_sws_corrected_layer_2_ZNE_only, phi_curr, dt_curr, back_azi, event_inclin_angle_at_station,
                                                    return_BPA=True, src_pol=src_pol_curr) # (Note: Uses src_pol in horizontal direction, as calc. P and A from horizontal dir at the moment)
        del st_ZNE_curr_sws_corrected_layer_2_ZNE_only
        gc.collect()

        return st_ZNE_curr, st_ZNE_curr_sws_corrected_layer_2, st_ZNE_curr_sws_corrected_layer_1_and_2
    

    def perform_sws_analysis(self, coord_system="ZNE", sws_method="EV", return_clusters_data=True, num_threads=numba.config.NUMBA_DEFAULT_NUM_THREADS):
        """Function to perform splitting analysis. Works in LQT coordinate system 
        as then performs shear-wave-splitting in 3D.
        
        Parameters
        ----------
        coord_system : str
            Coordinate system to perform analysis in. Options are: LQT, ZNE. Will convert 
            splitting angles back into coordinates relative to ZNE whatever system it 
            performs the splitting within. Default = ZNE. 

        sws_method : str
            Method with which to calculate sws parameters. Options are: EV, EV_and_XC.
            EV - Eigenvalue method (as in Silver and Chan (1991), Teanby (2004), Walsh et 
            al. (2013)). EV_and_XC - Same as EV, except also performs cross-correlation 
            for automation approach, as in Wustefeld et al. (2010). Default is EV.

        return_clusters_data : bool
            If True, returns clustering data information. This is primarily used for 
            plotting. Default is False.

        num_threads : int
            Number of threads to use for parallel computing. Default is to use all 
            available threads on the system.

        Returns
        -------
        self.sws_result_df : pandas DataFrame
            A pandas DataFrame containing the key splitting results.

        """
        # Save any parameters to class object:
        self.coord_system = coord_system
        self.sws_method = sws_method
        
        # Perform initial parameter checks:
        if ( sws_method != "EV" ) and ( sws_method != "EV_and_XC" ):
            print("Error: sws_method = ", sws_method, "not recognised. Exiting.")
            sys.exit()

        # Create datastores:
        self.sws_result_df = pd.DataFrame(data={'station': [], 'phi_from_Q': [], 'phi_from_N': [], 'phi_from_U': [], 'phi_err': [], 'dt': [], 'dt_err': [], 'src_pol_from_N': [], 'src_pol_from_U': [], 'src_pol_from_N_err': [], 'src_pol_from_U_err': [], 'Q_w': [], 'lambda2/lambda1 ratio': [], 'ray_back_azi': [], 'ray_inc': []})
        if return_clusters_data:
            self.clustering_info = {}
        self.phi_dt_grid_average = {}
        self.event_station_win_idxs = {}

        # Loop over stations in stream:
        stations_list = []
        for tr in self.st: 
            if tr.stats.station not in stations_list:
                stations_list.append(tr.stats.station)
        self.stations_list = stations_list
        for station in stations_list:
            # 1. Rotate channels into LQT and BPA coordinate systems:
            # Note: Always rotates to LQT, regardless of specified coord_system, 
            #       but if coord_system = ZNE then will set ray to come in 
            #       vertically.
            st_ZNE_curr = self.st.select(station=station).copy()
            try:
                back_azi = self.nonlinloc_hyp_data.phase_data[station]['S']['SAzim'] + 180.
                if back_azi >= 360.:
                    back_azi = back_azi - 360.
                if self.coord_system == "LQT":
                    event_inclin_angle_at_station = self.nonlinloc_hyp_data.phase_data[station]['S']['RDip']
                    print("Warning: LQT coord. system not yet fully tested. \n Might produce spurious results...")
                elif self.coord_system == "ZNE":
                    event_inclin_angle_at_station = 0. # Rotates ray to arrive at vertical incidence (positive up), simulating NE components.
                else:
                    print("Error: coord_system =", self.coord_system, "not supported. Exiting.")
                    sys.exit()
            except (KeyError, AttributeError) as e:
                if len(self.stations_in) > 0:
                    station_idx_tmp = self.stations_in.index(station)
                    back_azi = self.back_azis_all_stations[station_idx_tmp]
                    event_inclin_angle_at_station = self.receiver_inc_angles_all_stations[station_idx_tmp]
                else:
                    print("No S phase pick for station:", station, "therefore skipping this station.")
                    continue
            # And rotate into emerging ray coord system, LQT:
            st_LQT_curr = _rotate_ZNE_to_LQT(st_ZNE_curr, back_azi, event_inclin_angle_at_station)
            # # And rotate into propagation coordinate system (as in Walsh et al. (2013)), BPA:
            # # (Note rotation by back_azi, as want P to be oriented to North for splitting angle calculation)
            # try:
            #     st_BPA_curr = _rotate_LQT_to_BPA(st_LQT_curr, back_azi)
            # except:
            #     print("Warning: Q and/or T components not found. Skipping this event-receiver observation.")
            #     continue
            # del st_LQT_curr #st_ZNE_curr
            # gc.collect()

            # 2. Get S wave channels and trim to pick:
            if self.nonlinloc_event_path:
                arrival_time_curr = self.nonlinloc_hyp_data.phase_data[station]['S']['arrival_time']
            else:
                arrival_time_curr = self.S_phase_arrival_times[station_idx_tmp]
            st_LQT_curr.trim(starttime=arrival_time_curr - self.overall_win_start_pre_fast_S_pick,
                                endtime=arrival_time_curr + self.overall_win_start_post_fast_S_pick + self.max_t_shift_s)
            st_ZNE_curr.trim(starttime=arrival_time_curr - self.overall_win_start_pre_fast_S_pick,
                                endtime=arrival_time_curr + self.overall_win_start_post_fast_S_pick + self.max_t_shift_s)
            try:
                tr_Q = st_LQT_curr.select(station=station, channel="??Q")[0]
                tr_T = st_LQT_curr.select(station=station, channel="??T")[0]
            except IndexError:
                print("Warning: Insufficient data to perform splitting. Skipping this event-receiver observation.")
                continue

            # 3. Get window indices:
            self.fs = tr_T.stats.sampling_rate
            win_start_idxs, win_end_idxs = self._select_windows()

            # 4. Calculate splitting angle and delay times for windows:
            # (Silver and Chan (1991) and Teanby2004 eigenvalue method)
            # 4.a. Get data for all windows:
            if self.sws_method == "EV":
                grid_search_results_all_win_EV, lags_labels, phis_labels = self._calc_splitting_eig_val_method(tr_Q.data, tr_T.data, win_start_idxs, win_end_idxs, 
                                                                                                                    sws_method=self.sws_method, num_threads=num_threads)
            elif self.sws_method == "EV_and_XC":
                try:
                    grid_search_results_all_win_EV, grid_search_results_all_win_XC, lags_labels, phis_labels = self._calc_splitting_eig_val_method(tr_Q.data, tr_T.data, win_start_idxs, win_end_idxs, 
                                                                                                                    sws_method=self.sws_method, num_threads=num_threads)
                except np.linalg.LinAlgError:
                    # And check that returned values without issues:
                    print("Warning: NaN error in _calc_splitting_eig_val_method(). Skipping station:", station)
                    continue
            # Note: Use lambda2 divided by lambda1 as in Wuestefeld2010 (most stable):
            self.lags_labels = lags_labels 
            self.phis_labels = phis_labels 
            # 4.b. Get lag and phi values and errors associated with windows:
            phis, lags, phi_errs, lag_errs, min_eig_ratios = self._get_phi_and_lag_errors(grid_search_results_all_win_EV, tr_T)

            # 6. Perform clustering for all windows to find best result:
            # (Teanby2004 method, but in new coordinate space with dbscan clustering)
            if return_clusters_data:
                opt_phi, opt_lag, opt_phi_err, opt_lag_err, opt_eig_ratio, clusters_dict, min_var_idx = self._sws_win_clustering(lags, phis, lag_errs, phi_errs, min_eig_ratios=min_eig_ratios, method="dbscan", return_clusters_data=True)
            else:
                opt_phi, opt_lag, opt_phi_err, opt_lag_err, opt_eig_ratio = self._sws_win_clustering(lags, phis, lag_errs, phi_errs, min_eig_ratios=min_eig_ratios, method="dbscan")
            # And check that clustered:
            if not opt_phi:
                # If didn't cluster, skip station:
                continue

            # 7. Calculate Wustefeld et al. (2010) quality factor:
            # (For automated approach)
            # (Only if sws_method = "EV_and_XC")
            if self.sws_method == "EV_and_XC":
                Q_w = self._calc_Q_w(opt_phi, opt_lag, grid_search_results_all_win_XC, tr_T, method="dbscan")
            else:
                Q_w = np.nan        

            # 8. And calculate source polarisation:
            # Get wfs:
            # st_ZNE_curr = self.st.select(station=station).copy()
            st_ZNE_curr_sws_corrected = remove_splitting(st_ZNE_curr, opt_phi, opt_lag, back_azi, event_inclin_angle_at_station, return_BPA=False)
            # And find src pol angle relative to N and U (Using ZNE):
            src_pol_deg, src_pol_deg_err = _find_src_pol(st_ZNE_curr_sws_corrected.select(channel="??E")[0].data, 
                                                            st_ZNE_curr_sws_corrected.select(channel="??N")[0].data, 
                                                            st_ZNE_curr_sws_corrected.select(channel="??Z")[0].data)
            del st_ZNE_curr, st_ZNE_curr_sws_corrected
            gc.collect()

            # 9. Convert phi in terms of clockwise from Q to relative to N and Z up:
            # (Note: only do for output phi)
            opt_phi_vec = self._convert_phi_from_Q_to_NZ_coords(back_azi, event_inclin_angle_at_station, opt_phi)

            # 10. And append data to overall datastore:
            # Find ray path data to output:
            if self.nonlinloc_event_path:
                # If nonlinloc supplied data:
                ray_back_azi, ray_inc_at_station = _get_ray_back_azi_and_inc_from_nonlinloc(self.nonlinloc_hyp_data, station)
            elif len(self.stations_in) > 0:
                # Or if sws format data:
                station_idx_tmp = self.stations_in.index(station)
                ray_back_azi = self.back_azis_all_stations[station_idx_tmp]
                ray_inc_at_station = self.receiver_inc_angles_all_stations[station_idx_tmp]
            else:
                # Else if can't specify, then return nans:
                ray_back_azi = np.nan
                ray_inc_at_station = np.nan
            # And append data to result df:
            df_tmp = pd.DataFrame(data={'station': [station], 'phi_from_Q': [opt_phi], 'phi_from_N': [opt_phi_vec[0]], 'phi_from_U': [opt_phi_vec[1]], 'phi_err': [opt_phi_err], 'dt': [opt_lag], 'dt_err': [opt_lag_err], 'src_pol_from_N': [src_pol_deg[0]], 'src_pol_from_U': [src_pol_deg[1]], 'src_pol_from_N_err': [src_pol_deg_err[0]], 'src_pol_from_U_err': [src_pol_deg_err[1]], 'Q_w' : [Q_w], 'lambda2/lambda1 ratio': [opt_eig_ratio], 'ray_back_azi': [ray_back_azi], 'ray_inc': [ray_inc_at_station]})
            self.sws_result_df = pd.concat([self.sws_result_df, df_tmp])
            try:
                opt_phi_idx = np.where(self.phis_labels == opt_phi)[0][0]
                opt_lag_idx = np.where(self.lags_labels == opt_lag)[0][0]
            except IndexError:
                raise CustomError("Cannot find optimal phi or lag.")
            self.phi_dt_grid_average[station] = np.average(grid_search_results_all_win_EV, axis=0) # (lambda2 divided by lambda1 as in Wuestefeld2010 (most stable))
            self.event_station_win_idxs[station] = {}
            self.event_station_win_idxs[station]['win_start_idxs'] = win_start_idxs
            self.event_station_win_idxs[station]['win_end_idxs'] = win_end_idxs
            if return_clusters_data:
                self.clustering_info[station] = {}
                self.clustering_info[station]['min_var_idx'] = min_var_idx
                self.clustering_info[station]['clusters_dict'] = clusters_dict

            # ???. Apply automation approach of Wuestefeld2010 ?!?

        return self.sws_result_df


    def perform_sws_analysis_multi_layer(self, coord_system="ZNE", multi_layer_method="explicit", num_threads=numba.config.NUMBA_DEFAULT_NUM_THREADS):
        """Function to perform splitting analysis for a multi-layered medium. Currently 
         only a 2-layer medium is supported. Works in LQT coordinate system, therefore 
        supporting shear-wave-splitting in 3D.
        Currently doesn't support any method other than <sws_method> = EV and 
        doesn't support returning clustered data.
        Method assumes that apparent delay-time is longer than fast S-wave arrival duration.
        
        Parameters
        ----------
        coord_system : str
            Coordinate system to perform analysis in. Options are: LQT, ZNE. Will convert 
            splitting angles back into coordinates relative to ZNE whatever system it 
            performs the splitting within. Default = ZNE. 

        multi_layer_method : str
            Multi-layer method algorithm to apply. Two options are:
            1. explicit - Applies layers individually, one at a time. Efficient and far fewer 
            free parameters, as computation scales as (n_phi * n_dt) * n-layers.
            2. direct - Applies layers directly together in the inversion. Computationally 
            expensive relative to explicit method, with many free parameters, as computation 
            scales as (n_phi * n_dt) ^ n-layers.

        num_threads : int
            Number of threads to use for parallel computing. Default is to use all 
            available threads on the system.

        Returns
        -------
        self.sws_result_df : pandas DataFrame
            A pandas DataFrame containing the key splitting results for an apparent splitting 
            measurement (i.e. assuming only one layer).

        self.sws_multi_layer_result_df pandas DataFrame
            A pandas DataFrame containing the key splitting results for hte multi-layer result.

        """
        # Save any parameters to class object:
        self.coord_system = coord_system

        # Create datastores:
        sws_result_df_out = pd.DataFrame(data={'station': [], 'phi_from_Q': [], 'phi_from_N': [], 
                                                'phi_from_U': [], 'phi_err': [], 'dt': [], 'dt_err': [], 
                                                'src_pol_from_N': [], 'src_pol_from_U': [], 'src_pol_from_N_err': [], 
                                                'src_pol_from_U_err': [], 'Q_w': [], 'lambda2/lambda1 ratio': [], 'ray_back_azi': [], 'ray_inc': []})
        self.sws_multi_layer_result_df = pd.DataFrame(data={'station': [], 'phi1_from_Q': [], 'phi1_from_N': [], 
                                                'phi1_from_U': [], 'phi1_err': [], 'dt1': [], 'dt1_err': [], 
                                                'phi2_from_Q': [], 'phi2_from_N': [], 'phi2_from_U': [], 'phi2_err': [], 
                                                'dt2': [], 'dt2_err': [], 
                                                'src_pol_from_N': [], 'src_pol_from_U': [], 'src_pol_from_N_err': [], 
                                                'src_pol_from_U_err': [], 'Q_w': [], 'lambda2/lambda1 ratio': [], 'lambda2/lambda1 ratio1': [], 
                                                'lambda2/lambda1 ratio2': [], 'ray_back_azi': [], 'ray_inc': []})
        self.phi_dt_grid_average = {}
        self.phi_dt_grid_average_layer1 = {}
        self.phi_dt_grid_average_layer2 = {}
        self.event_station_win_idxs = {}

        # 0. Get initial, apparent splitting parameters:
        # (assuming single layer)
        self.perform_sws_analysis(coord_system=coord_system, sws_method="EV", return_clusters_data=True)

        # Calculate multi-layer splitting for all sstations:
        # Loop over stations in stream:
        stations_list = []
        for tr in self.st: 
            if tr.stats.station not in stations_list:
                stations_list.append(tr.stats.station)
        self.stations_list = stations_list
        for station in stations_list:
            # ---------------- Prep. data for current station: ----------------
            # 1. Rotate channels into LQT and BPA coordinate systems:
            # Note: Always rotates to LQT, regardless of specified coord_system, 
            #       but if coord_system = ZNE then will set ray to come in 
            #       vertically.
            st_ZNE_curr = self.st.select(station=station).copy()
            try:
                back_azi = self.nonlinloc_hyp_data.phase_data[station]['S']['SAzim'] + 180.
                if back_azi >= 360.:
                    back_azi = back_azi - 360.
                if self.coord_system == "LQT":
                    event_inclin_angle_at_station = self.nonlinloc_hyp_data.phase_data[station]['S']['RDip']
                    print("Warning: LQT coord. system not yet fully tested. \n Might produce spurious results...")
                elif self.coord_system == "ZNE":
                    event_inclin_angle_at_station = 0. # Rotates ray to arrive at vertical incidence (positive up), simulating NE components.
                else:
                    print("Error: coord_system =", self.coord_system, "not supported. Exiting.")
                    sys.exit()
            except (KeyError, AttributeError) as e:
                if len(self.stations_in) > 0:
                    station_idx_tmp = self.stations_in.index(station)
                    back_azi = self.back_azis_all_stations[station_idx_tmp]
                    event_inclin_angle_at_station = self.receiver_inc_angles_all_stations[station_idx_tmp]
                else:
                    print("No S phase pick for station:", station, "therefore skipping this station.")
                    continue
            # And rotate into emerging ray coord system, LQT:
            st_LQT_curr = _rotate_ZNE_to_LQT(st_ZNE_curr, back_azi, event_inclin_angle_at_station)

            # 2. Get S wave channels and trim to pick:
            if self.nonlinloc_event_path:
                arrival_time_curr = self.nonlinloc_hyp_data.phase_data[station]['S']['arrival_time']
            else:
                arrival_time_curr = self.S_phase_arrival_times[station_idx_tmp]
            st_LQT_curr.trim(starttime=arrival_time_curr - self.overall_win_start_pre_fast_S_pick,
                                endtime=arrival_time_curr + self.overall_win_start_post_fast_S_pick + self.max_t_shift_s)
            st_ZNE_curr.trim(starttime=arrival_time_curr - self.overall_win_start_pre_fast_S_pick,
                                endtime=arrival_time_curr + self.overall_win_start_post_fast_S_pick + self.max_t_shift_s)
            try:
                tr_Q = st_LQT_curr.select(station=station, channel="??Q")[0]
                tr_T = st_LQT_curr.select(station=station, channel="??T")[0]
            except IndexError:
                print("Warning: Insufficient data to perform splitting. Skipping this event-receiver observation.")
                continue
            self.fs = tr_T.stats.sampling_rate
            # ---------------- Finished prepping data for current station ----------------

            if multi_layer_method == "explicit":
                # ---------------- Apply explicit multi-layer method ----------------
                # 1. Split data into two windows:
                # (using apparent delay time)
                win_start_idxs, win_end_idxs = self._select_windows()
                dt_app = self.sws_result_df.loc[self.sws_result_df['station'] == station]["dt"].values[0]
                partition_idx = round( (self.overall_win_start_pre_fast_S_pick + dt_app) * self.fs )
                win_start_idxs_partition1 = win_start_idxs
                win_end_idxs_partition1 = partition_idx + np.ones(len(win_end_idxs)) # (Note: Fixed partition)
                win_start_idxs_partition2 = partition_idx + np.ones(len(win_start_idxs)) # (Note: Fixed partition)
                win_end_idxs_partition2 = win_end_idxs

                # 2. Measure the splitting parameters for the 2nd layer:
                # (Silver and Chan (1991) and Teanby2004 eigenvalue method)
                # (Weighted mean of window 1 and window 2)
                # 2.a. For first window:
                grid_search_results_all_win_EV_win1, lags_labels, phis_labels = self._calc_splitting_eig_val_method(tr_Q.data, tr_T.data, win_start_idxs_partition1, 
                                                                                                            win_end_idxs_partition1, sws_method="EV", num_threads=num_threads)
                grid_search_results_all_win_EV_win1[grid_search_results_all_win_EV_win1==0] = 1 # Remove effect of any exact zero eigenvalues (spurious results), while preseerving indices
                self.lags_labels = lags_labels 
                self.phis_labels = phis_labels 
                phis, lags, phi_errs, lag_errs, min_eig_ratios = self._get_phi_and_lag_errors(grid_search_results_all_win_EV_win1, tr_T)         
                opt_phi_win1, opt_lag_win1, opt_phi_err_win1, opt_lag_err_win1, opt_eig_ratio1, clusters_dict1, min_var_idx1 = self._sws_win_clustering(lags, phis, lag_errs, phi_errs, 
                                                                                                                        min_eig_ratios=min_eig_ratios, method="dbscan", return_clusters_data=True)
                if not opt_phi_win1:
                    continue # If didn't cluster, skip station
                # 2.b. For second window:
                grid_search_results_all_win_EV_win2, lags_labels, phis_labels = self._calc_splitting_eig_val_method(tr_Q.data, tr_T.data, win_start_idxs_partition2, 
                                                                                                            win_end_idxs_partition2, sws_method="EV", num_threads=num_threads)
                grid_search_results_all_win_EV_win2[grid_search_results_all_win_EV_win2==0] = 1 # Remove effect of any exact zero eigenvalues (spurious results), while preseerving indices
                self.lags_labels = lags_labels 
                self.phis_labels = phis_labels 
                phis, lags, phi_errs, lag_errs, min_eig_ratios = self._get_phi_and_lag_errors(grid_search_results_all_win_EV_win2, tr_T)         
                opt_phi_win2, opt_lag_win2, opt_phi_err_win2, opt_lag_err_win2, opt_eig_ratio2, clusters_dict2, min_var_idx2 = self._sws_win_clustering(lags, phis, lag_errs, phi_errs, 
                                                                                                                        min_eig_ratios=min_eig_ratios, method="dbscan", return_clusters_data=True)
                if not opt_phi_win2:
                    continue # If didn't cluster, skip station

                # 2.c. Pick best (most linearised) result:
                min_EV_both_wins = np.array([opt_eig_ratio1, opt_eig_ratio2])
                min_EV_both_wins[min_EV_both_wins==0] = 1e6 # Remove effect of any exact zero eigenvalues (spurious results), while preseerving indices
                best_win_idx = np.argmin(min_EV_both_wins)
                opt_phi_layer2 = np.array([opt_phi_win1, opt_phi_win2])[best_win_idx]
                opt_lag_layer2 = np.array([opt_lag_win1, opt_lag_win2])[best_win_idx]
                opt_phi_err_layer2 = np.array([opt_phi_err_win1, opt_phi_err_win2])[best_win_idx]
                opt_lag_err_layer2 = np.array([opt_lag_err_win1, opt_lag_err_win2])[best_win_idx]
                opt_eig_ratio_layer2 = np.min(min_EV_both_wins)
                if best_win_idx == 0:
                    grid_search_results_all_win_EV_layer2 = grid_search_results_all_win_EV_win1
                    self.clustering_info[station] = {}
                    self.clustering_info[station]['layer2'] = {}
                    self.clustering_info[station]['layer2']['min_var_idx'] = min_var_idx1
                    self.clustering_info[station]['layer2']['clusters_dict'] = clusters_dict1
                elif best_win_idx == 1:
                    grid_search_results_all_win_EV_layer2 = grid_search_results_all_win_EV_win2
                    self.clustering_info[station] = {}
                    self.clustering_info[station]['layer2'] = {}
                    self.clustering_info[station]['layer2']['min_var_idx'] = min_var_idx2
                    self.clustering_info[station]['layer2']['clusters_dict'] = clusters_dict2
                    
                # 3. Remove effect of layer 2 anisotropy:
                st_ZNE_curr_sws_layer_2_removed = remove_splitting(st_ZNE_curr, opt_phi_layer2, opt_lag_layer2, back_azi, event_inclin_angle_at_station, return_BPA=False)
                st_LQT_curr_sws_layer_2_removed = _rotate_ZNE_to_LQT(st_ZNE_curr_sws_layer_2_removed, back_azi, event_inclin_angle_at_station)

                # 4. Find splitting parameters for layer 1:
                # (by perform splitting for entire trace post layer 2 correction)
                tr_Q = st_LQT_curr_sws_layer_2_removed.select(station=station, channel="??Q")[0]
                tr_T = st_LQT_curr_sws_layer_2_removed.select(station=station, channel="??T")[0]
                grid_search_results_all_win_EV_layer1, lags_labels, phis_labels = self._calc_splitting_eig_val_method(tr_Q.data, tr_T.data, win_start_idxs, 
                                                                                                            win_end_idxs, sws_method="EV", num_threads=num_threads)
                grid_search_results_all_win_EV_layer1[grid_search_results_all_win_EV_layer1==0] = 1 # Remove effect of any exact zero eigenvalues (spurious results), while preseerving indices
                self.lags_labels = lags_labels 
                self.phis_labels = phis_labels 
                phis, lags, phi_errs, lag_errs, min_eig_ratios = self._get_phi_and_lag_errors(grid_search_results_all_win_EV_layer1, tr_T)         
                opt_phi_layer1, opt_lag_layer1, opt_phi_err_layer1, opt_lag_err_layer1, opt_eig_ratio_layer1, clusters_dict_layer1, min_var_idx_layer1 = self._sws_win_clustering(lags, phis, lag_errs, phi_errs, 
                                                                                                                                min_eig_ratios=min_eig_ratios, method="dbscan", return_clusters_data=True)
                # And write clustering info:
                self.clustering_info[station]['layer1'] = {}
                self.clustering_info[station]['layer1']['min_var_idx'] = min_var_idx_layer1
                self.clustering_info[station]['layer1']['clusters_dict'] = clusters_dict_layer1
                if not opt_phi_layer1:
                    continue # If didn't cluster, skip station
                
                # 5. And calculate source polarisation:
                # Get wfs:
                st_ZNE_curr_sws_corrected = remove_splitting(st_ZNE_curr_sws_layer_2_removed, opt_phi_layer1, opt_lag_layer1, back_azi, event_inclin_angle_at_station, 
                                                            return_BPA=False)
                # And find src pol angle relative to N and U (Using ZNE):
                src_pol_deg, src_pol_deg_err = _find_src_pol(st_ZNE_curr_sws_corrected.select(channel="??E")[0].data, 
                                                                st_ZNE_curr_sws_corrected.select(channel="??N")[0].data, 
                                                                st_ZNE_curr_sws_corrected.select(channel="??Z")[0].data)
                del st_ZNE_curr, st_ZNE_curr_sws_layer_2_removed, st_LQT_curr_sws_layer_2_removed
                gc.collect()

                # ---------------- Finished applying explicit multi-layer method ----------------

            elif multi_layer_method == "direct":
                # ---------------- Apply direct multi-layer method ----------------
                # 1. Get window idxs:
                win_start_idxs, win_end_idxs = self._select_windows()

                # 2. Find splitting parameters directly for multiple layers:
                grid_search_result_multi_layer_inv, lags_labels, phis_labels = self._calc_splitting_eig_val_method(tr_Q.data, tr_T.data, win_start_idxs, 
                                                                                                            win_end_idxs, sws_method="EV", n_layers=2)
                self.lags_labels = lags_labels 
                self.phis_labels = phis_labels 

                # 3. Remove effect of any exact zero eigenvalues (spurious results), while preseerving indices:
                grid_search_result_multi_layer_inv[grid_search_result_multi_layer_inv==0] = 1e6 # Remove effect of any exact zero eigenvalues (spurious results), while preseerving indices

                # 4. Find optimal splitting parameters:
                # (Note: Currently doesn't do this with clustering, but just an absolute minimum!!!):
                abs_min_indices = np.unravel_index(np.argmin(grid_search_result_multi_layer_inv, axis=None), grid_search_result_multi_layer_inv.shape)
                opt_phi_layer1, opt_lag_layer1 = self.phis_labels[abs_min_indices[4]], self.lags_labels[abs_min_indices[3]]
                opt_phi_err_layer1, opt_lag_err_layer1 = 0, 0 # (Note: Currently don't calculate errors for this method)
                opt_phi_layer2, opt_lag_layer2 = self.phis_labels[abs_min_indices[2]], self.lags_labels[abs_min_indices[1]]
                opt_phi_err_layer2, opt_lag_err_layer2 = 0, 0 # (Note: Currently don't calculate errors for this method)
                opt_eig_ratio  = grid_search_result_multi_layer_inv[abs_min_indices[0], abs_min_indices[1], abs_min_indices[2], abs_min_indices[3], 
                                                                       abs_min_indices[4]]
                opt_eig_ratio_layer1, opt_eig_ratio_layer2 = opt_eig_ratio, opt_eig_ratio
                grid_search_results_all_win_EV_layer1 = grid_search_result_multi_layer_inv[:, abs_min_indices[1], abs_min_indices[2], 
                                                                                           :, :]
                grid_search_results_all_win_EV_layer2 = np.zeros(np.shape(grid_search_results_all_win_EV_layer1)) # Set layer 2 to zeros, simply as can't untangle result.
                del grid_search_result_multi_layer_inv
                gc.collect()

                # 5. And calculate source polarisation:
                # Get wfs:
                st_ZNE_curr_layer_2_corr = remove_splitting(st_ZNE_curr, opt_phi_layer2, opt_lag_layer2, back_azi, event_inclin_angle_at_station, 
                                                            return_BPA=False)
                st_ZNE_curr_sws_corrected = remove_splitting(st_ZNE_curr_layer_2_corr, opt_phi_layer1, opt_lag_layer1, back_azi, event_inclin_angle_at_station, 
                                                            return_BPA=False)
                # And find src pol angle relative to N and U (Using ZNE):
                src_pol_deg, src_pol_deg_err = _find_src_pol(st_ZNE_curr_sws_corrected.select(channel="??E")[0].data, 
                                                                st_ZNE_curr_sws_corrected.select(channel="??N")[0].data, 
                                                                st_ZNE_curr_sws_corrected.select(channel="??Z")[0].data)
                del st_ZNE_curr, st_ZNE_curr_layer_2_corr
                gc.collect()

                # ---------------- Finished applying direct multi-layer method ----------------
            
            else:
                print("Error: multi_layer_method = ", multi_layer_method, "not recognised. Exiting.")
                sys.exit()

            # 6. Convert phi in terms of clockwise from Q to relative to N and Z up:
            # (Note: only do for output phi)
            # For layer 1:
            opt_phi_vec_layer1 = self._convert_phi_from_Q_to_NZ_coords(back_azi, event_inclin_angle_at_station, opt_phi_layer1)
            # For layer 2:
            opt_phi_vec_layer2 = self._convert_phi_from_Q_to_NZ_coords(back_azi, event_inclin_angle_at_station, opt_phi_layer2)

            # 7. Calculate overall eigenvalue ratio (measure of linearity):
            st_tmp = _rotate_ZNE_to_LQT(st_ZNE_curr_sws_corrected, back_azi, event_inclin_angle_at_station)
            xy_arr = np.vstack((st_tmp.select(channel="??T")[0].data, st_tmp.select(channel="??Q")[0].data))
            lambdas_unsort = np.linalg.eigvalsh(np.cov(xy_arr))
            lambdas = np.sort(lambdas_unsort)
            opt_eig_ratio = lambdas[0] / lambdas[1] 
            del st_ZNE_curr_sws_corrected, st_tmp, xy_arr
            gc.collect()

            # 8. And append data to overall datastore:
            # Find ray path data to output:
            if self.nonlinloc_event_path:
                # If nonlinloc supplied data:
                ray_back_azi, ray_inc_at_station = _get_ray_back_azi_and_inc_from_nonlinloc(self.nonlinloc_hyp_data, station)
            elif len(self.stations_in) > 0:
                # Or if sws format data:
                station_idx_tmp = self.stations_in.index(station)
                ray_back_azi = self.back_azis_all_stations[station_idx_tmp]
                ray_inc_at_station = self.receiver_inc_angles_all_stations[station_idx_tmp]
            else:
                # Else if can't specify, then return nans:
                ray_back_azi = np.nan
                ray_inc_at_station = np.nan
            # And append data to result dfs:
            df_tmp = pd.DataFrame(data={'station': [station], 'phi1_from_Q': [opt_phi_layer1], 'phi1_from_N': [opt_phi_vec_layer1[0]], 'phi1_from_U': [opt_phi_vec_layer1[1]], 'phi1_err': [opt_phi_err_layer1], 'dt1': [opt_lag_layer1], 'dt1_err': [opt_lag_err_layer1], 
                                    'phi2_from_Q': [opt_phi_layer2], 'phi2_from_N': [opt_phi_vec_layer2[0]], 'phi2_from_U': [opt_phi_vec_layer2[1]], 'phi2_err': [opt_phi_err_layer2], 'dt2': [opt_lag_layer2], 'dt2_err': [opt_lag_err_layer2], 
                                        'src_pol_from_N': [src_pol_deg[0]], 'src_pol_from_U': [src_pol_deg[1]], 'src_pol_from_N_err': [src_pol_deg_err[0]], 'src_pol_from_U_err': [src_pol_deg_err[1]], 'Q_w' : [np.nan],  'lambda2/lambda1 ratio': [opt_eig_ratio],
                                         'lambda2/lambda1 ratio1': [opt_eig_ratio_layer1], 'lambda2/lambda1 ratio2': [opt_eig_ratio_layer2], 'ray_back_azi': [ray_back_azi], 'ray_inc': [ray_inc_at_station]})
            self.sws_multi_layer_result_df = pd.concat([self.sws_multi_layer_result_df, df_tmp])
            df_tmp = pd.DataFrame(data={'station': [station], 'phi_from_Q': [opt_phi_layer1], 'phi_from_N': [opt_phi_vec_layer1[0]], 'phi_from_U': [opt_phi_vec_layer1[1]], 'phi_err': [opt_phi_err_layer1], 'dt': [opt_lag_layer1], 'dt_err': [opt_lag_err_layer1], 
                                        'src_pol_from_N': [src_pol_deg[0]], 'src_pol_from_U': [src_pol_deg[1]], 'src_pol_from_N_err': [src_pol_deg_err[0]], 'src_pol_from_U_err': [src_pol_deg_err[1]], 'Q_w' : [np.nan],  'lambda2/lambda1 ratio': [opt_eig_ratio],
                                        'ray_back_azi': [ray_back_azi], 'ray_inc': [ray_inc_at_station]})
            self.sws_result_df = pd.concat([self.sws_result_df, df_tmp])
            try:
                opt_phi_idx = np.where(self.phis_labels == opt_phi_layer1)[0][0]
                opt_lag_idx = np.where(self.lags_labels == opt_lag_layer1)[0][0]
            except IndexError:
                raise CustomError("Cannot find optimal phi or lag.")
            self.phi_dt_grid_average[station] = np.average(grid_search_results_all_win_EV_layer1, axis=0) # (lambda2 divided by lambda1 as in Wuestefeld2010 (most stable))
            self.phi_dt_grid_average_layer1[station] = np.average(grid_search_results_all_win_EV_layer1, axis=0) # (lambda2 divided by lambda1 as in Wuestefeld2010 (most stable))
            self.phi_dt_grid_average_layer2[station] = np.average(grid_search_results_all_win_EV_layer2, axis=0) # (lambda2 divided by lambda1 as in Wuestefeld2010 (most stable))
            self.event_station_win_idxs[station] = {}
            self.event_station_win_idxs[station]['win_start_idxs'] = win_start_idxs
            self.event_station_win_idxs[station]['win_end_idxs'] = win_end_idxs

        return self.sws_result_df, self.sws_multi_layer_result_df


    def plot(self, outdir=None, suppress_direct_plotting=False):
        """Function to perform plotting...
        """
        # Loop over stations, plotting:
        for station in self.stations_list:
            # Get data:
            # Waveforms:
            try:
                st_ZNE_curr, st_ZNE_curr_sws_corrected = self._get_uncorr_and_corr_waveforms(station)
            except CustomError:
                print("Skipping waveform correction for station:", station)
                continue
            # Waveforms (multi-layer splitting):
            if self.sws_multi_layer_result_df is not None:
                print("Passed multi-layer result, therefore plotting this result.")
                # (Note: Get layer 2 correction, as intermediate stage correction, and layer 1+2 correction is full correction)
                del st_ZNE_curr_sws_corrected
                st_ZNE_curr, st_ZNE_curr_sws_corrected_layer_2, st_ZNE_curr_sws_corrected = self._get_uncorr_and_corr_waveforms_multi_layer(station)
            # Splitting parameters:
            try:
                phi_curr = float(self.sws_result_df.loc[self.sws_result_df['station'] == station]['phi_from_Q'].iloc[0])
                phi_from_N_curr = float(self.sws_result_df.loc[self.sws_result_df['station'] == station]['phi_from_N'].iloc[0])
                dt_curr = float(self.sws_result_df.loc[self.sws_result_df['station'] == station]['dt'].iloc[0])
                phi_err_curr = float(self.sws_result_df.loc[self.sws_result_df['station'] == station]['phi_err'].iloc[0])
                dt_err_curr = float(self.sws_result_df.loc[self.sws_result_df['station'] == station]['dt_err'].iloc[0])
                src_pol_curr = float(self.sws_result_df.loc[self.sws_result_df['station'] == station]['src_pol_from_N'].iloc[0])
                src_pol_err_curr = float(self.sws_result_df.loc[self.sws_result_df['station'] == station]['src_pol_from_N_err'].iloc[0])
                Q_w_curr = float(self.sws_result_df.loc[self.sws_result_df['station'] == station]['Q_w'].iloc[0])
                opt_eig_ratio_curr = float(self.sws_result_df.loc[self.sws_result_df['station'] == station]['lambda2/lambda1 ratio'].iloc[0])
                if self.sws_multi_layer_result_df is not None:
                    dt_layer1 = float(self.sws_multi_layer_result_df.loc[self.sws_multi_layer_result_df['station'] == station]['dt1'].iloc[0])
                    dt_err_layer1 = float(self.sws_multi_layer_result_df.loc[self.sws_multi_layer_result_df['station'] == station]['dt1_err'].iloc[0])
                    dt_layer2 = float(self.sws_multi_layer_result_df.loc[self.sws_multi_layer_result_df['station'] == station]['dt2'].iloc[0])
                    dt_err_layer2 = float(self.sws_multi_layer_result_df.loc[self.sws_multi_layer_result_df['station'] == station]['dt2_err'].iloc[0])
                    phi_layer1 = float(self.sws_multi_layer_result_df.loc[self.sws_multi_layer_result_df['station'] == station]['phi1_from_N'].iloc[0])
                    phi_err_layer1 = float(self.sws_multi_layer_result_df.loc[self.sws_multi_layer_result_df['station'] == station]['phi1_err'].iloc[0])
                    phi_layer2 = float(self.sws_multi_layer_result_df.loc[self.sws_multi_layer_result_df['station'] == station]['phi2_from_N'].iloc[0])
                    phi_err_layer2 = float(self.sws_multi_layer_result_df.loc[self.sws_multi_layer_result_df['station'] == station]['phi2_err'].iloc[0])
            except TypeError:
                # If cannot get parameters becuase splitting clustering failed, skip station:
                print("Cannot get splitting parameters because splitting clustering failed. Skipping station:", 
                        station)
                continue
            # And get unncorrected P and A waveforms:
            tr_tmp_P = st_ZNE_curr.select(channel="??N")[0].copy()
            tr_tmp_A = st_ZNE_curr.select(channel="??E")[0].copy()
            chan_prefixes_tmp = tr_tmp_P.stats.channel[0:2]
            tr_tmp_P.stats.channel = chan_prefixes_tmp+"P"
            tr_tmp_A.stats.channel = chan_prefixes_tmp+"A"
            tr_tmp_P.data, tr_tmp_A.data = _rotate_QT_comps(st_ZNE_curr.select(channel="??N")[0].data, 
                                                                -st_ZNE_curr.select(channel="??E")[0].data, 
                                                                np.deg2rad(src_pol_curr))
            st_ZNE_curr.append(tr_tmp_P)
            st_ZNE_curr.append(tr_tmp_A)
            del tr_tmp_P, tr_tmp_A
            gc.collect()
        
            # Plot data:
            # Setup figure:
            fig = plt.figure(constrained_layout=True, figsize=(20,16))
            if suppress_direct_plotting:
                plt.ion()
            gs = fig.add_gridspec(4, 5, wspace=0.3, hspace=0.3) # no. rows, no. cols
            wfs_ax = fig.add_subplot(gs[0:2, 0:2])
            wfs_ax.get_xaxis().set_visible(False)
            wfs_ax.get_yaxis().set_visible(False)
            wfs_ax_Z = wfs_ax.inset_axes([0, 2/3, 1.0, 1/3])#wfs_ax, width="100%", height="30%", loc=1)#, bbox_to_anchor=(.7, .5, .3, .5))
            wfs_ax_N = wfs_ax.inset_axes([0, 1/3, 1.0, 1/3])#wfs_ax, width="100%", height="30%", loc=2)
            wfs_ax_E = wfs_ax.inset_axes([0, 0.0, 1.0, 1/3])#wfs_ax, width="100%", height="30%", loc=3)
            wfs_ax_Z.get_xaxis().set_visible(False)
            wfs_ax_N.get_xaxis().set_visible(False)
            pa_wfs_ax = fig.add_subplot(gs[0:2, 3:5])
            pa_wfs_ax.get_xaxis().set_visible(False)
            pa_wfs_ax.get_yaxis().set_visible(False)
            wfs_ax_P_uncorr = pa_wfs_ax.inset_axes([0, 0.75, 1.0, 0.25])#wfs_ax, width="100%", height="30%", loc=3)
            wfs_ax_A_uncorr = pa_wfs_ax.inset_axes([0, 0.5, 1.0, 0.25])#wfs_ax, width="100%", height="30%", loc=3)
            wfs_ax_P_corr = pa_wfs_ax.inset_axes([0, 0.25, 1.0, 0.25])#wfs_ax, width="100%", height="30%", loc=3)
            wfs_ax_A_corr = pa_wfs_ax.inset_axes([0, 0.0, 1.0, 0.25])#wfs_ax, width="100%", height="30%", loc=3)
            wfs_ax_P_uncorr.get_xaxis().set_visible(False)
            wfs_ax_A_uncorr.get_xaxis().set_visible(False)
            wfs_ax_P_corr.get_xaxis().set_visible(False)
            fs_before_after_ax = fig.add_subplot(gs[2, 0:2])
            fs_before_after_ax.get_xaxis().set_visible(False)
            fs_before_after_ax.get_yaxis().set_visible(False)
            fs_before_ax = fs_before_after_ax.inset_axes([0, 0, 0.5, 1.0])
            fs_after_ax = fs_before_after_ax.inset_axes([0.5, 0, 0.5, 1.0])
            fs_after_ax.get_yaxis().set_visible(False)
            text_ax = fig.add_subplot(gs[2, 2])
            text_ax.axis('off')
            particle_motions_ax = fig.add_subplot(gs[3, 0:2])
            particle_motions_ax.get_xaxis().set_visible(False)
            particle_motions_ax.get_yaxis().set_visible(False)
            ne_uncorr_ax = particle_motions_ax.inset_axes([0, 0, 0.5, 1.0])
            ne_corr_ax = particle_motions_ax.inset_axes([0.5, 0, 0.5, 1.0])
            ne_corr_ax.get_yaxis().set_visible(False)
            phi_dt_ax = fig.add_subplot(gs[2:4, 3:5])
            if self.clustering_info:
                cluster_results_ax = fig.add_subplot(gs[3, 2])
                cluster_results_ax.get_xaxis().set_visible(False)
                cluster_results_ax.get_yaxis().set_visible(False)
                cluster_results_ax_phi = cluster_results_ax.inset_axes([0, 0.5, 1.0, 0.5])
                cluster_results_ax_dt = cluster_results_ax.inset_axes([0, 0.0, 1.0, 0.5])

            # Plot data on figure:
            t = np.arange(len(st_ZNE_curr.select(channel="??Z")[0].data)) / st_ZNE_curr.select(channel="??Z")[0].stats.sampling_rate
            # Waveforms:
            max_amp = 0.
            max_amp = np.max(np.abs(st_ZNE_curr.select(channel="??Z")[0].data))
            if np.max(np.abs(st_ZNE_curr.select(channel="??N")[0].data)) > max_amp:
                max_amp = np.max(np.abs(st_ZNE_curr.select(channel="??N")[0].data))
            if np.max(np.abs(st_ZNE_curr.select(channel="??E")[0].data)) > max_amp:
                max_amp = np.max(np.abs(st_ZNE_curr.select(channel="??E")[0].data))
            if np.max(np.abs(st_ZNE_curr.select(channel="??P")[0].data)) > max_amp:
                max_amp = np.max(np.abs(st_ZNE_curr.select(channel="??P")[0].data))
            if np.max(np.abs(st_ZNE_curr.select(channel="??A")[0].data)) > max_amp:
                max_amp = np.max(np.abs(st_ZNE_curr.select(channel="??A")[0].data))
            if np.max(np.abs(st_ZNE_curr_sws_corrected.select(channel="??Z")[0].data)) > max_amp:
                max_amp = np.max(np.abs(st_ZNE_curr_sws_corrected.select(channel="??Z")[0].data))
            if np.max(np.abs(st_ZNE_curr_sws_corrected.select(channel="??N")[0].data)) > max_amp:
                max_amp = np.max(np.abs(st_ZNE_curr_sws_corrected.select(channel="??N")[0].data))
            if np.max(np.abs(st_ZNE_curr_sws_corrected.select(channel="??E")[0].data)) > max_amp:
                max_amp = np.max(np.abs(st_ZNE_curr_sws_corrected.select(channel="??E")[0].data))
            if np.max(np.abs(st_ZNE_curr_sws_corrected.select(channel="??P")[0].data)) > max_amp:
                max_amp = np.max(np.abs(st_ZNE_curr_sws_corrected.select(channel="??P")[0].data))
            if np.max(np.abs(st_ZNE_curr_sws_corrected.select(channel="??A")[0].data)) > max_amp:
                max_amp = np.max(np.abs(st_ZNE_curr_sws_corrected.select(channel="??A")[0].data))
            wfs_ax_Z.plot(t, st_ZNE_curr.select(channel="??Z")[0].data, c='k', label='before')
            wfs_ax_N.plot(t, st_ZNE_curr.select(channel="??N")[0].data, c='k')
            wfs_ax_E.plot(t, st_ZNE_curr.select(channel="??E")[0].data, c='k')
            wfs_ax_P_uncorr.plot(t, st_ZNE_curr.select(channel="??P")[0].data, c='k')
            wfs_ax_A_uncorr.plot(t, st_ZNE_curr.select(channel="??A")[0].data, c='k')
            wfs_ax_Z.plot(t, st_ZNE_curr_sws_corrected.select(channel="??Z")[0].data, c='#D73215', label='after')
            wfs_ax_N.plot(t, st_ZNE_curr_sws_corrected.select(channel="??N")[0].data, c='#D73215')
            wfs_ax_E.plot(t, st_ZNE_curr_sws_corrected.select(channel="??E")[0].data, c='#D73215')
            wfs_ax_P_corr.plot(t, st_ZNE_curr_sws_corrected.select(channel="??P")[0].data, c='#D73215') #c='#1E69A9')
            wfs_ax_A_corr.plot(t, st_ZNE_curr_sws_corrected.select(channel="??A")[0].data, c='#D73215') #c='#1E69A9')
            # Plot intermediate post layer 2 correction (multi-layer splitting):
            if self.sws_multi_layer_result_df is not None:
                wfs_ax_P_uncorr.plot(t, st_ZNE_curr_sws_corrected_layer_2.select(channel="??P")[0].data, c='#0078B8', alpha=0.5)
                wfs_ax_A_uncorr.plot(t, st_ZNE_curr_sws_corrected_layer_2.select(channel="??A")[0].data, c='#0078B8', alpha=0.5)
            fs = st_ZNE_curr.select(channel="??N")[0].stats.sampling_rate
            for i in range(len(self.event_station_win_idxs[station]['win_start_idxs'])):
                wfs_ax_N.axvline(x = self.event_station_win_idxs[station]['win_start_idxs'][i] / fs, c='k', alpha=0.25)
                wfs_ax_N.axvline(x = self.event_station_win_idxs[station]['win_end_idxs'][i] / fs, c='k', alpha=0.25)
                wfs_ax_E.axvline(x = self.event_station_win_idxs[station]['win_start_idxs'][i] / fs, c='k', alpha=0.25)
                wfs_ax_E.axvline(x = self.event_station_win_idxs[station]['win_end_idxs'][i] / fs, c='k', alpha=0.25)
            wfs_ax_Z.set_xlim(np.min(t), np.max(t))
            wfs_ax_N.set_xlim(np.min(t), np.max(t))
            wfs_ax_E.set_xlim(np.min(t), np.max(t))
            wfs_ax_P_uncorr.set_xlim(np.min(t), np.max(t))
            wfs_ax_A_uncorr.set_xlim(np.min(t), np.max(t))
            wfs_ax_P_corr.set_xlim(np.min(t), np.max(t))
            wfs_ax_A_corr.set_xlim(np.min(t), np.max(t))
            wfs_ax_Z.set_ylim(-1.1*max_amp, 1.1*max_amp)
            wfs_ax_N.set_ylim(-1.1*max_amp, 1.1*max_amp)
            wfs_ax_E.set_ylim(-1.1*max_amp, 1.1*max_amp)
            wfs_ax_P_uncorr.set_ylim(-1.1*max_amp, 1.1*max_amp)
            wfs_ax_A_uncorr.set_ylim(-1.1*max_amp, 1.1*max_amp)
            wfs_ax_P_corr.set_ylim(-1.1*max_amp, 1.1*max_amp)
            wfs_ax_A_corr.set_ylim(-1.1*max_amp, 1.1*max_amp)
            wfs_ax_Z.legend()

            # Fast and slow, pre and post correction:
            fs_start_end_times = [self.event_station_win_idxs[station]['win_start_idxs'][-1] / fs,
                                    self.event_station_win_idxs[station]['win_end_idxs'][0] / fs]
            fs_before_ax.plot(t, st_ZNE_curr_sws_corrected.select(channel="??F")[0].data, c='k')
            fs_before_ax.plot(t, st_ZNE_curr_sws_corrected.select(channel="??S")[0].data, c='k', ls='--')
            fs_after_ax.plot(t, np.roll(st_ZNE_curr_sws_corrected.select(channel="??F")[0].data, int(fs * dt_curr/2)), c='k')
            fs_after_ax.plot(t, np.roll(st_ZNE_curr_sws_corrected.select(channel="??S")[0].data, -int(fs * dt_curr/2)), c='k', ls='--')
            fs_before_ax.set_xlim(fs_start_end_times)
            fs_after_ax.set_xlim(fs_start_end_times)
            fs_before_ax.set_xlabel("Time (s)")
            fs_after_ax.set_xlabel("Time (s)")
            fs_before_ax.set_ylabel("F/S amp.")

            # Uncorr NE:
            ne_uncorr_ax.plot(st_ZNE_curr.select(channel="??E")[0].data, st_ZNE_curr.select(channel="??N")[0].data, c='k')
            ne_uncorr_ax.set_xlim(-1.1*max_amp, 1.1*max_amp)
            ne_uncorr_ax.set_ylim(-1.1*max_amp, 1.1*max_amp)
            # Plot intermediate post layer 2 correction (multi-layer splitting):
            if self.sws_multi_layer_result_df is not None:
                ne_uncorr_ax.plot(st_ZNE_curr_sws_corrected_layer_2.select(channel="??E")[0].data, 
                                    st_ZNE_curr_sws_corrected_layer_2.select(channel="??N")[0].data, c='#0078B8', alpha=0.5)
            # Corr NE:
            ne_corr_ax.plot(st_ZNE_curr_sws_corrected.select(channel="??E")[0].data, 
                                                            st_ZNE_curr_sws_corrected.select(channel="??N")[0].data, c='#D73215')
            ne_corr_ax.set_xlim(-1.1*max_amp, 1.1*max_amp)
            ne_corr_ax.set_ylim(-1.1*max_amp, 1.1*max_amp)

            # phi - dt space:
            # Plot for single layer:
            if self.sws_multi_layer_result_df is None:
                Y, X = np.meshgrid(self.phis_labels, self.lags_labels)
                Z = self.phi_dt_grid_average[station]
                # phi_dt_ax.contourf(X, Y, Z, levels=10, cmap="magma")
                CS = phi_dt_ax.contourf(X, Y, Z, levels=20, cmap="magma")
                CS2 = phi_dt_ax.contour(CS, levels=CS.levels[1::2], colors='w', alpha=0.25)
                phi_dt_ax.errorbar(dt_curr , phi_curr, xerr=dt_err_curr, yerr=phi_err_curr, c='g', capsize=5)
            # Or plot for double layer:
            else:
                # Setup new axes:
                phi_dt_ax_layer1 = phi_dt_ax.inset_axes([0, 0, 1.0, 0.5])#([0, 0, 0.5, 1.0])
                phi_dt_ax_layer2 = phi_dt_ax.inset_axes([0, 0.5, 1.0, 0.5])#([0.5, 0, 0.5, 1.0])
                # And plot data:
                Y, X = np.meshgrid(self.phis_labels, self.lags_labels)
                Z1 = self.phi_dt_grid_average_layer1[station]
                Z2 = self.phi_dt_grid_average_layer2[station]
                CS = phi_dt_ax_layer1.contourf(X, Y, Z1, levels=20, cmap="magma")
                CS2 = phi_dt_ax_layer1.contour(CS, levels=CS.levels[1::2], colors='w', alpha=0.25)
                phi_dt_ax_layer1.errorbar(dt_layer1 , phi_layer1, xerr=dt_err_layer1, yerr=phi_err_layer1, c='g', capsize=5)  
                phi_dt_ax_layer1.set_xlim(self.lags_labels[0], self.lags_labels[-1])
                phi_dt_ax_layer1.set_ylim(self.phis_labels[0], self.phis_labels[-1])
                CS = phi_dt_ax_layer2.contourf(X, Y, Z2, levels=20, cmap="magma")
                CS2 = phi_dt_ax_layer2.contour(CS, levels=CS.levels[1::2], colors='w', alpha=0.25)
                phi_dt_ax_layer2.errorbar(dt_layer2 , phi_layer2, xerr=dt_err_layer2, yerr=phi_err_layer2, c='g', capsize=5)
                phi_dt_ax_layer2.set_xlim(self.lags_labels[0], self.lags_labels[-1])
                phi_dt_ax_layer2.set_ylim(self.phis_labels[0], self.phis_labels[-1])
                # And sort axes:
                phi_dt_ax_layer1.set_xlabel(r'$\delta t_{layer 1,2}$ ($s$)')
                # phi_dt_ax_layer2.set_xlabel(r'$\delta t_{layer 2}$ ($s$)')
                phi_dt_ax_layer2.get_xaxis().set_visible(False)
                phi_dt_ax_layer1.set_ylabel(r'$\phi_{layer 1}$ from Q ($^o$)')
                phi_dt_ax_layer2.set_ylabel(r'$\phi_{layer 2}$ from Q ($^o$)')
                # phi_dt_ax_layer2.get_yaxis().set_visible(False)

            # Add clustering data if available:
            if self.clustering_info:
                # Plot clustering for multi-layer explicit result, if exists:
                if "layer1" in self.clustering_info[station]:
                    clust_idxs = list(self.clustering_info[station]['layer1']['clusters_dict'].keys())
                    samp_idx_count = 0
                    for clust_idx in clust_idxs:
                        samp_idx_count_overall = samp_idx_count
                        for layer_id in ['layer1', 'layer2']:
                            samp_idx_count = samp_idx_count_overall
                            # PLot cluster phis:
                            y_tmp = self.clustering_info[station][layer_id]['clusters_dict'][clust_idx]['phis']
                            x_tmp = np.arange(samp_idx_count, samp_idx_count+len(y_tmp))
                            y_err = self.clustering_info[station][layer_id]['clusters_dict'][clust_idx]['phi_errs']
                            cluster_results_ax_phi.errorbar(x_tmp, y_tmp, yerr=y_err, markersize=2.5, alpha=0.5)
                            # And plot cluster dts:
                            y_tmp = self.clustering_info[station][layer_id]['clusters_dict'][clust_idx]['lags']
                            x_tmp = np.arange(samp_idx_count, samp_idx_count+len(y_tmp))
                            y_err = self.clustering_info[station][layer_id]['clusters_dict'][clust_idx]['lag_errs']
                            cluster_results_ax_dt.errorbar(x_tmp, y_tmp, yerr=y_err, markersize=2.5, alpha=0.5)                    
                            # And update x sample count:
                            samp_idx_count = samp_idx_count + len(y_tmp)
                # Or plot single clustering, if not explicit multi-layer result:
                else:    
                    clust_idxs = list(self.clustering_info[station]['clusters_dict'].keys())
                    samp_idx_count = 0
                    for clust_idx in clust_idxs:
                        # PLot cluster phis:
                        y_tmp = self.clustering_info[station]['clusters_dict'][clust_idx]['phis']
                        x_tmp = np.arange(samp_idx_count, samp_idx_count+len(y_tmp))
                        y_err = self.clustering_info[station]['clusters_dict'][clust_idx]['phi_errs']
                        cluster_results_ax_phi.errorbar(x_tmp, y_tmp, yerr=y_err, c='k', markersize=2.5, alpha=0.5)
                        # And plot cluster dts:
                        y_tmp = self.clustering_info[station]['clusters_dict'][clust_idx]['lags']
                        x_tmp = np.arange(samp_idx_count, samp_idx_count+len(y_tmp))
                        y_err = self.clustering_info[station]['clusters_dict'][clust_idx]['lag_errs']
                        cluster_results_ax_dt.errorbar(x_tmp, y_tmp, yerr=y_err, c='k', markersize=2.5, alpha=0.5)                    
                        # And update x sample count:
                        samp_idx_count = samp_idx_count + len(y_tmp)
                # And set limits and labels:
                cluster_results_ax_phi.set_ylim(-90, 90)
                cluster_results_ax_dt.set_ylim(0, np.max(self.lags_labels))
                cluster_results_ax_dt.set_xlabel("Cluster sample")
                cluster_results_ax_phi.set_ylabel(r"$\phi$ ($^o$)")
                cluster_results_ax_dt.set_ylabel(r"$\delta t$ ($s$)")

            # Add text:
            text_ax.text(0,0,"Event origin time : \n"+self.origin_time.strftime("%Y-%m-%dT%H:%M:%SZ"), fontsize='small')
            text_ax.text(0,-1,"Station : "+station, fontsize='small')
            # Plot intermediate post layer 2 correction (multi-layer splitting):
            if self.sws_multi_layer_result_df is not None:
                text_ax.text(0,-2,r"$\delta$ $t_{layer 1}$ : "+str(dt_layer1)+" +/-"+str(round(dt_err_layer1, 5))+" $s$", fontsize='small')
                text_ax.text(0,-3,r"$\delta$ $t_{layer 2}$ : "+str(dt_layer2)+" +/-"+str(round(dt_err_layer2, 5))+" $s$", fontsize='small')
                text_ax.text(0,-4,r"$\phi_{layer 1}$ from N : "+"{0:0.1f}".format(phi_layer1)+"$^o$"+" +/-"+"{0:0.1f}".format(phi_err_layer1)+"$^o$", fontsize='small')
                text_ax.text(0,-5,r"$\phi_{layer 2}$ from N : "+"{0:0.1f}".format(phi_layer2)+"$^o$"+" +/-"+"{0:0.1f}".format(phi_err_layer2)+"$^o$", fontsize='small')
                text_ax.text(0,-6,''.join(("src pol from N: ","{0:0.1f}".format(src_pol_curr),"$^o$"," +/-","{0:0.1f}".format(src_pol_err_curr),"$^o$")), fontsize='small')
                text_ax.text(0,-7,"Coord. sys. : "+self.coord_system, fontsize='small')
                text_ax.text(0,-8,r"$\lambda_2$/$\lambda_1$: "+str(round(opt_eig_ratio_curr, 3)), fontsize='small')
                if Q_w_curr <= 1.1:
                    text_ax.text(0,-9,"$Q_w$ : "+str(round(Q_w_curr, 3)), fontsize='small')
            else:
                text_ax.text(0,-2,r"$\delta$ $t$ : "+str(dt_curr)+" +/-"+str(round(dt_err_curr, 5))+" $s$", fontsize='small')
                text_ax.text(0,-3,r"$\phi$ from N : "+"{0:0.1f}".format(phi_from_N_curr)+"$^o$"+" +/-"+"{0:0.1f}".format(phi_err_curr)+"$^o$", fontsize='small')
                text_ax.text(0,-4,''.join(("src pol from N: ","{0:0.1f}".format(src_pol_curr),"$^o$"," +/-","{0:0.1f}".format(src_pol_err_curr),"$^o$")), fontsize='small')
                text_ax.text(0,-5,"Coord. sys. : "+self.coord_system, fontsize='small')
                text_ax.text(0,-6,r"$\lambda_2$/$\lambda_1$: "+str(round(opt_eig_ratio_curr, 3)), fontsize='small')
                if Q_w_curr <= 1.1:
                    text_ax.text(0,-7,"$Q_w$ : "+str(round(Q_w_curr, 3)), fontsize='small')
            text_ax.set_xlim(-2,10)
            text_ax.set_ylim(-10,2)

            # And do some plot labelling:
            wfs_ax_Z.set_ylabel("Z amp.")
            wfs_ax_N.set_ylabel("N amp.")
            wfs_ax_E.set_ylabel("E amp.")
            wfs_ax_P_uncorr.set_ylabel(''.join(("P amp.\n(", "{0:0.1f}".format(src_pol_curr), "$^o$)")))
            wfs_ax_A_uncorr.set_ylabel(''.join(("A amp.\n(", "{0:0.1f}".format(src_pol_curr+90.), "$^o$)")))
            wfs_ax_P_corr.set_ylabel(''.join(("P amp.\n(", "{0:0.1f}".format(src_pol_curr), "$^o$)")))
            wfs_ax_A_corr.set_ylabel(''.join(("A amp.\n(", "{0:0.1f}".format(src_pol_curr+90.), "$^o$)")))
            wfs_ax_A_corr.set_xlabel("Time (s)")
            ne_uncorr_ax.set_xlabel('E')
            ne_uncorr_ax.set_ylabel('N')
            ne_corr_ax.set_xlabel('E')
            ne_corr_ax.set_ylabel('N')
            phi_dt_ax.set_xlabel(r'$\delta$ t (s)')
            if self.sws_multi_layer_result_df is None:
                phi_dt_ax.set_ylabel(r'$\phi$ from Q ($^o$)')
            else:
                phi_dt_ax.set_ylabel(r'$\phi_{layer 1,2}$ from Q ($^o$)')
                phi_dt_ax.get_xaxis().set_visible(False)
                phi_dt_ax.get_yaxis().set_visible(False)

            # plt.colorbar()
            # plt.tight_layout()
            if outdir:
                os.makedirs(outdir, exist_ok=True)
                plt.savefig(os.path.join(outdir, ''.join((self.event_uid, "_", station, ".png"))), dpi=300)
            if suppress_direct_plotting:
                plt.ioff()
                plt.close(fig)
            else:
                plt.show()
    

    def save_result(self, outdir=os.getcwd()):
        """Function to save output. Output is a csv file with all the splitting data for the event, 
        for all stations. Saves result as <event_uid>, to <outdir>."""
        # Create outdir, if doesn't exist:
        os.makedirs(outdir, exist_ok=True)
        # if not os.path.exists(outdir):
        #     os.makedirs(outdir)
        # ANd write data to file:
        fname_out = os.path.join(outdir, ''.join((self.event_uid, "_sws_result.csv")))
        self.sws_result_df.to_csv(fname_out, index=False)
        print("Saved sws result to:", fname_out)
    

    def save_wfs(self, outdir=os.getcwd()):
        """Function to save waveforms outputs. Outputs are unccorrected and corrected waveforms for 
        all events. Saves result as <event_uid>.mseed, to <outdir>."""
        # Create outdir, if doesn't exist:
        os.makedirs(outdir, exist_ok=True)
        # if not os.path.exists(outdir):
        #     os.makedirs(outdir)
        # Loop over stations, appending data to streams:
        st_uncorr_out = obspy.Stream()
        st_corr_out = obspy.Stream()
        for station in self.stations_list:
            # Get waveforms:
            try:
                st_ZNE_curr, st_ZNE_curr_sws_corrected = self._get_uncorr_and_corr_waveforms(station)
            except CustomError:
                continue
            # And append to output streams:
            for tr in st_ZNE_curr:
                st_uncorr_out.append(tr)
            for tr in st_ZNE_curr_sws_corrected:
                st_corr_out.append(tr)    
        # And write out:
        try:
            st_uncorr_out.write(os.path.join(outdir, ''.join((self.event_uid, "_wfs_uncorr.mseed"))), format="MSEED")
            st_corr_out.write(os.path.join(outdir, ''.join((self.event_uid, "_wfs_corr.mseed"))), format="MSEED")
            print("Saved sws wfs to:", os.path.join(outdir, ''.join((self.event_uid, "_wfs_*.mseed"))))
        except obspy.core.util.obspy_types.ObsPyException:
            print("Warning: Empty stream therefore unable to write to file. Continueing.")
            pass 

            
    
    
