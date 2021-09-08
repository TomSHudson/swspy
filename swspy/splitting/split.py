#-----------------------------------------------------------------------------------------------------------------------------------------

# Script Description:

# Input variables:

# Output variables:

# Created by Tom Hudson, 30th July 2021

#-----------------------------------------------------------------------------------------------------------------------------------------

# Import neccessary modules:
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import pandas as pd 
from numba import jit
from scipy import stats, interpolate
from sklearn import cluster
from sklearn.cluster import DBSCAN, KMeans, AgglomerativeClustering
import obspy
from obspy import UTCDateTime as UTCDateTime
import sys, os
import glob
import matplotlib.pyplot as plt
import subprocess
import gc
from NonLinLocPy import read_nonlinloc # For reading NonLinLoc data (can install via pip)

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
    Note: Only works for a single station, as only supply a single back azimuth."""
    # Rotate LQT to BPA:
    st_BPA = st_LQT.copy()
    # Convert back-azimuth to radians:
    back_azi_rad = back_azi * np.pi / 180.
    # Define rotation matrix (for counter-clockwise rotation):
    rot_matrix = np.array([[np.cos(back_azi_rad), -np.sin(back_azi_rad)], [np.sin(back_azi_rad), np.cos(back_azi_rad)]])
    # And perform the rotation (y = Q, P in this case, x = T, A in this case):
    vec = np.vstack((st_LQT.select(channel="??T")[0].data, st_LQT.select(channel="??Q")[0].data))
    vec_rot = np.dot(rot_matrix, vec)
    st_BPA.select(channel="??T")[0].data = np.array(vec_rot[0,:])
    st_BPA.select(channel="??Q")[0].data = np.array(vec_rot[1,:])
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
    # Rotate BPA to LQT (just inverse angle of LQT -> BPA):
    st_LQT = _rotate_LQT_to_BPA(st_BPA, -back_azi)
    return st_LQT


def _rotate_QT_comps(data_arr_Q, data_arr_T, rot_angle_rad):
    """
    Function to rotate Q (SV polarisation) and T (SH polarisation) components by a rotation angle <rot_angle_rad>,
    in radians. Output is a set of rotated traces in a rotated coordinate system about 
    degrees clockwise from Q.
    """
    # Convert angle from counter-clockwise from x to clockwise:
    theta_rot = -rot_angle_rad #-1. * (rot_angle_rad + (np.pi / 2. ))
    # Define rotation matrix:
    rot_matrix = np.array([[np.cos(theta_rot), -np.sin(theta_rot)], [np.sin(theta_rot), np.cos(theta_rot)]])
    # And perform the rotation:
    vec = np.vstack((data_arr_T, data_arr_Q))
    vec_rot = np.dot(rot_matrix, vec)
    data_arr_Q_rot = np.array(vec_rot[1,:])
    data_arr_T_rot = np.array(vec_rot[0,:])
    return data_arr_Q_rot, data_arr_T_rot


def remove_splitting(st_ZNE_uncorr, phi, dt, back_azi, event_inclin_angle_at_station):
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

    Returns
    -------
    st_LQT_corr : obspy stream object
        Corrected data
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
    st_BPA_uncorr = _rotate_LQT_to_BPA(st_LQT_uncorr, back_azi)
    # Perform SWS correction:
    x_in, y_in = st_BPA_uncorr.select(channel="??T")[0].data, st_LQT_uncorr.select(channel="??Q")[0].data
    fs = st_BPA_uncorr.select(channel="??T")[0].stats.sampling_rate
    # 1. Rotate data into splitting coordinates:
    y, x = _rotate_QT_comps(y_in, x_in, phi * np.pi/180)
    # 2. Apply reverse time shift to Q and T data:
    x = np.roll(x, -int((dt / 2) * fs))
    y = np.roll(y, int((dt / 2) * fs))
    # 3. And rotate back to QT (PA) coordinates:
    y, x = _rotate_QT_comps(y, x, -phi * np.pi/180)
    # And put data back in stream form:
    st_BPA_corr = st_BPA_uncorr.copy()
    st_BPA_corr.select(channel="??T")[0].data = x 
    st_BPA_corr.select(channel="??Q")[0].data = y
    # And rotate back into ZNE coords:
    st_LQT_corr = _rotate_BPA_to_LQT(st_BPA_corr, back_azi)
    st_ZNE_corr = _rotate_LQT_to_ZNE(st_LQT_corr, back_azi, event_inclin_angle_at_station)
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
        Confidence level (e.g. if alpha = 0.05, then 
        95% confidence level found).
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



@jit(nopython=True)
def _phi_dt_grid_search(data_arr_Q, data_arr_T, win_start_idxs, win_end_idxs, n_t_steps, n_angle_steps, n_win, fs, rotate_step_deg, grid_search_results_all_win):
    """Function to do numba accelerated grid search of phis and dts."""

    # Loop over start and end windows:
    for a in range(n_win):
        start_win_idx = win_start_idxs[a]
        for b in range(n_win):
            end_win_idx = win_end_idxs[b]
            grid_search_idx = int(n_win*a + b)

            # Loop over angles:
            for j in range(n_angle_steps):
                angle_shift_rad_curr = ((j * rotate_step_deg) - 90.) * np.pi / 180. # (Note: -90 as should loop between -90 and 90 (see phi_labels))

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

                    # Calculate eigenvalues:
                    xy_arr = np.vstack((rolled_rot_T_curr, rolled_rot_Q_curr))
                    lambdas_unsort = np.linalg.eigvalsh(np.cov(xy_arr))
                    lambdas = np.sort(lambdas_unsort)

                    # And save result to datastores:
                    # Note: Use lambda2 divided by lambda1 as in Wuestefeld2010 (most stable)
                    grid_search_results_all_win[grid_search_idx,i,j] = lambdas[0] / lambdas[1]
   
    return grid_search_results_all_win



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


    overall_win_start_post_fast_S_pick : float (default = 0.5 s)
        Overall window start time in seconds after S pick.
    
    win_S_pick_tolerance : 

    rotate_step_deg : 

    n_win : 


    Methods
    -------
    perform_sws_analysis : Function to perform shear-wave splitting analysis.

    """

    def __init__(self, st, nonlinloc_event_path):
        """Initiate the class object.

        Parameters
        ----------
        st : obspy Stream object
            Obspy stream containing all the waveform data for all the stations and 
            channels to perform the splitting on.

        nonlinloc_event_path : str
            Path to NonLinLoc .grid0.loc.hyp file for the event.
        """
        # Define parameters:
        self.st = st 
        self.nonlinloc_hyp_data = read_nonlinloc.read_hyp_file(nonlinloc_event_path)
        self.nonlinloc_event_path = nonlinloc_event_path
        self.origin_time = self.nonlinloc_hyp_data.origin_time
        # Define attributes:
        self.overall_win_start_pre_fast_S_pick = 0.1
        self.overall_win_start_post_fast_S_pick = 0.2
        self.win_S_pick_tolerance = 0.1
        self.rotate_step_deg = 2.0
        self.max_t_shift_s = 0.1
        self.n_win = 10

    def _select_windows(self):
        """
        Function to specify window start and end time indices.
        """
        start_idxs_range = int((self.overall_win_start_pre_fast_S_pick - self.win_S_pick_tolerance) * self.fs) 
        win_start_idxs = np.linspace(0, start_idxs_range, num=self.n_win, dtype=int) #np.random.randint(start_idxs_range, size=self.n_win)
        end_idxs_start = int((self.overall_win_start_pre_fast_S_pick + self.overall_win_start_post_fast_S_pick) * self.fs) 
        win_end_idxs = np.linspace(end_idxs_start, end_idxs_start + start_idxs_range, num=self.n_win, dtype=int) #np.random.randint(end_idxs_start, end_idxs_start + start_idxs_range, size=self.n_win)
        return win_start_idxs, win_end_idxs
    

    def _calc_splitting_eig_val_method(self, data_arr_Q, data_arr_T, win_start_idxs, win_end_idxs):
        """
        Function to calculate splitting via eigenvalue method.
        """
        # Setup paramters:
        n_t_steps = int(self.max_t_shift_s * self.fs)
        n_angle_steps = int(180. / self.rotate_step_deg) + 1
        n_win = self.n_win
        fs = self.fs
        rotate_step_deg = self.rotate_step_deg

        # Setup datastores:
        grid_search_results_all_win = np.zeros((n_win**2, n_t_steps, n_angle_steps), dtype=float)
        lags_labels = np.arange(0., n_t_steps, 1) / fs 
        phis_labels = np.arange(-90, 90 + rotate_step_deg, rotate_step_deg)

        # Perform grid search:
        # (note that numba accelerated, hence why function outside class)
        grid_search_results_all_win = _phi_dt_grid_search(data_arr_Q, data_arr_T, win_start_idxs, win_end_idxs, n_t_steps, n_angle_steps, 
                                                        n_win, fs, rotate_step_deg, grid_search_results_all_win)
                        
        return grid_search_results_all_win, lags_labels, phis_labels


    def _get_phi_and_lag_errors(self, phi_dt_single_win, lags_labels, phis_labels, tr_for_dof, interp_fac=4):
        """
        Finds the error associated with phi and lag for a given grid search window result.
        Returns errors in phi and lag.
        Calculates errors based on the Silver and Chan (1991) method using the 95% confidence 
        interval, found using an f-test. (Or 90% ?!)
        """
        # Define the error surface to work with:
        # (Done explicitely simply so that can change easily)
        if interp_fac > 1:
            interp_spline = interpolate.RectBivariateSpline(lags_labels, phis_labels, phi_dt_single_win)
            x_tmp = lags_labels
            y_tmp = phis_labels
            error_surf = interp_spline(np.linspace(x_tmp[0], x_tmp[-1], interp_fac*len(x_tmp)), np.linspace(y_tmp[0], y_tmp[-1], interp_fac*len(y_tmp)))
        else:
            error_surf = phi_dt_single_win
        
        # Find grid search array points where within confidence interval:
        # Use transverse component to calculate dof
        dof = calc_dof(tr_for_dof.data)
        conf_bound = ftest(phi_dt_single_win, dof, alpha=0.05, k=2)
        conf_mask = error_surf <= conf_bound

        # Find lag dt error:
        # (= 1/4 width of confidence box (see Silver1991))
        lag_mask = conf_mask.any(axis=0)
        true_idxs = np.where(lag_mask)[0]
        lag_step_s = lags_labels[1] - lags_labels[0]
        lag_err = (true_idxs[-1] - true_idxs[0] + 1) * lag_step_s * 0.25

        # Find fast direction phi error:
        # (= 1/4 width of confidence box (see Silver1991))
        # (Note: This must deal with angle symetry > 90 or < -90)
        phi_mask = conf_mask.any(axis=1)
        phi_mask_with_pos_neg_overlap = np.hstack((phi_mask,phi_mask, phi_mask))
        max_false_len = np.diff(np.where(phi_mask_with_pos_neg_overlap)).max() - 1
        # shortest line that contains ALL true values is then:
        max_true_len = len(phi_mask) - max_false_len
        phi_step_deg = phis_labels[1] - phis_labels[0]
        phi_err = max_true_len * phi_step_deg * 0.25

        return phi_err, lag_err 

    
    def _sws_win_clustering(self, lags, phis, lag_errs, phi_errs, method="dbscan"):
        """Function to perform sws clustering of phis and lags. This clustering is based on the method of 
        Teanby2004, except that this function uses new coordinate system to deal with the cyclic nature  
        of phi about -90,90, and therefore uses a different clustering algorithm (dbscan) to perform 
        optimal clustering within this new space.
        Note: Performs analysis on normallised lag data.
        """
        # Weight samples by their error variances:
        # samples_weights = 1. - ((lag_errs/lags)**2 + (phi_errs/phis)**2) # (= 1 - (var_lag_norm + var_phi_norm))
        
        # Convert phis and lags into new coordinate system:
        samples_new_coords =  np.dstack(( ( lags / np.max(lags) ) * np.cos(2 * phis), ( lags / np.max(lags) ) * np.sin(2 * phis) ))[0,:,:]

        # And perform clustering:
        # ward = AgglomerativeClustering(n_clusters=None, linkage='ward',distance_threshold=0.25)
        # ward.fit(samples_new_coords)#, sample_weight=samples_weights)
        db = DBSCAN(eps=0.25, min_samples=int(np.sqrt(len(lags))))
        clustering = db.fit(samples_new_coords)#, sample_weight=samples_weights)
        # Separate samples into clusters:
        n_clusters = len(set(clustering.labels_)) - (1 if -1 in clustering.labels_ else 0) # Note: -1 are noise coords
        clusters_dict = {}
        for i in range(n_clusters):
            curr_cluster_idxs = np.where(clustering.labels_ == i)[0]
            clusters_dict[str(i)] = {}
            clusters_dict[str(i)]['lags'] = lags[curr_cluster_idxs]
            clusters_dict[str(i)]['lag_errs'] = lag_errs[curr_cluster_idxs]
            clusters_dict[str(i)]['phis'] = phis[curr_cluster_idxs]
            clusters_dict[str(i)]['phi_errs'] = phi_errs[curr_cluster_idxs]
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

        return opt_phi, opt_lag, opt_phi_err, opt_lag_err

    
    def _rot_phi_from_sws_coords_to_deg_from_N(self, phi_pre_rot_deg, back_azi):
        """Function to rotate phi from LQT splitting to degrees from N."""
        phi_rot_deg = phi_pre_rot_deg + back_azi
        if phi_rot_deg > 360.:
            phi_rot_deg = phi_rot_deg - 360.
        return phi_rot_deg 


    def perform_sws_analysis(self, coord_system="LQT"):
        """Function to perform splitting analysis. Works in LQT coordinate system 
        as then performs shear-wave-splitting in 3D.
        
        Parameters
        ----------
        coord_system : str
            Coordinate system to perform analysis in. Options are: LQT, ZNE. Will convert 
            splitting angles back into coordinates relative to ZNE whatever system it 
            performs the splitting within. Default = LQT. 

        """
        # Save any parameters to class object:
        self.coord_system = coord_system

        # Create datastores:
        self.sws_result_df = pd.DataFrame(data={'station': [], 'phi': [], 'phi_from_N': [], 'phi_err': [], 'dt': [], 'dt_err': []})
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
                elif self.coord_system == "ZNE":
                    event_inclin_angle_at_station = 0. # Rotates ray to arrive at vertical incidence, simulating NE components.
                else:
                    print("Error: coord_system =", self.coord_system, "not supported. Exiting.")
                    sys.exit()
            except KeyError:
                print("No S phase pick for station:", station, "therefore skipping this station.")
                continue
            # And rotate into emerging ray coord system, LQT:
            st_LQT_curr = _rotate_ZNE_to_LQT(st_ZNE_curr, back_azi, event_inclin_angle_at_station)
            # And rotate into propagation coordinate system (as in Walsh et al. (2013)), BPA:
            st_BPA_curr = _rotate_LQT_to_BPA(st_LQT_curr, back_azi)
            del st_ZNE_curr, st_LQT_curr
            gc.collect()

            # 2. Get horizontal channels and trim to pick:
            st_BPA_curr.trim(starttime=self.nonlinloc_hyp_data.phase_data[station]['S']['arrival_time'] - self.overall_win_start_pre_fast_S_pick,
                                endtime=self.nonlinloc_hyp_data.phase_data[station]['S']['arrival_time'] + self.overall_win_start_post_fast_S_pick 
                                            + self.max_t_shift_s)
            tr_P = st_BPA_curr.select(station=station, channel="??Q")[0]
            tr_A = st_BPA_curr.select(station=station, channel="??T")[0]

            # 3. Get window indices:
            self.fs = tr_A.stats.sampling_rate
            win_start_idxs, win_end_idxs = self._select_windows()

            # 4. Calculate splitting angle and delay times for windows:
            # (Silver and Chan (1991) and Teanby2004 eigenvalue method)
            # 4.a. Get data for all windows:
            grid_search_results_all_win, lags_labels, phis_labels = self._calc_splitting_eig_val_method(tr_P.data, tr_A.data, win_start_idxs, win_end_idxs)
            self.grid_search_results_all_win = grid_search_results_all_win
            self.lags_labels = lags_labels 
            self.phis_labels = phis_labels 
            # 4.b. Get lag and phi values and errors associated with windows:
            lags = np.zeros(grid_search_results_all_win.shape[0])
            phis = np.zeros(grid_search_results_all_win.shape[0])
            lag_errs = np.zeros(grid_search_results_all_win.shape[0])
            phi_errs = np.zeros(grid_search_results_all_win.shape[0])
            for i in range(grid_search_results_all_win.shape[0]):
                # Only use positive grid search window!:
                first_pos_idx = np.where(self.lags_labels >= 0.)[0][0]
                grid_search_result_curr_win = grid_search_results_all_win[i,first_pos_idx:,:]
                # Get lag and phi:
                min_idxs = np.where(grid_search_result_curr_win == np.min(grid_search_result_curr_win)) 
                lags[i] = self.lags_labels[min_idxs[0][0] + first_pos_idx] # + term to deal with positive shift
                phis[i] = self.phis_labels[min_idxs[1][0]]
                # Get associated error (from f-test with 95% confidence interval):
                # (Note: Uses transverse trace for dof estimation (see Silver and Chan 1991))
                phi_errs[i], lag_errs[i] = self._get_phi_and_lag_errors(grid_search_result_curr_win, self.lags_labels[first_pos_idx:], phis_labels, tr_A)

            # 6. Perform clustering for all windows to find best result:
            # (Teanby2004 method, but in new coordinate space with dbscan clustering)
            opt_phi, opt_lag, opt_phi_err, opt_lag_err = self._sws_win_clustering(lags, phis, lag_errs, phi_errs, method="dbscan")

            # 7. Rotate output phi back into angle relative to N:
            opt_phi_from_N = self._rot_phi_from_sws_coords_to_deg_from_N(opt_phi, back_azi)

            # 8. And append data to overall datastore:
            df_tmp = pd.DataFrame(data={'station': [station], 'phi': [opt_phi], 'phi_from_N': [opt_phi_from_N], 'phi_err': [opt_phi_err], 'dt': [opt_lag], 'dt_err': [opt_lag_err]})
            self.sws_result_df = self.sws_result_df.append(df_tmp)
            opt_phi_idx = np.where(self.phis_labels == opt_phi)[0][0]
            opt_lag_idx = np.where(self.lags_labels == opt_lag)[0][0]
            self.phi_dt_grid_average[station] = np.average(grid_search_results_all_win, axis=0)
            self.event_station_win_idxs[station] = {}
            self.event_station_win_idxs[station]['win_start_idxs'] = win_start_idxs
            self.event_station_win_idxs[station]['win_end_idxs'] = win_end_idxs

            # ???. Apply automation approach of Wuestefeld2010 ?!?

        return self.sws_result_df



    def plot(self, out_fname=None):
        """Function to perform plotting...
        """
        # Loop over stations, plotting:
        for station in self.stations_list:
            # Get data:
            # Waveforms:
            st_ZNE_curr = self.st.select(station=station).copy()
            try:
                back_azi = self.nonlinloc_hyp_data.phase_data[station]['S']['SAzim'] + 180.
                if back_azi >= 360.:
                    back_azi = back_azi - 360.
                if self.coord_system == "LQT":
                    event_inclin_angle_at_station = self.nonlinloc_hyp_data.phase_data[station]['S']['RDip']
                elif self.coord_system == "ZNE":
                    event_inclin_angle_at_station = 0. # Rotates ray to arrive at vertical incidence, simulating NE components.
                else:
                    print("Error: coord_system =", self.coord_system, "not supported. Exiting.")
                    sys.exit()
            except KeyError:
                print("No S phase pick for station:", station, "therefore skipping this station.")
                continue
            st_ZNE_curr.trim(starttime=self.nonlinloc_hyp_data.phase_data[station]['S']['arrival_time'] - self.overall_win_start_pre_fast_S_pick,
                    endtime=self.nonlinloc_hyp_data.phase_data[station]['S']['arrival_time'] + self.overall_win_start_post_fast_S_pick 
                                + self.max_t_shift_s)

            # And remove splitting:
            phi_curr = float(self.sws_result_df.loc[self.sws_result_df['station'] == station]['phi'])
            dt_curr = float(self.sws_result_df.loc[self.sws_result_df['station'] == station]['dt'])
            phi_err_curr = float(self.sws_result_df.loc[self.sws_result_df['station'] == station]['phi_err'])
            dt_err_curr = float(self.sws_result_df.loc[self.sws_result_df['station'] == station]['dt_err'])
            st_ZNE_curr_sws_corrected = remove_splitting(st_ZNE_curr, phi_curr, dt_curr, back_azi, event_inclin_angle_at_station)

            # Plot data:
            # Setup figure:
            fig = plt.figure(constrained_layout=True, figsize=(8,6))
            gs = fig.add_gridspec(2, 3)
            wfs_ax = fig.add_subplot(gs[0, 0:2])
            wfs_ax.get_xaxis().set_visible(False)
            wfs_ax.get_yaxis().set_visible(False)
            # wfs_ax_Z = inset_axes(wfs_ax, width="100%", height="100%")#, bbox_to_anchor=(.7, .5, .3, .5))
            wfs_ax_Z = wfs_ax.inset_axes([0, 0.666666, 1.0, 0.333333])#wfs_ax, width="100%", height="30%", loc=1)#, bbox_to_anchor=(.7, .5, .3, .5))
            wfs_ax_N = wfs_ax.inset_axes([0, 0.333333, 1.0, 0.333333])#wfs_ax, width="100%", height="30%", loc=2)
            wfs_ax_E = wfs_ax.inset_axes([0, 0.0, 1.0, 0.333333])#wfs_ax, width="100%", height="30%", loc=3)
            text_ax = fig.add_subplot(gs[0, 2])
            text_ax.axis('off')
            ne_uncorr_ax = fig.add_subplot(gs[1, 0])
            ne_corr_ax = fig.add_subplot(gs[1, 1])
            phi_dt_ax = fig.add_subplot(gs[1, 2])

            # Plot data on figure:
            t = np.arange(len(st_ZNE_curr.select(channel="??Z")[0].data)) / st_ZNE_curr.select(channel="??Z")[0].stats.sampling_rate
            # Waveforms:
            max_amp = 0.
            max_amp = np.max(np.abs(st_ZNE_curr.select(channel="??Z")[0].data))
            if np.max(np.abs(st_ZNE_curr.select(channel="??N")[0].data)) > max_amp:
                max_amp = np.max(np.abs(st_ZNE_curr.select(channel="??N")[0].data))
            if np.max(np.abs(st_ZNE_curr.select(channel="??E")[0].data)) > max_amp:
                max_amp = np.max(np.abs(st_ZNE_curr.select(channel="??E")[0].data))
            wfs_ax_Z.plot(t, st_ZNE_curr.select(channel="??Z")[0].data, c='k')
            wfs_ax_N.plot(t, st_ZNE_curr.select(channel="??N")[0].data, c='k')
            wfs_ax_E.plot(t, st_ZNE_curr.select(channel="??E")[0].data, c='k')
            wfs_ax_Z.plot(t, st_ZNE_curr_sws_corrected.select(channel="??Z")[0].data, c='#D73215')
            wfs_ax_N.plot(t, st_ZNE_curr_sws_corrected.select(channel="??N")[0].data, c='#D73215')
            wfs_ax_E.plot(t, st_ZNE_curr_sws_corrected.select(channel="??E")[0].data, c='#D73215')
            fs = st_ZNE_curr.select(channel="??N")[0].stats.sampling_rate
            for i in range(len(self.event_station_win_idxs[station]['win_start_idxs'])):
                wfs_ax_N.axvline(x = self.event_station_win_idxs[station]['win_start_idxs'][i] / fs, c='k', alpha=0.25)
                wfs_ax_N.axvline(x = self.event_station_win_idxs[station]['win_end_idxs'][i] / fs, c='k', alpha=0.25)
                wfs_ax_E.axvline(x = self.event_station_win_idxs[station]['win_start_idxs'][i] / fs, c='k', alpha=0.25)
                wfs_ax_E.axvline(x = self.event_station_win_idxs[station]['win_end_idxs'][i] / fs, c='k', alpha=0.25)
            wfs_ax_Z.set_xlim(np.min(t), np.max(t))
            wfs_ax_N.set_xlim(np.min(t), np.max(t))
            wfs_ax_E.set_xlim(np.min(t), np.max(t))
            wfs_ax_Z.set_ylim(-1.1*max_amp, 1.1*max_amp)
            wfs_ax_N.set_ylim(-1.1*max_amp, 1.1*max_amp)
            wfs_ax_E.set_ylim(-1.1*max_amp, 1.1*max_amp)
            # Uncorr NE:
            ne_uncorr_ax.plot(st_ZNE_curr.select(channel="??E")[0].data, st_ZNE_curr.select(channel="??N")[0].data)
            ne_uncorr_ax.set_xlim(-1.1*max_amp, 1.1*max_amp)
            ne_uncorr_ax.set_ylim(-1.1*max_amp, 1.1*max_amp)
            # Corr NE:
            ne_corr_ax.plot(st_ZNE_curr_sws_corrected.select(channel="??E")[0].data, st_ZNE_curr_sws_corrected.select(channel="??N")[0].data)
            ne_corr_ax.set_xlim(-1.1*max_amp, 1.1*max_amp)
            ne_corr_ax.set_ylim(-1.1*max_amp, 1.1*max_amp)
            # phi - dt space:
            Y, X = np.meshgrid(self.phis_labels, self.lags_labels)
            Z = self.phi_dt_grid_average[station]
            phi_dt_ax.contourf(X, Y, Z, levels=10, cmap="magma_r")
            phi_dt_ax.errorbar(dt_curr , phi_curr, xerr=dt_err_curr, yerr=phi_err_curr, c='g')
            # Add text:
            text_ax.text(0,0,"Event origin time : \n"+str(self.origin_time), fontsize='small')
            text_ax.text(0,-1,"Station : "+station, fontsize='small')
            text_ax.text(0,-2,"$\phi_{QT coords}$ : "+str(phi_curr)+"$^o$"+" +/-"+str(phi_err_curr), fontsize='small')
            text_ax.text(0,-3,"$\phi$ ($^o$ from N) : "+str(round(float(self.sws_result_df.loc[self.sws_result_df['station'] == station]['phi_from_N']),1))+"$^o$"+" +/-"+str(phi_err_curr), fontsize='small')
            text_ax.text(0,-4,"$\delta$ $t$ : "+str(dt_curr)+"$s$"+" +/-"+str(round(dt_err_curr, 5)), fontsize='small')
            text_ax.text(0,-5,"Coord. sys. : "+self.coord_system, fontsize='small')
            text_ax.set_xlim(-2,10)
            text_ax.set_ylim(-10,2)


            # And do some plot labelling:
            wfs_ax_E.set_xlabel("Time (s)")
            wfs_ax_Z.set_ylabel("Z amp.")
            wfs_ax_N.set_ylabel("N amp.")
            wfs_ax_E.set_ylabel("E amp.")
            ne_uncorr_ax.set_xlabel('E')
            ne_uncorr_ax.set_ylabel('N')
            ne_corr_ax.set_xlabel('E')
            ne_corr_ax.set_ylabel('N')
            phi_dt_ax.set_xlabel('$\delta$ t (s)')
            phi_dt_ax.set_ylabel('$\phi$ ($^o$)')

            # plt.colorbar()
            plt.tight_layout()
            if out_fname:
                plt.savefig(out_fname, dpi=300)
            plt.show()

    
    
