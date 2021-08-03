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
import obspy
from obspy import UTCDateTime as UTCDateTime
import sys, os
import glob
import matplotlib.pyplot as plt
import subprocess
import gc
from NonLinLocPy import read_nonlinloc # For reading NonLinLoc data (can install via pip)


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
        # Define attributes:
        self.overall_win_start_pre_fast_S_pick = 0.1
        self.overall_win_start_post_fast_S_pick = 0.5
        self.win_S_pick_tolerance = 0.01
        self.rotate_step_deg = 2.0
        self.max_t_shift_s = 0.2
        self.n_win = 10

    def _select_windows(self):
        """
        Function to specify window start and end time indices.
        """
        start_idxs_range = int((self.overall_win_start_pre_fast_S_pick - self.win_S_pick_tolerance) * self.fs) 
        win_start_idxs = np.random.randint(start_idxs_range, size=self.n_win)
        end_idxs_start = int((self.overall_win_start_pre_fast_S_pick + self.overall_win_start_post_fast_S_pick) * self.fs) 
        win_end_idxs = np.random.randint(end_idxs_start, end_idxs_start + start_idxs_range, size=self.n_win)
        return win_start_idxs, win_end_idxs

    def _calc_splitting_eig_val_method(self, data_arr_N, data_arr_E, win_start_idxs, win_end_idxs):
        """
        Function to calculate splitting via eigenvalue method.
        """
        # Create data stores:
        n_t_steps = int(self.max_t_shift_s / self.fs)
        n_angle_steps = int(360. / self.rotate_step_deg)
        grid_search_results_all = np.zeros((self.n_win**2, n_t_steps, n_angle_steps), dtype=float)

        # Perform grid search:
        # Loop over start and end windows:
        for a in range(self.n_win):
            start_win_idx = win_start_idxs[a]
            for b in range(self.n_win):
                end_win_idx = win_start_idxs[b]
                grid_search_idx = int(self.n_win*a + b)
                # Loop over time shifts:
                for i in range(n_t_steps):
                    t_shift_curr = int( i - ( n_t_steps / 2.) )
                    # Loop over angles:
                    for j in range(n_angle_steps):
                        angle_shift_curr_rad = j * self.rotate_step_deg * np.pi / 180.
                        #HERE!!!


                    
                



    def perform_sws_analysis(self):
        """Function to perform splitting analysis."""
        # Loop over stations in stream:
        stations_list = []
        for tr in self.st: 
            if tr.stats.station not in stations_list:
                stations_list.append(tr.stats.station)
        for station in stations_list:
            # 1. Get horizontal channels:
            tr_N = self.st.select(station=station, channel="??N")[0]
            tr_E = self.st.select(station=station, channel="??E")[0]
            
            # 2. Get window indices:
            self.fs = tr_N.stats.sampling_rate
            win_start_idxs, win_end_idxs = self._select_windows()

            # 3. Calculate splitting angle and delay time 
            #    (via eigenvalue method):
            self._calc_splitting_eig_val_method(tr_N.data, tr_E.data, win_start_idxs, win_end_idxs)


        
    
    
