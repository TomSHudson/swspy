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
        self.nonlinloc_hyp_data = read_nonlinloc(nonlinloc_event_path)
        # Define attributes:
        self.overall_win_start_pre_fast_S_pick = 0.1
        self.overall_win_start_post_fast_S_pick = 0.5
        self.win_S_pick_tolerance = 0.01
        self.rotate_step_deg = 2.0
        self.n_win = 10

    def _select_windows(self, fs):
        """
        Function to specify window start and end time indices.
        """
        start_idxs_range = int((self.overall_win_start_pre_fast_S_pick - self.win_S_pick_tolerance) * fs) 
        win_start_idxs = np.random.randint(start_idxs_range)
        end_idxs_start = int((self.overall_win_start_pre_fast_S_pick + self.overall_win_start_post_fast_S_pick) * fs) 
        win_end_idxs = np.random.randint(end_idxs_start, end_idxs_start + start_idxs_range)
        return win_start_idxs, win_end_idxs

    def perform_sws_analysis():
        """Function to perform splitting analysis."""
        # Loop over stations in stream:
        stations_list = []
        for tr in self.st: 
            if tr.stats.station not in stations_list:
                append(tr.stats.station)
        for station in stations_list:
            # 1. Get horizontal channels:
            tr_N = self.st.select(station=station, channel="??N")[0]
            tr_E = self.st.select(station=station, channel="??E")[0]
            
            # 2. Get window indices:
            HERE!!! (Using self._select_windows())


        
    
    
