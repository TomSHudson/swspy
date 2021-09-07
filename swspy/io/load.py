#-----------------------------------------------------------------------------------------------------------------------------------------

# Script Description:
# Submodule to load waveform data from a file or archive.

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

class load_waveforms:
    """
    A class to load waveforms from file or an archive.

    Notes:
    - Will currently only load archived data from the format year/jul_day/*station*
    - Does not currently remove instrument response

    Parameters
    ----------
    path : str
        The path to the overall data archive or file to load waveforms from.

    archive_vs_file: str (default = "archive")
        Describes what the parameter <path> is associated with. If archive, 
        then <path> is to an archive. If file, then <path> is the path to a 
        obspy readable file (e.g. mseed).

    starttime : obspy UTCDateTime object
        The starttime to cut the data by. Any filters are applied before 
        cutting.
    
    endtime : obspy UTCDateTime object
        The endtime to cut the data by. Any filters are applied before 
        cutting.

    Attributes
    ----------
    filter : bool (default = False)
        If True, then filters the data by <filter_freq_min_max>.
    
    filter_freq_min_max : list of two floats (default = [2.0, 20.0])
        Filter parameters for filtering the data.
    
    zero_phase : bool (default = True)
        If True, applies zero phase filter (if <filter> = True).

    remove_response : bool (default = False)
        If True, removes instrument response using the response file 
        specified by <response_file>.
        (Note: Not currently implemented!)
    
    response_file_path : str (default = None)
        Path to response file used to remove instrument response.

    Methods
    -------
    read_waveform_data(stations=[], channels="*")
        Read in the waveform data for the specified time.

    """

    def __init__(self, path, starttime, endtime, archive_vs_file="archive"):
        "Initiate load_waveforms object."
        # Specified directly by user:
        self.path = path
        self.starttime = starttime
        self.endtime = endtime
        # Optional attributes:
        self.filter = False
        self.filter_freq_min_max = [1.0, 50.0]
        self.zero_phase = True
        self.remove_response = False
        self.response_file_path = None

    def _force_stream_sample_alignment(self, st):
        """Function to force alignment if samples are out by less than a sample."""
        for i in range(1, len(st)):
            if np.abs(st[i].stats.starttime - st[0].stats.starttime) <= 1./st[i].stats.sampling_rate:
                st[i].stats.starttime = st[0].stats.starttime # Shift the start time so that all traces are alligned.
                # if st[i].stats.sampling_rate == st[0].stats.sampling_rate:
                #     st[i].stats.sampling_rate = st[0].stats.sampling_rate 
        return st

    def read_waveform_data(self, stations=None, channels="*"):
        """Function to read waveform data. Filters if specified.

        Parameters
        ----------
        stations : list of strs (default = None)
            List of stations to import. If None then imports all 
            available stations (default).
        
        channels: str (default = "*")
            List of channels to import. Default is to import all channels.

        Returns:
        st : obspy stream
            Obspy stream object containing all the data requested.

        """
        # initiate any further variables specified by user:
        self.stations = stations
        self.channels = channels

        # Load data:
        year = self.starttime.year
        julday = self.starttime.julday
        datadir = os.path.join(self.path, str(year), str(julday).zfill(3))
        st = obspy.Stream()
        # Loop over stations (if specified):
        if self.stations:
            for station in self.stations:
                st_tmp = obspy.read(os.path.join(datadir, ''.join(("*", station, "*", 
                                    self.channels, "*")))).detrend("demean")
                for tr_tmp in st_tmp:
                    st.append(tr_tmp)
                del st_tmp
                gc.collect()
        else:
            st = obspy.read(os.path.join(datadir, ''.join(("*", self.channels, 
                                "*")))).detrend("demean")

        # Apply any filtering, if specified:
        if self.filter:
            st.filter('bandpass', freqmin=self.filter_freq_min_max[0], freqmax=self.filter_freq_min_max[1], 
                        corners=4, zerophase=self.zero_phase)

        # Trim data:
        st.trim(starttime=self.starttime, endtime=self.endtime)

        # Force allignment:
        st = self._force_stream_sample_alignment(st)

        # And return stream:
        return st 









