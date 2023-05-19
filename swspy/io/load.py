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
import subprocess
import gc

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
        obspy readable file (e.g. mseed). Default is archive.

    starttime : obspy UTCDateTime object
        The starttime to cut the data by. Any filters are applied before 
        cutting. If not supplied then will load all data from the supplied 
        file path (archive_vs_file must = file for this to be valid).
    
    endtime : obspy UTCDateTime object
        The endtime to cut the data by. Any filters are applied before 
        cutting. If not supplied then will load all data from the supplied 
        file path (archive_vs_file must = file for this to be valid).

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

    downsample_factor : int (default = 1)
        Factor by which to downsample the data, to speed up processing.
        If <downsample_factor> = 1, obviously doens't apply downsampling.

    upsample_factor : int (default = 1)
        Factor by which to upsample the data, to smooth waveforms for enhanced 
        timeshift processing. Currently uses weighted average slopes 
        interpolation method.
        If <upsample_factor> = 1, doens't apply upsampling.


    Methods
    -------
    read_waveform_data(stations=[], channels="*")
        Read in the waveform data for the specified time.

    """

    def __init__(self, path, starttime=None, endtime=None, archive_vs_file="archive", downsample_factor=1, upsample_factor=1):
        "Initiate load_waveforms object."
        # Specified directly by user:
        self.path = path
        self.starttime = starttime
        self.endtime = endtime
        self.archive_vs_file = archive_vs_file
        # Optional attributes:
        self.filter = False
        self.filter_freq_min_max = [1.0, 50.0]
        self.zero_phase = True
        self.remove_response = False
        self.response_file_path = None
        self.downsample_factor = downsample_factor
        self.upsample_factor = upsample_factor
        # Do some initial checks:
        if not starttime:
            if archive_vs_file != "file":
                print("Error: starttime and endtime must be specified if archive_vs_file is archive not file")
                raise
        if not endtime:
            if archive_vs_file != "file":
                print("Error: starttime and endtime must be specified if archive_vs_file is archive not file")
                raise


    def _force_stream_sample_alignment(self, st):
        """Function to force alignment if samples are out by less than a sample."""
        for i in range(1, len(st)):
            if np.abs(st[i].stats.starttime - st[0].stats.starttime) <= 1./st[i].stats.sampling_rate:
                st[i].stats.starttime = st[0].stats.starttime # Shift the start time so that all traces are alligned.
                # if st[i].stats.sampling_rate == st[0].stats.sampling_rate:
                #     st[i].stats.sampling_rate = st[0].stats.sampling_rate 
        return st

    
    def _force_stream_length_consistency(self, st):
        """Function to force traces in the streams to be the same length. Note: will force stream to take length 
        of shortest trace."""
        # Get minimum trace length:
        tr_min_len = len(st[0].data)
        for i in range(len(st)):
            if tr_min_len > len(st[i].data):
                tr_min_len = len(st[i].data)
        # And force all traces to be same length as shortest trace:
        for i in range(len(st)):
            if len(st[i].data) > tr_min_len:
                st[i].data = st[i].data[0:tr_min_len]
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
        self.event_uid = "*"

        # Load data:
        if self.archive_vs_file == "archive":
            year = self.starttime.year
            julday = self.starttime.julday
            datadir = os.path.join(self.path, str(year), str(julday).zfill(3))
        elif self.archive_vs_file == "file":
            datadir = os.path.dirname(self.path)
            self.event_uid = self.path.split(os.path.sep)[-1]
        else:
            print("Error: archive_vs_file = "+self.archive_vs_file+" is not recognised. Exiting.")
            raise 
        st = obspy.Stream()
        # Loop over stations (if specified):
        if self.stations:
            for station in self.stations:
                try:
                    st_tmp = obspy.read(os.path.join(datadir, ''.join(("*", self.event_uid, station, "*", 
                                    self.channels, "*")))).detrend("demean")
                    for tr_tmp in st_tmp:
                        st.append(tr_tmp)
                    del st_tmp
                    gc.collect()
                except TypeError:
                    continue
        else:
            try:
                st = obspy.read(os.path.join(datadir, ''.join(("*", self.event_uid, "*", self.channels, 
                                "*")))).detrend("demean")
            # Deal with incorrectly formatted individual files:
            except TypeError:
                st = obspy.Stream()
                for fname_tmp in glob.glob(os.path.join(datadir, ''.join(("*", self.event_uid, "*", 
                                            self.channels, "*")))):
                    try:
                        st_tmp = obspy.read(fname_tmp)
                        for tr in st_tmp:
                            st.append(tr)
                    except TypeError:
                        continue




        # Apply any filtering, if specified:
        if self.filter:
            st.filter('bandpass', freqmin=self.filter_freq_min_max[0], freqmax=self.filter_freq_min_max[1], 
                        corners=4, zerophase=self.zero_phase)

        # Trim data:
        if self.starttime:
            if self.endtime:
                st.trim(starttime=self.starttime, endtime=self.endtime)

        # Force allignment:
        st = self._force_stream_sample_alignment(st)

        # And force all traces in stream to be same length:
        st = self._force_stream_length_consistency(st)

        # And upsample data, if specified:
        if self.upsample_factor > 1:
            st.interpolate(sampling_rate=self.upsample_factor*st[0].stats.sampling_rate, 
                            method="weighted_average_slopes") # ( or method="lanczos")

        # And downsample data, if specified:
        if self.downsample_factor > 1:
            st.decimate(self.downsample_factor, no_filter=True)

        # And return stream:
        return st 










