#-----------------------------------------------------------------------------------------------------------------------------------------

# Script Description:
# Script to automate splitting analysis for many events.

# Input variables:

# Output variables:

# Created by Tom Hudson, 13th September 2021

#-----------------------------------------------------------------------------------------------------------------------------------------

# Import neccessary modules:
import swspy
import obspy
from obspy import UTCDateTime
import numpy as np
import matplotlib.pyplot as plt
import glob 
import os, sys 
from pathlib import Path
import pandas as pd
import gc 
from NonLinLocPy import read_nonlinloc # For reading NonLinLoc data (can install via pip)


def run_events_from_nlloc(mseed_archive_dir, nlloc_dir, outdir, output_plots=False, event_prepad=1.0, event_postpad=30.0, freqmin=1.0, freqmax=100.0):
    """
    Function to run many events through shear-wave splitting analysis using 
    nonlinloc and mseed data (in archive format: <mseed_archive_dir>/<year>/
    <julday>/...).

    Parameters
    ----------
    mseed_archive_dir : str
        Path to mseed archive overall directory. Subdirectory paths should be 
        in format: <mseed_archive_dir>/<year>/<julday>/yearjulday_station_channel.m.

    nlloc_dir : str
        Path to nlloc .grid0.loc.hyp output files corresponding to events that want 
        to process.

    outdir : str
        Path to output directory to save data to. Saves results to:
        csv event summary file: <outdir>/<data>/event_uid.csv
        And if <output_plots> is specified, then will output plots to:
        png event station file: <outdir>/<data>/<event_uid>_<station>.png

    output_plots : bool
        If True, will save output plots to:
        <outdir>/<data>/<event_uid>_<station>.png

    Returns
    -------
    Data output to files in outdir, as specified above.
    """
    # Get event list:
    nlloc_fnames = glob.glob(os.path.join(nlloc_dir, 'loc.*.*.*.grid0.loc.hyp'))
    if len(nlloc_fnames) == 0:
        print("Error: No events found in", nlloc_fnames, ". Are files of format loc.*.*.*.grid0.loc.hyp?", "Exiting.")
        sys.exit()

    # Create output directories if not already created:
    Path(os.path.join(outdir, "data")).mkdir(parents=True, exist_ok=True)
    if output_plots:
        Path(os.path.join(outdir, "plots")).mkdir(parents=True, exist_ok=True)

    # Loop over events, processing:
    count = 0
    for nlloc_fname in nlloc_fnames:
        print(''.join(("Processing for event: ", count, "/", len(nlloc_fnames))))
        # Get waveform data:
        nlloc_hyp_data = read_nonlinloc.read_hyp_file(nlloc_fname)
        starttime = nlloc_hyp_data.origin_time - event_prepad
        endtime = nlloc_hyp_data.origin_time + event_postpad
        load_wfs_obj = swspy.io.load_waveforms(mseed_archive_dir, starttime=starttime, endtime=endtime)
        load_wfs_obj.filter = True
        load_wfs_obj.filter_freq_min_max = [freqmin, freqmax]
        st = load_wfs_obj.read_waveform_data()

        # Calculate splitting for event:
        splitting_event = swspy.splitting.create_splitting_object(st, nonlinloc_event_path=nonlinloc_event_path)
        # HERE!!!


        # Tidy:
        del splitting_event, st, load_wfs_obj, nlloc_hyp_data
        gc.collect()



        




