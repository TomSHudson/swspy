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

class proc_many_events:
    """
    Class to process many events to calculate shear-wave splitting.

    Parameters
    ----------
    None.

    Attributes
    ----------
    filter : bool (default = False)
        If True, then filters the data by <filter_freq_min_max>.
    
    filter_freq_min_max : list of two floats (default = [1.0, 100.0])
        Filter parameters for filtering the data.

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

    downsample_factor : int (default = 1)
        Factor by which to downsample the data, to speed up processing.
        If <downsample_factor> = 1, doens't apply downsampling.

    upsample_factor : int (default = 1)
        Factor by which to upsample the data, to smooth waveforms for enhanced 
        timeshift processing. Currently uses weighted average slopes 
        interpolation method.
        If <upsample_factor> = 1, doens't apply upsampling.

    coord_system : str
        Coordinate system to perform analysis in. Options are: LQT, ZNE. Will convert 
        splitting angles back into coordinates relative to ZNE whatever system it 
        performs the splitting within. Default = ZNE. 

    sws_method : str
        Method with which to calculate sws parameters. Options are: EV, EV_and_XC.
        EV - Eigenvalue method (as in Silver and Chan (1991), Teanby (2004), Walsh et 
        al. (2013)). EV_and_XC - Same as EV, except also performs cross-correlation 
        for automation approach, as in Wustefeld et al. (2010). Default is EV_and_XC.

    output_wfs : bool
        If True, will save uncorrected and corrected waveforms to:
        <outdir>/<data>/<event_uid>_wfs_uncorr.png
        and
        <outdir>/<data>/<event_uid>_wfs_corr.png
        Default is True.

    output_plots : bool
        If True, will save output plots to:
        <outdir>/<plots>/<event_uid>_<station>.png
        Default is False.

    suppress_direct_plotting : bool
        If True, suppresses direct plotting so plots are only saved to file (and will
        only save plots to file if <output_plots=True>). Default = False

    Methods
    -------
    run_events_from_nlloc
    run_events_sws_fmt

    """

    def __init__(self):
        """Initiate the class object.

        Parameters
        ----------
        None.

        """
        # Define attributes:
        # Define general signal processing attributes:
        self.filter = False
        self.filter_freq_min_max = [1.0, 100.0]
        # Define splitting attributes:
        self.overall_win_start_pre_fast_S_pick = 0.1
        self.overall_win_start_post_fast_S_pick = 0.2
        self.win_S_pick_tolerance = 0.1
        self.rotate_step_deg = 2.0
        self.max_t_shift_s = 0.1
        self.n_win = 10
        self.downsample_factor = 1
        self.upsample_factor = 1
        self.coord_system = "ZNE"
        self.sws_method = "EV_and_XC"
        # Define other processing auxilary params:
        self.output_wfs = True
        self.output_plots = False
        self.suppress_direct_plotting = True


    def run_events_from_nlloc(self, mseed_archive_dir, nlloc_dir, outdir, event_prepad=1.0, event_postpad=30.0, nproc=1):
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

        event_prepad : float

        event_postpad : float

        nproc : int

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
        data_outdir = os.path.join(outdir, "data")
        Path(data_outdir).mkdir(parents=True, exist_ok=True)
        if self.output_plots:
            plot_outdir = os.path.join(outdir, "plots")
            Path(plot_outdir).mkdir(parents=True, exist_ok=True)

        # Loop over events, processing:
        count = 0
        for nlloc_fname in nlloc_fnames:
            print(''.join(("Processing for event: ", str(count), "/", str(len(nlloc_fnames)))))
            count+=1
            # 1. Get waveform and other event data:
            try:
                nlloc_hyp_data = swspy.io.read_nonlinloc.read_hyp_file(nlloc_fname)
                starttime = nlloc_hyp_data.origin_time - event_prepad
                endtime = nlloc_hyp_data.origin_time + event_postpad
                load_wfs_obj = swspy.io.load_waveforms(mseed_archive_dir, starttime=starttime, endtime=endtime, downsample_factor=self.downsample_factor, upsample_factor=self.upsample_factor)
                load_wfs_obj.filter = self.filter
                load_wfs_obj.filter_freq_min_max = self.filter_freq_min_max
                st = load_wfs_obj.read_waveform_data()
            except Exception as e:
                print("Warning:", e)
                print("Skipping this event.")
                continue

            # 2. Calculate splitting for event:
            # 2.i. Setup splitting event object:
            splitting_event = swspy.splitting.create_splitting_object(st, nonlinloc_event_path=nlloc_fname)
            splitting_event.overall_win_start_pre_fast_S_pick = self.overall_win_start_pre_fast_S_pick
            splitting_event.overall_win_start_post_fast_S_pick = self.overall_win_start_post_fast_S_pick
            splitting_event.win_S_pick_tolerance = self.win_S_pick_tolerance
            splitting_event.rotate_step_deg = self.rotate_step_deg
            splitting_event.max_t_shift_s = self.max_t_shift_s
            splitting_event.n_win = self.n_win
            # 2.ii. Perform splitting analysis:
            splitting_event.perform_sws_analysis(coord_system=self.coord_system, sws_method=self.sws_method) #(coord_system="LQT") #(coord_system="ZNE")

            # 3. Save splitting for event:
            splitting_event.save_result(outdir=data_outdir)
            if self.output_wfs:
                splitting_event.save_wfs(outdir=data_outdir)
            if self.output_plots:
                splitting_event.plot(outdir=plot_outdir, suppress_direct_plotting=self.suppress_direct_plotting)

            # Tidy:
            del splitting_event, st, load_wfs_obj, nlloc_hyp_data
            gc.collect()
        
        print("Finished processing shear-wave splitting for data in:", nlloc_dir)
        print("Data saved to:", outdir)

    
    def run_events_sws_fmt(self, datadir, outdir, S_pick_time_after_start_s=10.0):
        """
        Function to run many events through shear-wave splitting analysis using 
        sws format and sac data.
        
        --------------------------- sws format notes: ---------------------------
       
        Data directories containing sac data must be 
        formatted as follows:
        <datadir>/event_uid/*.?H*

        Sac data for each event must be trimmed with <S_pick_time_after_start_s> 
        seconds padding before the S pick and sufficient padding after. Additionally, 
        sac data must have back azimuth information (and inclination info. if not 
        using ZNE coordinate system).
        -------------------------------------------------------------------------

        Parameters
        ----------

        datadir : str
            The overall directory path containing all events with event unique IDs 
            as each directory, with each directory containing SAC files for each 
            component of each station. I.e. data follows format:
            <datadir>/event_uid/*.?H*

        outdir : str
            Path to output directory to save data to. Saves results to:
            csv event summary file: <outdir>/<data>/event_uid.csv
            And if <output_plots> is specified, then will output plots to:
            png event station file: <outdir>/<data>/<event_uid>_<station>.png

        S_pick_time_after_start_s : float
            Time, in seconds, of S pick after start of SAC trace. Default is 10.0 s.
        
        """
        # Get event uids to loop over:
        event_uids_tmp = glob.glob(os.path.join(datadir, "*"))
        event_uids_tmp.sort()
        event_uids = []
        for i in range(len(event_uids_tmp)):
            if os.path.isdir(event_uids_tmp[i]):
                event_uids.append(os.path.basename(event_uids_tmp[i]))
        del event_uids_tmp

        # Create output directories if not already created:
        data_outdir = os.path.join(outdir, "data")
        Path(data_outdir).mkdir(parents=True, exist_ok=True)
        if self.output_plots:
            plot_outdir = os.path.join(outdir, "plots")
            Path(plot_outdir).mkdir(parents=True, exist_ok=True)

        # Loop over events:
        count = 1
        for event_uid in event_uids:
            print(''.join(("Processing for event UID: ", event_uid, " (", str(count), "/", str(len(event_uids)), ")")))
            count+=1
            # 1. Get waveform data:
            event_sac_fnames_path = os.path.join(datadir, event_uid, "*.?H*")
            load_wfs_obj = swspy.io.load_waveforms(event_sac_fnames_path, archive_vs_file="file", downsample_factor=self.downsample_factor, upsample_factor=self.upsample_factor)
            load_wfs_obj.filter = self.filter
            load_wfs_obj.filter_freq_min_max = self.filter_freq_min_max
            try:
                st = load_wfs_obj.read_waveform_data()
            except:
                print("Error: Cannot read mseed files. Check that files mist be of the format:", os.path.join(datadir, event_uid, "*.?H*"))
                print("Exiting.")
                sys.exit()

            # 2. Calculate splitting for event:
            # 2.i. Setup splitting event object:
            # Get station info:
            stations_in = []
            back_azis_all_stations = []
            receiver_inc_angles_all_stations = []
            S_phase_arrival_times = []
            for tr in st:
                if tr.stats.station not in stations_in:
                    stations_in.append(tr.stats.station)
                    back_azis_all_stations.append(tr.stats.sac['baz'])
                    if self.coord_system == "ZNE":
                        receiver_inc_angles_all_stations.append(0.)
                    else:
                        print("Error: Non-ZNE coordinate system currently not supported for run_events_sws_fmt() automation method. Exiting.")
                        sys.exit()
                    S_phase_arrival_times.append(tr.stats.starttime + S_pick_time_after_start_s)
            # Create the splitting event object:
            splitting_event = swspy.splitting.create_splitting_object(st, stations_in=stations_in, S_phase_arrival_times=S_phase_arrival_times, 
                                                                        back_azis_all_stations=back_azis_all_stations, 
                                                                        receiver_inc_angles_all_stations=receiver_inc_angles_all_stations)
            # Assign splitting parameters:
            splitting_event.overall_win_start_pre_fast_S_pick = self.overall_win_start_pre_fast_S_pick
            splitting_event.overall_win_start_post_fast_S_pick = self.overall_win_start_post_fast_S_pick
            splitting_event.win_S_pick_tolerance = self.win_S_pick_tolerance
            splitting_event.rotate_step_deg = self.rotate_step_deg
            splitting_event.max_t_shift_s = self.max_t_shift_s
            splitting_event.n_win = self.n_win
            # 2.ii. Perform splitting analysis:
            splitting_event.perform_sws_analysis(coord_system=self.coord_system, sws_method=self.sws_method) #(coord_system="LQT") #(coord_system="ZNE")

            # 3. Save splitting for event:
            splitting_event.save_result(outdir=data_outdir)
            if self.output_wfs:
                splitting_event.save_wfs(outdir=data_outdir)
            if self.output_plots:
                splitting_event.plot(outdir=plot_outdir, suppress_direct_plotting=self.suppress_direct_plotting)

            # Tidy:
            del splitting_event, st, load_wfs_obj
            gc.collect()

        print("Finished processing shear-wave splitting for data in:", datadir)
        print("Data saved to:", outdir)





