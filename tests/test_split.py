#-----------------------------------------------------------------------------------------------------------------------------------------

# Script Description:
# To perform unit tests.

# Input variables:

# Output variables:

# Created by Tom Hudson, 6th July 2022

#-----------------------------------------------------------------------------------------------------------------------------------------

# Import neccessary modules:
import gc
import obspy
import swspy 



def test_perform_sws_analysis_icequake_example():
    """Function to test perform_sws_analysis() function for an icequake example."""
    # Load data:
    example_path = "../"
    archive_path = "data/mseed"
    archive_vs_file = "archive"
    nonlinloc_event_path = "data/loc.Tom__RunNLLoc000.20090121.042009.grid0.loc.hyp"

    starttime = obspy.UTCDateTime("20090121T042009.18523") - 0.5
    endtime = obspy.UTCDateTime("20090121T042009.18523") + 2.5
    load_wfs_obj = swspy.io.load_waveforms(archive_path, starttime=starttime, endtime=endtime)
    load_wfs_obj.filter = True
    load_wfs_obj.filter_freq_min_max = [1.0, 80.0]
    st_in = load_wfs_obj.read_waveform_data()
    st = obspy.Stream()
    st = st.append(st_in[0])
    del st_in
    gc.collect()
    
    # Calculate splitting:
    splitting_event = swspy.splitting.create_splitting_object(st, nonlinloc_event_path=nonlinloc_event_path) #(st, nonlinloc_event_path) #(st.select(station="ST01"), nonlinloc_event_path)
    splitting_event.overall_win_start_pre_fast_S_pick = 0.3 #0.1
    splitting_event.win_S_pick_tolerance = 0.1
    splitting_event.overall_win_start_post_fast_S_pick = 0.2 #0.2
    splitting_event.rotate_step_deg = 1.0 #2.5
    splitting_event.max_t_shift_s = 0.12
    splitting_event.n_win = 10 #5 #10
    # splitting_event.perform_sp
    splitting_event.perform_sws_analysis(coord_system="ZNE", sws_method="EV_and_XC") #(coord_system="LQT") #(coord_system="ZNE")

