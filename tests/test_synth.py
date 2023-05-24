# Test cases for syntetic splitting
# 
# These tests create synthetic data containing a known amount of SWS, 
# add noise  and run the measurment over this data. Tests pass if the
# measured splitting is within error of the input parameters.

import pytest
import numpy
import swspy

@pytest.mark.parametrize("dt, phi_N, src_pol_N, src_pol_U, snr",
        [(0.5, 60.0, 0.0, 0.0, 100)])
def test_single_synthetic(dt, phi_N, src_pol_N, src_pol_U, snr):
    """
    Test with synthetic for  given splitting and basic source parameters

    We create a split Ricker wavlet for some 'sensible' frequency etc.
    and apply the splitting parameters and noise. Then check the splitting
    gives results within error of the input values.
    
    dt: amount of splitting (s)
    phi_N: fast direction from north (degrees)
    src_pol_N: source polarization direction from north (degrees)
    src_pol_U: source polariztion direction from up (degrees)
    """
    dur = 10.0 # Duration of stream in s
    fs = 1000.0 # Frequency of stream in Hz
    src_freq = 10.0 # Dominant frequency of source in Hz
    wavelet = 'ricker'
    t_src = 1.0 # Peak in initial source time function from start in s
    back_azi = 0.0 # Back azimuth in degrees
    event_inclin_angle_at_station = 0.0
    
    # Setup trace and apply splitting
    ZNE_st = swspy.splitting.forward_model.create_src_time_func(dur, fs, 
        src_pol_from_N=src_pol_N, src_pol_from_up=src_pol_U, t_src=t_src,
        wavelet=wavelet, src_freq=src_freq)

    ZNE_split = swspy.splitting.forward_model.add_splitting(ZNE_st, phi_N, 
        dt, back_azi, event_inclin_angle_at_station, snr=snr)

    # Setup and apply splitting measurment
    S_phase_arrival_times = [ZNE_split[0].stats.starttime+t_src]
    back_azis_all_stations = [back_azi]
    receiver_inc_angles_all_stations = [event_inclin_angle_at_station]

    splitting_event = swspy.splitting.create_splitting_object(ZNE_split,
        event_uid='test', stations_in=["synth"], 
        S_phase_arrival_times=S_phase_arrival_times,
        back_azis_all_stations=back_azis_all_stations,
        receiver_inc_angles_all_stations=receiver_inc_angles_all_stations) 

    splitting_event.overall_win_start_pre_fast_S_pick = 0.5
    splitting_event.win_S_pick_tolerance = 0.2
    splitting_event.overall_win_start_post_fast_S_pick = 1.0
    splitting_event.rotate_step_deg = 2
    splitting_event.max_t_shift_s = 0.75
    splitting_event.n_win = 10
    splitting_event.perform_sws_analysis(coord_system="ZNE", sws_method="EV")

    # Extract results - these are in a dataframe and we want the (single) value
    phi_N_res = splitting_event.sws_result_df['phi_from_N'].values[0]
    phi_err_res = splitting_event.sws_result_df['phi_err'].values[0]
    dt_res = splitting_event.sws_result_df['dt'].values[0]
    dt_err_res = splitting_event.sws_result_df['dt_err'].values[0]
    lam2_lam1 = splitting_event.sws_result_df['lambda2/lambda1 ratio'].values[0]

    assert(lam2_lam1 >= 0.0)
    swspy.testing.assert_angle_allclose(phi_N, phi_N_res, rtol=0.0, atol=phi_err_res)
    numpy.testing.assert_allclose(dt, dt_res, rtol=0.0, atol=dt_err_res)
    
    
    
