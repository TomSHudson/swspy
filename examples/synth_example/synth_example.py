#!/usr/bin/env python
# coding: utf-8

# # Simple synthetic example

# In[1]:


#get_ipython().run_line_magic('load_ext', 'autoreload')
#get_ipython().run_line_magic('autoreload', '2')


# In[2]:


import swspy
import os, sys
import numpy as np
import matplotlib 
import matplotlib.pyplot as plt
from scipy import signal
#get_ipython().run_line_magic('matplotlib', 'notebook')


# ### 1. Create source-time function:

# In[3]:


# Create source-time function:
ZNE_st = swspy.splitting.forward_model.create_src_time_func(10., 1000, src_pol_from_N=0, src_pol_from_up=0)
ZNE_st.plot()


# ### 2. Apply a layer of splitting:

# In[4]:


# Specify layer anisotropy parameters:
phi_from_N = 60.
dt = 0.5 #0.05
back_azi = 0
event_inclin_angle_at_station = 0

# Apply splitting:
ZNE_st_layer1 = swspy.splitting.forward_model.add_splitting(ZNE_st, phi_from_N, dt, back_azi, event_inclin_angle_at_station)
ZNE_st_layer1.plot()

plt.figure(figsize=(4,4))
plt.plot(ZNE_st_layer1.select(channel="??E")[0].data, ZNE_st_layer1.select(channel="??N")[0].data)
abs_max_tmp = np.max(np.array(np.max(np.abs(ZNE_st_layer1.select(channel="??N")[0].data)), np.max(np.abs(ZNE_st_layer1.select(channel="??E")[0].data))))
plt.xlim([-abs_max_tmp, abs_max_tmp])
plt.ylim([-abs_max_tmp, abs_max_tmp])
plt.title("Particle motion")
plt.xlabel("E ($m$ $s^{-1}$)")
plt.xlabel("N ($m$ $s^{-1}$)")
plt.show()


# ### 3. Measure splitting on single layer:

# In[ ]:


# Calculate splitting:
event_uid = "single-layer"
S_phase_arrival_times = [ZNE_st_layer1[0].stats.starttime+1.0]
back_azis_all_stations = [back_azi]
receiver_inc_angles_all_stations = [event_inclin_angle_at_station]
splitting_event = swspy.splitting.create_splitting_object(ZNE_st_layer1, event_uid=event_uid, stations_in=["synth"], S_phase_arrival_times=S_phase_arrival_times, back_azis_all_stations=back_azis_all_stations, receiver_inc_angles_all_stations=receiver_inc_angles_all_stations) 
splitting_event.overall_win_start_pre_fast_S_pick = 0.5
splitting_event.win_S_pick_tolerance = 0.2
splitting_event.overall_win_start_post_fast_S_pick = 1.0
splitting_event.rotate_step_deg = 1.0 
splitting_event.max_t_shift_s = 1.0
splitting_event.n_win = 10
splitting_event.perform_sws_analysis(coord_system="ZNE", sws_method="EV")

# And plot splitting result:
splitting_event.plot(outdir=os.path.join("outputs", "plots"))

# And save result to file:
splitting_event.save_result(outdir=os.path.join("outputs", "data"))


# In[ ]:





# In[ ]:


# phi_from_N = 60.
# dt = 0.05 #0.5
# back_azi = 0
# event_inclin_angle_at_station = 0
# ZNE_st_with_sws = swspy.splitting.forward_model.add_splitting(ZNE_st_layer1, phi_from_N, dt, back_azi, event_inclin_angle_at_station)
# ZNE_st_with_sws.plot()

# plt.figure(figsize=(4,4))
# plt.plot(ZNE_st_with_sws.select(channel="??E")[0].data, ZNE_st_with_sws.select(channel="??N")[0].data)
# plt.show()


# In[ ]:




