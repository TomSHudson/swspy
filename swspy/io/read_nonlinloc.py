#!/usr/bin/python

#-----------------------------------------------------------------------------------------------------------------------------------------

# Script Description:
# Script to convert nonlinloc input and output files to python friendly format.

# Input variables:

# Output variables:

# Usage:
# from NonLinLocPy import read_nonlinloc 
# hyp_file_data = read_nonlinloc.read_hyp_file("loc.Tom_RunNLLoc000.20110603.142037.grid0.loc.hyp")

# Created by Tom Hudson, 24th Febraury 2020

#-----------------------------------------------------------------------------------------------------------------------------------------

# ------------------- Import neccessary modules ------------------
import obspy
import sys, os
import re
# ------------------- End: Import neccessary modules ------------------

# ------------------- Define generally useful functions/classes -------------------

class read_hyp_file:
    """Class to import all useful information from a NonLinLoc hyp file."""
    def __init__(self, hyp_fname):
        self.hyp_fname = hyp_fname
        # And read in data:
        self.read()

    def read(self):
        """Function to read the lines of the file and append to the class structure."""
        hyp_file = open(self.hyp_fname, 'r')
        hyp_file_lines = hyp_file.readlines()
        for i in range(len(hyp_file_lines)):
            line_split = re.split(' +', hyp_file_lines[i])
            if line_split[0] == 'HYPOCENTER':
                self.max_prob_hypocenter = {}
                self.max_prob_hypocenter['x'] = float(line_split[2])
                self.max_prob_hypocenter['y'] = float(line_split[4])
                self.max_prob_hypocenter['z'] = float(line_split[6])
            if line_split[0] == 'GEOGRAPHIC':
                self.origin_time = obspy.UTCDateTime(''.join(line_split[2:8]))
                self.max_prob_hypocenter['lat'] = float(line_split[9])
                self.max_prob_hypocenter['lon'] = float(line_split[11])
                self.max_prob_hypocenter['depth'] = float(line_split[13])
            if line_split[0] == 'QUALITY':
                self.t_rms = float(line_split[8])
                self.n_phase = float(line_split[10])
                self.azi_gap = float(line_split[12])
            if line_split[0] == 'STATISTICS':
                self.std_x_km = float(line_split[8])
                self.std_y_km = float(line_split[14])
                self.std_z_km = float(line_split[18])
            if line_split[0] == 'STAT_GEOG':
                self.expect_hypocenter = {}
                self.expect_hypocenter['lat'] = float(line_split[2])
                self.expect_hypocenter['lon'] = float(line_split[4])
                self.expect_hypocenter['depth'] = float(line_split[6])
            if line_split[0] == 'PHASE':
                self.phase_data = {}
                new_line_count = i+1
                for j in range(new_line_count, len(hyp_file_lines)-3):
                    line_split = re.split(' +', hyp_file_lines[j])
                    station = line_split[0]
                    if station not in list(self.phase_data.keys()):
                        self.phase_data[station] = {}
                    phase_id = line_split[4]
                    self.phase_data[station][phase_id] = {}
                    try:
                        self.phase_data[station][phase_id]['arrival_time'] = obspy.UTCDateTime(''.join((line_split[6], line_split[7], line_split[8].split('.')[0].zfill(2), '.', line_split[8].split('.')[1])))
                    except IndexError:
                        self.phase_data[station][phase_id]['arrival_time'] = obspy.UTCDateTime(''.join((line_split[6], line_split[7], line_split[8].zfill(2), '.000'))) # If no subsecond information
                    self.phase_data[station][phase_id]['arrival_time_err'] = float(line_split[10])
                    self.phase_data[station][phase_id]['TTpred'] = float(line_split[15])
                    self.phase_data[station][phase_id]['Res'] = float(line_split[16])
                    self.phase_data[station][phase_id]['Weight'] = float(line_split[17])
                    self.phase_data[station][phase_id]['SDist'] = float(line_split[21])
                    self.phase_data[station][phase_id]['SAzim'] = float(line_split[22])
                    self.phase_data[station][phase_id]['RAzim'] = float(line_split[23])
                    self.phase_data[station][phase_id]['RDip'] = float(line_split[24])
                    self.phase_data[station][phase_id]['StaLoc'] = {}
                    self.phase_data[station][phase_id]['StaLoc']['x'] = float(line_split[18])
                    self.phase_data[station][phase_id]['StaLoc']['y'] = float(line_split[19])
                    self.phase_data[station][phase_id]['StaLoc']['z'] = float(line_split[20])
                    self.phase_data[station][phase_id]['polarity'] = line_split[5]
                # And break the loop as already found all the information:
                break 
                

# ------------------- End: Define generally useful functions/classes -------------------
