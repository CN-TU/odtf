#!/usr/bin/env python


import numpy as np
import csv
import fileinput
import sys

config_files={'data(in)':"data_in.csv", 'membership_scores(in)':"membership.csv", 'data(out)':"data_out.csv"}

# Begin 
if len(sys.argv) == 1:
	print "ERROR: Configuration file is required!"
	quit()

# Read configuration file
for line in fileinput.input():
	name,val=line.split(":")
	if (name in config_files):
		config_files[name]=val.rstrip()

print "Loading data..."
# Loading the training dataset
reader = csv.reader(open(config_files['data(in)'], "rb"), delimiter=",")
data_in_ = list(reader)
data_in = np.array(data_in_).astype("float")

# Loading the labels of the training dataset 
reader = csv.reader(open(config_files['membership_scores(in)'], "rb"), delimiter=",")
membership_scores_ = list(reader)
membership_scores = np.array(membership_scores_).astype("float")

from scipy import stats
from numpy import inf
membership_scores[membership_scores == inf] = len(data_in[0])

import skfuzzy as fuzz

data_out = np.concatenate((data_in, membership_scores), axis=1)

print "Saving output data into a csv file..."
np.savetxt(config_files['data(out)'], data_out, delimiter=",")


