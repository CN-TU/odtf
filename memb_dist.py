#!/usr/bin/env python
#
# FIV, Feb 2018

import numpy as np
import csv
import fileinput
import sys

config_files={'labels':"test_labels.csv", 'membership':"membership.csv", 'predictions':"predictions.csv"}

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
reader = csv.reader(open(config_files['labels'], "rb"), delimiter=",")
aux = list(reader)
lab = np.array(aux).astype("int")

# Loading the labels of the training dataset 
reader = csv.reader(open(config_files['predictions'], "rb"), delimiter=",")
aux = list(reader)
pred = np.array(aux).astype("int")

# Loading the labels of the training dataset 
reader = csv.reader(open(config_files['membership'], "rb"), delimiter=",")
aux = list(reader)
memb = np.array(aux).astype("float")

misclass = np.bitwise_xor(lab, pred)
goodclass = np.logical_not(misclass)
tp = np.bitwise_and(lab, pred) 
fp = np.bitwise_and(misclass, pred)
tn = np.bitwise_and(goodclass, np.logical_not(pred))
fn = np.bitwise_and(misclass, np.logical_not(pred))
misclass = misclass.astype(float)

ind = np.where(tp==1)
m_tp = np.take(memb, ind)
ind = np.where(fp==1)
m_fp = np.take(memb, ind)
ind = np.where(tn==1)
m_tn = np.take(memb, ind)
ind = np.where(fn==1)
m_fn = np.take(memb, ind)

q10 = np.percentile(m_tp, 25, axis=1)
q50 = np.percentile(m_tp, 50, axis=1)
q90 = np.percentile(m_tp, 75, axis=1)
print ("TP -- q25: %s, q50: %s, q75: %s" % (q10[0], q50[0], q90[0]))

q10 = np.percentile(m_fp, 25, axis=1)
q50 = np.percentile(m_fp, 50, axis=1)
q90 = np.percentile(m_fp, 75, axis=1)
print ("FP -- q25: %s, q50: %s, q75: %s" % (q10[0], q50[0], q90[0]))

q10 = np.percentile(m_tn, 25, axis=1)
q50 = np.percentile(m_tn, 50, axis=1)
q90 = np.percentile(m_tn, 75, axis=1)
print ("TN -- q25: %s, q50: %s, q75: %s" % (q10[0], q50[0], q90[0]))

q10 = np.percentile(m_fn, 25, axis=1)
q50 = np.percentile(m_fn, 50, axis=1)
q90 = np.percentile(m_fn, 75, axis=1)
print ("FN -- q25: %s, q50: %s, q75: %s" % (q10[0], q50[0], q90[0]))
