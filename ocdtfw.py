#!/usr/bin/env python
# ODTF (One-Class Decision Tree Fuzzyfication) operates on a decision tree model by ranking
# test-samples with a one-class membership score based on the distance to decision thresholds 
# (i.e., class boundaries)
#
# FIV, Feb 2018

from scipy.spatial import distance
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn import datasets
from sklearn import tree
from scipy import stats
import operator
import math as ma
import numpy as np
import csv
import fileinput
import sys

config_files={'training_data(in)':"training_data.csv", 'training_labels(in)':"training_labels.csv", 'testing_data(in)':"testing_data.csv", 'testing_labels(in)':"testing_labels.csv", 'membership(out)':"membership.csv", 'thresholds(out)':"thresholds.csv", 'features_linked_to_thresholds(out)':"features.csv",'predictions(out)':"predictions.csv",'feature_importance(out)':"importance.csv", 'feature_closest(out)':"featclosest.csv"}

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
reader = csv.reader(open(config_files['training_data(in)'], "rb"), delimiter=",")
trainX = list(reader)
tr_X = np.array(trainX).astype("float")

# Loading the labels of the training dataset 
reader = csv.reader(open(config_files['training_labels(in)'], "rb"), delimiter=",")
trainy = list(reader)
tr_y = np.array(trainy).astype("int")

# Loading the testing dataset
reader = csv.reader(open(config_files['testing_data(in)'], "rb"), delimiter=",")
testX = list(reader)
X = np.array(testX).astype("float")

# Loading the labels of the testing dataset 
reader = csv.reader(open(config_files['testing_labels(in)'], "rb"), delimiter=",")
testy = list(reader)
y = np.array(testy).astype("int")

# Build decision tree classifier
print "Building DT..."
dt = tree.DecisionTreeClassifier(criterion='entropy',max_depth=8)
dt.fit(tr_X, tr_y)
res = dt.predict(X)
res = res.astype(int)
res = np.reshape(res, len(res)) 
y = np.reshape(y, len(y)) 
misclass = np.bitwise_xor(res, y)
misclass = misclass.astype(float)
aux = sum(misclass)/len(X)
print ("Misclassified samples: %s " % aux)

# DT feature selection
importances = dt.feature_importances_
print "Selecting features..."
indices = np.nonzero(importances)
tr_X=tr_X[:,indices]
tr_X=np.copy(tr_X[:,0])
X=X[:,indices]
X=np.copy(X[:,0])

# Build decision tree classifier after removing 0-importance features
print "Building DT..."
dt = tree.DecisionTreeClassifier(criterion='entropy',max_depth=8)
dt.fit(tr_X, tr_y)
res = dt.predict(X)
res = res.astype(int)
res = np.reshape(res, len(res))

# Inverting labels and predictions to obtain metrics with "0-normal traffic" as positive class  
res = np.logical_not(res)
y = np.logical_not(y)

misclass = np.bitwise_xor(res, y)
goodclass = np.logical_not(misclass)
tp = sum(np.bitwise_and(res, y)) 
fp = sum(np.bitwise_and(misclass, res))
tn = sum(np.bitwise_and(goodclass, np.logical_not(res)))
fn = sum(np.bitwise_and(misclass, np.logical_not(res)))
misclass = misclass.astype(float)
aux = sum(misclass)/len(X)
print ("Normal traffic -0- as positive class")
print ("Misclassified samples: %s " % aux)
print ("Total: %s, TP: %s, FP: %s, TN: %s, FN: %s" % (len(X), tp, fp, tn, fn))
auc = roc_auc_score(y, res)
acc = accuracy_score(y, res) 
prec = precision_score(y, res)
recall = recall_score(y, res)
f1 = f1_score(y, res)
print ("Acc: %s, Prec: %s, Recall: %s, F1: %s, ROC: %s" % (acc, prec, recall, f1, auc))	

# Inverting labels and predictions again
res = np.logical_not(res)
y = np.logical_not(y)

node_indicator_tr = dt.decision_path(tr_X)
leave_id_tr = dt.apply(tr_X)
node_indicator = dt.decision_path(X)
leave_id = dt.apply(X)
feature = dt.tree_.feature
threshold = dt.tree_.threshold

feature = feature+2
indices = np.nonzero(feature)
feature = feature-2
feats = np.take(feature, indices)
thres = np.take(threshold, indices)
feats = np.copy(feats[0])
thres = np.copy(thres[0])

XX = np.zeros((len(tr_X),len(feats)))
Xv = np.zeros((len(tr_X),len(feats)))
Xpl= np.zeros((len(tr_X),len(X[0])))
print "<"
print "Calculating train distances..."
print ("Total %s thousand-samples (.):" % str(int(len(tr_X)/1000)))
for i in range(0,len(tr_X)):
	if (i % 1000 == 0):
		sys.stdout.write('.')
		sys.stdout.flush()
	node_index = node_indicator_tr.indices[node_indicator_tr.indptr[i]:node_indicator_tr.indptr[i + 1]]
	for node_id in node_index:
	    if leave_id_tr[i] != node_id:
		Xpl[i,feature[node_id]]=threshold[node_id]
	for j in range(0,len(feats)):
		aux=thres[j]-tr_X[i, feats[j]]
		a=np.copy(tr_X[i])
		b=np.copy(tr_X[i])
		a[feats[j]]=a[feats[j]]+1.0001*aux
		ap = dt.predict(a.reshape(1, -1))
		bp = dt.predict(b.reshape(1, -1))
		if ap != bp:
			XX[i,j]=abs(aux)
			Xv[i,j]=1
Xpl_uniq = np.unique(Xpl, axis=0)

print "<"
print "Calculating medians..."
median_nodes = np.zeros(len(feats))
Xd_trans = XX.transpose()
Xv_trans = Xv.transpose()
print ("Total %s decision-leaves (.):" % str(int(len(feats))))
for i in range(0,len(feats)):
	sys.stdout.write('.')
	sys.stdout.flush()
	node_i_dist = Xd_trans[i] 
	node_i_val = Xv_trans[i]
	indices = np.nonzero(node_i_val)
	if len(indices[0])>0:
		arr_aux = np.take(node_i_dist, indices)
		median_nodes[i] = np.median(arr_aux)	

Xdist_node = np.zeros(len(X))
Xfc = np.zeros(len(X))
print "<"
print "Calculating test distances..."
print ("Total %s thousand-samples (.):" % str(int(len(X)/1000)))
for i in range(0,len(X)):
	Xcomp=np.zeros(len(X[0]))
	Xdists=np.zeros(len(X[0]))
	Val_coor_ex=np.zeros(len(X[0]))
	Xfcsamp=np.zeros(len(X[0]))	
	if (i % 1000 == 0):
		sys.stdout.write('.')
		sys.stdout.flush()
	dist_to_points = distance.cdist(X[i].reshape(1, -1), Xpl_uniq, 'euclidean')
	ind = np.argmin(dist_to_points)
	sel_point=Xpl_uniq[ind]
	for j in range(0,len(X[0])):
		ind_th = np.where(thres==sel_point[j])
		ind_ft = np.where(feats==j)
		k = np.intersect1d(ind_th,ind_ft)
		if k.size:
			k=np.copy(k[0])
			Xcomp[j]=np.power((X[i,j]-sel_point[j])/median_nodes[k],2)
	euc_dist = np.sqrt(sum(Xcomp))
	for j in range(0,len(feats)):
		aux=thres[j]-X[i, feats[j]]
		a=np.copy(X[i])
		b=np.copy(X[i])
		a[feats[j]]=a[feats[j]]+1.0001*aux
		ap = dt.predict(a.reshape(1, -1))
		bp = dt.predict(b.reshape(1, -1))
		if ap != bp:
			aux2=abs(aux)/median_nodes[j]
			if Val_coor_ex[feats[j]]==1:
				if Xdists[feats[j]] > aux2:
					Xdists[feats[j]] = aux2
					Xfcsamp[feats[j]] = j
			else:
				Xdists[feats[j]] = aux2
				Xfcsamp[feats[j]] = j
			Val_coor_ex[feats[j]]=1
	indices = np.nonzero(Val_coor_ex)
	th_dist = np.copy(euc_dist)
	Xfc[i] = -1
	if indices[0].size:
		Xshort = np.take(Xdists, indices)
		th_dist = min(Xshort[0])
		index = np.argmin(Xshort[0])
		Xfc[i] = Xfcsamp[index]
	Xdist_node[i]= min(euc_dist,th_dist)

feature = np.copy(feats)
threshold = np.copy(thres)
print "<"
print "Saving memberships into a csv file..."
np.savetxt(config_files['membership(out)'], Xdist_node, delimiter=",")
print "Saving predictions into a csv file..."
np.savetxt(config_files['predictions(out)'], res, fmt='%i', delimiter=",")
print "Saving thresholds into a csv file..."
np.savetxt(config_files['thresholds(out)'], threshold, delimiter=",")
print "Saving feature_map into a csv file..."
np.savetxt(config_files['features_linked_to_thresholds(out)'], feature, fmt='%i', delimiter=",")
print "Saving feature-importance into a csv file..."
np.savetxt(config_files['feature_importance(out)'], importances, delimiter=",")
print "Saving feature-closest-indices into a csv file..."
np.savetxt(config_files['feature_closest(out)'], Xfc, fmt='%i', delimiter=",")

