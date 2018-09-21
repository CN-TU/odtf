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
from sklearn import tree
import numpy as np
import csv
import fileinput
import sys
import time
import skfuzzy as fuzz

config_files={'training_data(in)':"training_data.csv", 'training_labels(in)':"training_labels.csv", 'testing_data(in)':"testing_data.csv", 'testing_labels(in)':"testing_labels.csv", 'membership(out)':"membership.csv", 'thresholds(out)':"thresholds.csv", 'features_linked_to_thresholds(out)':"features.csv",'predictions(out)':"predictions.csv",'feature_importance(out)':"importance.csv", 'feature_closest(out)':"featclosest.csv", 'fuzzyfication' : "True"}

debug = True # set "debug=True" to show elapsed time 

def print_debug(output_string):
	if debug:
		print(output_string) 

def calc_training_data(i):
	global node_indicator_tr, leave_id_tr, tr_X, tr_X_sorted, thres, threshold	# input variables
	global XX, Xv, Xpl	# output variables
	if (i % 1000 == 0):
		sys.stdout.write('.')
		sys.stdout.flush()
	node_index = node_indicator_tr.indices[node_indicator_tr.indptr[i]:node_indicator_tr.indptr[i + 1]]
	for node_id in node_index:
		if leave_id_tr[i] != node_id:
			Xpl[i,feature[node_id]]=threshold[node_id]
	
	aux_array=thres-tr_X_sorted[i]
	a_array = np.tile(tr_X[i], (len_feats+1,1)) # the last element: ap_array[len_feats] is the  original data point
	
	for j in range(0,len_feats):
		a_array[j, feats[j]]=a_array[j, feats[j]]+1.0001*aux_array[j]
		
	ap_array = dt.predict(a_array)
	
	for j in range(0,len_feats):
		if ap_array[j] != ap_array[len_feats]:
			XX[i,j]=abs(aux_array[j])
			Xv[i,j]=1
			
def calc_test_data(i):
	global X, Xpl_uniq, feats, X_sorted, median_nodes, indices, thres 		# input variables
	global Xfc, Xdist_node 		# output variables
	
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
			if median_nodes[k]==0:
				Xcomp[j]=1
			else:
				Xcomp[j]=np.power((X[i,j]-sel_point[j])/median_nodes[k],2)
	euc_dist = np.sqrt(sum(Xcomp))
	
	aux_array=thres-X_sorted[i]
	a_array = np.tile(X[i], (len_feats+1,1)) # the last element: ap_array[len_feats] is the  original data point
	
	for j in range(0,len_feats):
		a_array[j, feats[j]]=a_array[j, feats[j]]+1.0001*aux_array[j]
		
	ap_array = dt.predict(a_array)
	
	for j in range(0,len_feats):
		if ap_array[j] != ap_array[len_feats]:
			if median_nodes[j]==0:
				aux2=1
			else:
				aux2=abs(aux_array[j])/median_nodes[j]
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
	if indices[0].size:
		Xshort = np.take(Xdists, indices)
		th_dist = min(Xshort[0])
		index = np.argmin(Xshort[0])
		Xfc[i] = Xfcsamp[index]
	Xdist_node[i]= min(euc_dist,th_dist)

if debug:
	print(sys.argv)
	start_time = time.time()

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
print "Training set:\t", config_files['training_data(in)']
# Loading the training dataset
reader = csv.reader(open(config_files['training_data(in)'], "rb"), delimiter=",")
trainX = list(reader)
tr_X = np.array(trainX).astype("float")


# Loading the labels of the training dataset 
print "Train. labels:\t", config_files['training_labels(in)']
reader = csv.reader(open(config_files['training_labels(in)'], "rb"), delimiter=",")
trainy = list(reader)
tr_y = np.array(trainy).astype("int")


# Loading the test dataset
print "Test set:\t", config_files['testing_data(in)']
reader = csv.reader(open(config_files['testing_data(in)'], "rb"), delimiter=",")
testX = list(reader)
X = np.array(testX).astype("float")


# Loading the labels of the testing dataset 
print "Test labels:\t", config_files['testing_labels(in)']
reader = csv.reader(open(config_files['testing_labels(in)'], "rb"), delimiter=",")
testy = list(reader)
y = np.array(testy).astype("int")

# Build decision tree classifier
print_debug("--- %s seconds ---" % (time.time() - start_time))

print "Building DT..."
dt = tree.DecisionTreeClassifier(max_depth=8, min_samples_leaf=5, random_state=0) # add random_state=0 to get a deterministic behaviour
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
print_debug("--- %s seconds ---" % (time.time() - start_time))
	
print "Building DT..."
dt = tree.DecisionTreeClassifier(max_depth=8, min_samples_leaf=5, random_state=0) # add random_state=0 to get a deterministic behaviour
dt.fit(tr_X, tr_y)
res = dt.predict(X)
res = res.astype(int)
res = np.reshape(res, len(res))

from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, f1_score 

auc = roc_auc_score(y, res)
acc = accuracy_score(y, res) 
print ("Acc: %.2f, ROC/AUC: %.2f" % (acc, auc))	
prec = precision_score(y, res, average='macro')
recall = recall_score(y, res, average='macro')
f1 = f1_score(y, res, average='macro')
print ("Average   - Prec: %.2f, Recall: %.2f, F1: %.2f" % (prec, recall, f1))	
prec = precision_score(y, res, average='binary', pos_label=0)
recall = recall_score(y, res, average='binary', pos_label=0)
f1 = f1_score(y, res, average='binary', pos_label=0)
print ("Pos_lab:0 - Prec: %.2f, Recall: %.2f, F1: %.2f" % (prec, recall, f1))	
prec = precision_score(y, res, average='binary', pos_label=1)
recall = recall_score(y, res, average='binary', pos_label=1)
f1 = f1_score(y, res, average='binary', pos_label=1)
print ("Pos_lab:1 - Prec: %.2f, Recall: %.2f, F1: %.2f" % (prec, recall, f1))	

print_debug("--- %s seconds ---" % (time.time() - start_time))


	
if config_files['fuzzyfication'] == "True":
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
	len_feats = len(feats)
	thres = np.copy(thres[0])
	
	XX = np.zeros((len(tr_X),len_feats))
	Xv = np.zeros((len(tr_X),len_feats))
	Xpl= np.zeros((len(tr_X),len(X[0])))
	tr_X_sorted=np.zeros((len(tr_X),len_feats))
	X_sorted=np.zeros((len(X),len_feats))
		
	print "<"
	print "Calculating training distances..."
	print ("Total %s thousand-samples (.):" % str(int(len(tr_X)/1000)))
	
	for i in range(0,len_feats):
		tr_X_sorted[:,i]=tr_X[:,feats[i]]
		X_sorted[:,i]=X[:,feats[i]]
	
	for i in range(0,len(tr_X)):
		calc_training_data(i)
		
	Xpl_uniq = np.unique(Xpl, axis=0)
	
	print_debug("--- %s seconds ---" % (time.time() - start_time))
		
	print "<"
	print "Calculating medians..."
	median_nodes = np.zeros(len_feats) 
	Xd_trans = XX.transpose()
	Xv_trans = Xv.transpose()
	print ("Total %s decision-leaves (.):" % str(int(len_feats)))
	for i in range(0,len_feats):
		sys.stdout.write('.')
		sys.stdout.flush()
		node_i_dist = Xd_trans[i] 
		node_i_val = Xv_trans[i]
		indices = np.nonzero(node_i_val)
		if len(indices[0])>0:
			arr_aux = np.take(node_i_dist, indices)
			median_nodes[i] = np.median(arr_aux)	
	
	Xdist_node = np.zeros(len(X))
	Xfc = np.full(len(X), -1)
	
	print_debug("--- %s seconds ---" % (time.time() - start_time))
		
	print "<"
	print "Calculating test distances..."
	print ("Total %s thousand-samples (.):" % str(int(len(X)/1000)))
	for i in range(0,len(X)):
		calc_test_data(i)
	
	print_debug("--- %s seconds ---" % (time.time() - start_time))
	
	feature = np.copy(feats)
	threshold = np.copy(thres)
	print "<"
	print "Saving memberships into a csv file...\t", config_files['membership(out)']
	#Smemb = fuzz.smf(Xdist_node,0,1)[np.newaxis]
	#Omemb = fuzz.smf(Xdist_node,1,3)[np.newaxis]
	#Fmemb = np.concatenate((Smemb.T, Omemb.T), axis=1)
	np.savetxt(config_files['membership(out)'], Xdist_node, delimiter=",")
	#np.savetxt(config_files['membership(out)'], Fmemb, delimiter=",") #Saving fuzzy sets
	
print "Saving predictions into a csv file...\t", config_files['predictions(out)']
np.savetxt(config_files['predictions(out)'], res, fmt='%i', delimiter=",")

if config_files['fuzzyfication'] == "True":
	print "Saving thresholds into a csv file...\t", config_files['thresholds(out)']
	np.savetxt(config_files['thresholds(out)'], threshold, delimiter=",")
	print "Saving feature_map into a csv file...\t", config_files['features_linked_to_thresholds(out)']
	np.savetxt(config_files['features_linked_to_thresholds(out)'], feature, fmt='%i', delimiter=",")
	print "Saving feature-importance into a csv file...\t", config_files['feature_importance(out)']
	np.savetxt(config_files['feature_importance(out)'], importances, delimiter=",")
	print "Saving feature-closest-indices into a csv file...\t", config_files['feature_closest(out)']
	np.savetxt(config_files['feature_closest(out)'], Xfc, fmt='%i', delimiter=",")

print_debug("--- %s seconds ---" % (time.time() - start_time))

