#!/usr/bin/python2.7
import os
import sys
import argparse
import time
import numpy as np
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV

from sklearn.neighbors import KNeighborsClassifier

from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
from sklearn import metrics

from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix



def svm_grid_search(dataset, labels):
	C_s = 10.0 ** np.arange(-1, 3)
	gammas = 10.0 ** np.arange(-1, 3)
	tuned_parameters = [{'kernel': ['rbf'], 'gamma': gammas,'C': C_s}]
	clf = GridSearchCV(svm.SVC(C=1), tuned_parameters, cv=3)
	clf.fit(dataset, labels)
	return (clf.best_params_['C'], clf.best_params_['gamma'])


def linearSVM_grid_search(dataset, labels):
	C_s = 10.0 ** np.arange(-1, 3)
	tuned_parameters = [{'C': C_s}]
	clf = GridSearchCV(svm.LinearSVC(C=1), tuned_parameters, cv=3)
	clf.fit(dataset, labels)
	return clf.best_params_['C']

def main():

	parser = argparse.ArgumentParser()
	parser.add_argument("dataset_train_filename", type=str, help="Dataset train file name (*.npy)")
	parser.add_argument("labels_train_filename", type=str, help="Label train filename (*.npy)")
	parser.add_argument("dataset_test_filename", type=str, help="Dataset test file name (*.npy)")
	parser.add_argument("labels_test_filename", type=str, help="Label test filename (*.npy)")
	parser.add_argument("method", type=str, help="Classifier", choices=['svm', 'linear_svm', 'rf', 'knn'])
	parser.add_argument("output_filename", type=str, help="Predicted output filename")
	args = parser.parse_args()

	dataset_train_filename = args.dataset_train_filename
	labels_train_filename = args.labels_train_filename
	dataset_test_filename = args.dataset_test_filename
	labels_test_filename = args.labels_test_filename
	method = args.method
	output_filename = args.output_filename
	
	train_data = np.load(dataset_train_filename)
	train_labels =  np.load(labels_train_filename).astype(np.int32)	
	test_data = np.load(dataset_test_filename)
	test_labels =  np.load(labels_test_filename).astype(np.int32)
	num_classes = len(np.unique(train_labels))
	num_train = train_data.shape[0]
	
	print "Read dataset Ok"
	print "num_train {}".format(num_train)
	print 'num classes {}'.format(num_classes)
	time_ini = time.clock()

	# create classifier object
	cls = None
	if method == "linear_svm":
		c = linearSVM_grid_search(train_data, train_labels)
		print "Params-> c value {}".format(c)
		cls = svm.LinearSVC(C=c)
	elif method == "svm":
		c , gamma = svm_grid_search(train_data, train_labels)
		print "Params -> C: "+ str(c) + ", gamma: "+str(gamma)
		cls = svm.SVC(C=c, gamma=gamma)
	elif method == "rf":
		print "RF default params"
		cls = RandomForestClassifier(max_depth=50, n_estimators=500)
	else:
		print "KNN"
		cls = KNeighborsClassifier(1)
	
	# train classifier
	cls.fit(train_data, train_labels)
	time_sec = (time.clock() - time_ini)
	print "time train {}".format(time_sec)

	# predict labels of test samples
	time_ini = time.clock()

	pred_test_labels = cls.predict(test_data)
	time_sec = (time.clock() - time_ini)
	print "time classification {}".format(time_sec)
	
	# accuracy, recall, precision, and f1-score 
	# compute overall accuracy 
	acc_test = accuracy_score(test_labels, pred_test_labels)
	print "Overall Accuracy {}: ".format(acc_test)

	# compute overral recall
	recall = recall_score(test_labels, pred_test_labels, average="macro")  
	print "Overall recall {}: ".format(recall)
		
	# compute overral precision
	precision = precision_score(test_labels, pred_test_labels, average="macro")
	print "Overall precision {}: ".format(precision)
	
	# compute overral f1_score
	f1 = f1_score(test_labels, pred_test_labels, average="macro")
	print "Overall f1_score {}: ".format(f1)

	#report	
	print classification_report(test_labels, pred_test_labels, digits=11)

	# compute accuracy for each class
	cmatrix = confusion_matrix(test_labels, pred_test_labels)
	#print cmatrix
	cmatrix = np.transpose(cmatrix) # cols trueLabel and rows predicted
	for i in range(num_classes):
			accuracy_class = 0
			if np.sum(cmatrix[:,i]) > 0:
				accuracy_class = float(cmatrix[i,i]) / float(np.sum(cmatrix[:,i]))
			print "Accuracy class {} : {}".format(i+1, accuracy_class)
	
	np.save(output_filename, pred_test_labels)

if __name__ == '__main__':
	main()
