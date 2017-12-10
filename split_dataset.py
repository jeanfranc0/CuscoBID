#!/usr/bin/python2.7
import os
import sys
import argparse
import time
import numpy as np
import random
from os.path import basename

def main():

	parser = argparse.ArgumentParser()
	parser.add_argument("dataset_path", type=str, help="Directory of input images")
	parser.add_argument("nsplits", type=int, help="Number of splits")
	parser.add_argument("perc_train", type=float, help="Percentage of train samples")
	parser.add_argument("output_path", type=str, help="Output path")
	args = parser.parse_args()

	# read parameters
	dataset_path = args.dataset_path
	nsplits = args.nsplits
	perc_train = args.perc_train
	output_path = args.output_path

	image_filenames = [f for f in os.listdir(dataset_path) if f.endswith(".jpg")]
	image_filenames.sort()
	nimages = len(image_filenames)

	image_basenames = []
	for filename in image_filenames:
		image_basenames.append(os.path.splitext(filename)[0])

	ntrainsamples = int(round(nimages * perc_train))
	ntestsamples = nimages - ntrainsamples

	print "ntrainsamples {} ntestsamples {}".format(ntrainsamples, ntestsamples)

	random.seed(213344)
	np.random.seed(213344)
	for it in range(1,nsplits+1):

		random_list = np.arange(nimages)
		np.random.shuffle(random_list)  
		#random_list = random_list.tolist()
		train_samples_idxs = random_list[:ntrainsamples]
		test_samples_idxs = random_list[ntrainsamples:]

		train_filenames = []
		test_filenames = []
		for i in range(ntrainsamples):
			print image_basenames[train_samples_idxs[i]]
			train_filenames.append(image_basenames[train_samples_idxs[i]])
		
		for i in range(ntestsamples):
			print image_basenames[test_samples_idxs[i]]
			test_filenames.append(image_basenames[test_samples_idxs[i]])


		output_split_path = "{}/split{}/".format(output_path, str(int(it)))
		if not os.path.exists(output_split_path):
			os.makedirs(output_split_path)

		output_train_path = "{}/train/".format(output_split_path)
		output_test_path = "{}/test/".format(output_split_path)
		
		if not os.path.exists(output_train_path):
			os.makedirs(output_train_path)
		if not os.path.exists(output_test_path):
			os.makedirs(output_test_path)

		for i in range(ntrainsamples):
			input_image_path = "{}/{}.jpg".format(dataset_path, train_filenames[i])
			image_train_path = "{}/{}.jpg".format(output_train_path, train_filenames[i])
			os.symlink(os.path.abspath(input_image_path), os.path.abspath(image_train_path))

		for i in range(ntestsamples):
			input_image_path = "{}/{}.jpg".format(dataset_path, test_filenames[i])
			image_test_path = "{}/{}.jpg".format(output_test_path, test_filenames[i])
			os.symlink(os.path.abspath(input_image_path), os.path.abspath(image_test_path))
	
	
if __name__ == '__main__':
    main()