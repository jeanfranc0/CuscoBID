#!/usr/bin/python2.7
import os
import argparse
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import cv2
import bovw_utils

def hard_coding(dataset, codebook):
	# 1NN
	codebook_size = len(codebook)
	codebook_labels = np.arange(codebook_size) + 1
	neigh = KNeighborsClassifier(n_neighbors=1)
	neigh.fit(codebook, codebook_labels)
	# hard assignment
	dataset_codebook_labels = neigh.predict(dataset)
	return dataset_codebook_labels

def pooling(list_codebook_labels, codebook_size):
	bovw_features, bin_edges = np.histogram(list_codebook_labels, bins=range(1,codebook_size+2))
	bovw_features = bovw_features.astype(np.float64)
	bovw_features = bovw_features / float(codebook_size)
	return bovw_features

def get_bovw_descriptors(list_local_decriptors, codebook):
	codebook_size = len(codebook)
	print codebook_size
	print len(list_local_decriptors)
	bovw_dataset = np.zeros((len(list_local_decriptors), codebook_size))
	index = 0
	for local_descriptors in list_local_decriptors:
		list_codebook_labels = hard_coding(local_descriptors, codebook)
		bovw_dataset[index][:] = pooling(list_codebook_labels, codebook_size)
		index += 1
	return bovw_dataset

def main():
	parser = argparse.ArgumentParser()
	parser.add_argument("img_dir", type=str, help="Directory of images")
	parser.add_argument("codebook_filename", type=str, help="Codebook file (*.npy)")
	parser.add_argument("output_bovw_filename", type=str, help="Output BOVW descriptors filename (*.npy)")
	parser.add_argument("output_labels_filename", type=str, help="labels filename (*.npy)")
	args = parser.parse_args()

	img_dir = args.img_dir
	codebook_filename = args.codebook_filename
	output_bovw_filename = args.output_bovw_filename
	output_labels_filename = args.output_labels_filename

	# read codebook from file
	codebook = np.load(codebook_filename)

	# read images and extract local descriptors
	list_local_decriptors, list_img_labels = bovw_utils.extract_local_descriptors(img_dir)

	# extract bag of visual words (BOVW) descriptors
	bovw_dataset = get_bovw_descriptors(list_local_decriptors, codebook)

	list_img_labels = np.array(list_img_labels).astype(np.uint32)

	np.save(output_bovw_filename, bovw_dataset)
	np.save(output_labels_filename, list_img_labels)

if __name__ == "__main__":
	main()
