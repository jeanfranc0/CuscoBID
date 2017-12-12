#!/usr/bin/python2.7
import argparse
import numpy as np
from random import shuffle
from sklearn.cluster import KMeans
import cv2
import bovw_utils

def merge_datasets(list_datasets):
	output_nsamples = sum([dataset.shape[0] for dataset in list_datasets])
	nfeats = list_datasets[0].shape[1]
	merged_dataset = np.zeros((output_nsamples, nfeats))
	output_index = 0
	for dataset in list_datasets:
		merged_dataset[output_index:output_index+dataset.shape[0], :] = dataset
		output_index += dataset.shape[0]
	return merged_dataset


def merge_datasets_from_indices_list(list_datasets, list_indices):
	nindices = len(list_indices)
	output_nsamples = sum([list_datasets[list_indices[i]].shape[0] for i in range(nindices)])
	nfeats = list_datasets[0].shape[1]
	merged_dataset = np.zeros((output_nsamples, nfeats))
	output_index = 0
	for dataset_index in list_indices:
		dataset = list_datasets[dataset_index]
		merged_dataset[output_index:output_index+dataset.shape[0], :] = dataset
		output_index += dataset.shape[0]
	return merged_dataset


def random_codebook(dataset, codebook_size):
	random_samples = np.arange(dataset.shape[0])
	np.random.shuffle(random_samples)
	# select top "codebook_size" elements from the shuffle list
	selected_indices = random_samples[:codebook_size]
	codebook = dataset[selected_indices, :]
	return codebook


def kmeans_codebook(dataset, codebook_size):
	kmeans_model = KMeans(n_clusters=codebook_size) #  max_iter=50
	kmeans_model.fit(dataset)
	codebook = kmeans_model.cluster_centers_
	return codebook


def kmeans_codebook_from_every_class(list_datasets, labels, codebook_size, indices_per_class):
	num_labels = len(set(labels))
	# defined number of elements (for the codebook) to be selected from every class
	nsamples_per_label = codebook_size / num_labels
	print "codebook size per class {}".format(nsamples_per_label)
	# select samples for the codebook from every image class
	list_partial_codebooks = []
	for i in range(num_labels):
		merged_dataset = merge_datasets_from_indices_list(list_datasets, indices_per_class[i])
		print merged_dataset.shape
		print "nindices per class {} : {}".format(i+1, len(indices_per_class[i]))
		if i == (num_labels - 1):
			nsamples_per_label = codebook_size - (num_labels-1)*nsamples_per_label
		list_partial_codebooks.append(kmeans_codebook(merged_dataset, nsamples_per_label))
	codebook = merge_datasets(list_partial_codebooks)
	return codebook


def stratified_kmeans_codebook(list_datasets, labels, codebook_size):
	num_datasets = len(list_datasets)
	num_labels = len(set(labels))
	# get dataset indices (in "list_datasets") for every class
	indices_per_label = [[] for i in range(num_labels)]
	for i in range(num_datasets):
		indices_per_label[labels[i]-1].append(i)
	return kmeans_codebook_from_every_class(list_datasets, labels, codebook_size, indices_per_label)


def fast_stratified_kmeans_codebook(list_datasets, labels, codebook_size, max_nimages_per_class):
	num_datasets = len(list_datasets)
	num_labels = len(set(labels))
	# get dataset indices (in "list_datasets") for every class
	indices_per_label = [[] for i in range(num_labels)]
	for i in range(num_datasets):
		indices_per_label[labels[i]-1].append(i)
	# randomly select some samples per class
	selected_indices_per_label = []
	for j in range(num_labels):
		shuffle(indices_per_label[j])
		selected_indices_per_label.append(indices_per_label[j][:max_nimages_per_class])

	return kmeans_codebook_from_every_class(list_datasets, labels, codebook_size, selected_indices_per_label)


def main():
	parser = argparse.ArgumentParser()
	parser.add_argument("img_dir", type=str, help="Directory of images")
	parser.add_argument("codebook_size", type=int, help="Size of the dictionary")
	parser.add_argument("codebook_method", type=str, help="Codebook method", choices=['random', 'kmeans', 'st_kmeans', 'fast_st_kmeans'])
	parser.add_argument("output_codebook", type=str, help="Output codebook filename")
	args = parser.parse_args()

	img_dir = args.img_dir
	codebook_size = args.codebook_size
	output_codebook = args.output_codebook
	codebook_method = args.codebook_method

	# read images and extract descriptors
	list_decriptors, list_img_labels = bovw_utils.extract_local_descriptors(img_dir)

	total_points = sum([dataset.shape[0] for dataset in list_decriptors])

	nfeats = list_decriptors[0].shape[1] # number of features
	
	print "Feature extraction Ok (nfeats {}, total_points {})".format(nfeats, total_points)

	# obtain codebook
	if codebook_method == "random":
		merged_dataset = merge_datasets(list_decriptors)
		list_decriptors = None
		codebook = random_codebook(merged_dataset, codebook_size)
	elif codebook_method == "kmeans":
		merged_dataset = merge_datasets(list_decriptors)
		list_decriptors = None
		codebook = kmeans_codebook(merged_dataset, codebook_size)
	elif codebook_method == "st_kmeans":
		codebook = stratified_kmeans_codebook(list_decriptors, list_img_labels, codebook_size)
	else:
		codebook = fast_stratified_kmeans_codebook(list_decriptors, list_img_labels, codebook_size, 50)

	np.save(output_codebook, codebook)

if __name__ == "__main__":
	main()
