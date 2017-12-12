import os
import numpy as np
import cv2

def extract_local_descriptors(img_dir):
	list_decriptors = []
	list_img_labels = []
	# set descriptor method
	desc_method = cv2.SURF()
	#desc_method = cv2.SIFT()
	# get image filenames inside the image directory input
	img_files = [ f for f in os.listdir(img_dir)]
	img_files.sort()
	image_size = 300
	for img_file in img_files:
		arr_filename = img_file.split('_') # asuming that the format is class_imagenumber.jpg
		# obtain label
		label = int(arr_filename[0])
		img_path = img_dir + img_file
		img = cv2.imread(img_path, 0)
		img = cv2.resize(img, (image_size, image_size))
		# detect points of interest and compute descriptors
		kp, descriptors = desc_method.detectAndCompute(img, None)
		list_decriptors.append(descriptors)
		list_img_labels.append(label)
		print "img_path {} npoints {}".format(img_path, descriptors.shape[0])

	return list_decriptors, list_img_labels



