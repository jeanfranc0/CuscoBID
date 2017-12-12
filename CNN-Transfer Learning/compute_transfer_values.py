import argparse
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import time
from datetime import timedelta
import os

from PIL import Image
import models_test
from datetime import timedelta
from utils import plot_2D_images, new_fc_layer
import inception
from inception import transfer_values_cache

# global variables

num_channels = 3
standard_size = None
angles_array = [-5,10]
num_color_transf = 2
batch_size_for_transfer_comp = 50

session = None
image = None
wrap_pre_process_image = None

model = None
data_aug = None
transfer_len = 0

# ******************* Resize pad and crop functions *********************
# Resize image in such a way that the maximum width or height is equal to standard_size
def standarize_image_size(pil_input_img):
	small_height = 0
	small_width  = 0
	input_img_width, input_img_height = pil_input_img.size
	if input_img_height > input_img_width:
		small_height = int(standard_size)
		small_width = int((float(input_img_width) / input_img_height) * standard_size)
	else:
		small_width = int(standard_size)
		small_height = int((float(input_img_height) / input_img_width) * standard_size)
	pil_small_img = pil_input_img.resize((small_width, small_height), Image.ANTIALIAS)
	return pil_small_img

# Fill with zeros to obtain a squared image
def pad_with_zeros(pil_img):
	old_size = pil_img.size
	width, height = pil_img.size
	max_size = max(width, height)
	new_size = (max_size, max_size)
	pil_new_img = Image.new("RGB", new_size)
	pil_new_img.paste(pil_img, ((new_size[0]-old_size[0])/2, 
						(new_size[1]-old_size[1])/2))
	return pil_new_img

# Resize the image to "standard size" and fill with zeros to get a squared image
def standarize_and_pad_with_zeros(pil_img):
	pil_std_image = standarize_image_size(pil_img)
	pil_pad_image = pad_with_zeros(pil_std_image)
	return pil_pad_image

#*********** Functions to compute train and test transfer values **************

# Tensor flow operations to obtain a new image with random color transformations
def generate_random_color_transformation(image):
	image = tf.image.random_hue(image, max_delta=0.05)
	image = tf.image.random_contrast(image, lower=0.8, upper=1.0)
	image = tf.image.random_brightness(image, max_delta=0.2)
	image = tf.image.random_saturation(image, lower=0.0, upper=2.0)
	return image

# compute multiple color transformations
def compute_color_transformations(pil_pad_img, cls, offset_index_image, images_dataset, cls_dataset):
	global session
	pad_img = np.asarray(pil_pad_img)
	for i in xrange(num_color_transf):
		feed_dict = { image :  pad_img}
		new_img = session.run(wrap_pre_process_image, feed_dict)
		images_dataset[offset_index_image + i, :, :, :] = new_img
		cls_dataset[offset_index_image + i] = cls

		#if offset_index_image == 1:
		#	plt.imshow(np.asarray(pil_pad_img , dtype=np.uint8))
		#	plt.imshow(np.asarray(new_img, dtype=np.uint8))
		#	plt.show()

	return num_color_transf

# comute rotation and color transformations
def compute_image_transformations(orig_pil_img, cls, offset_index_image, images_train, cls_train):
	num_transformations = len(angles_array)
	current_index = offset_index_image
	for i in xrange(num_transformations):
		pil_img = orig_pil_img.copy()
		size_pil_img = pil_img.size
		#print "size_pil_img {}".format(size_pil_img)
		pil_new_img = pil_img.rotate(angles_array[i]) # FIXME rotate the image
		size_pil_new_image = pil_new_img.size
		#print "size_pil_new_image {}".format(size_pil_new_image)
		pil_pad_img = pad_with_zeros(pil_new_img)
		pad_img = np.asarray(pil_pad_img , dtype=np.float32)
		images_train[current_index, :, :, :] = pad_img
		cls_train[current_index] = cls
		current_index += 1
		num_added = compute_color_transformations(pil_pad_img, cls, current_index, images_train, cls_train)
		current_index += num_added

	return (current_index - offset_index_image)

# compute train transfer values in batches
def compute_batch_train_transfer_values(dataset_dir, list_filenames):
	# compute number of original images and total number of images (after transformations)
	nimages = len(list_filenames)
	if data_aug == "si":		
		num_rot_transf = len(angles_array)
		final_nimages = nimages * (num_rot_transf + 1) * (num_color_transf + 1)
		# allocate memory
		transfer_values_train = np.zeros((final_nimages, transfer_len), dtype=np.float32)
		images_train = np.zeros((final_nimages, standard_size, standard_size, num_channels), dtype=np.float32)
		cls_train = np.zeros(final_nimages, dtype=np.int)

		current_index = 0
		for image_filename in list_filenames:
			arr_filename = image_filename.split('_') # asuming that the format is class_imagenumber.jpg
			# obtain class
			cls = int(arr_filename[0]) - 1
			image_path = dataset_dir + image_filename
			pil_image = Image.open(image_path)
			#print image_path
			size_pil_image = pil_image.size
			#print "image_filename size_pil_image {}".format(size_pil_image)
			# set image information
			pil_std_image = standarize_image_size(pil_image)
			pil_pad_image = pad_with_zeros(pil_std_image)
			pad_image = np.asarray(pil_pad_image , dtype=np.float32)
			images_train[current_index, :, :, :] = pad_image
			# set class information
			size_pil_std_image = pil_std_image.size
			#print "size_pil_std_image {}".format(size_pil_std_image)
			cls_train[current_index] = cls
			current_index += 1
			num_added = compute_color_transformations(pil_pad_image, cls, current_index, images_train, cls_train)
			current_index += num_added
			# include rotated images in the dataset
			num_added = compute_image_transformations(pil_std_image, cls, current_index, images_train, cls_train)
			current_index += num_added
	else:
		#num_rot_transf = len(angles_array)
		#final_nimages = nimages * (num_rot_transf + 1) * (num_color_transf + 1)
		# allocate memory
		transfer_values_train = np.zeros((nimages, transfer_len), dtype=np.float32)
		images_train = np.zeros((nimages, standard_size, standard_size, num_channels), dtype=np.float32)
		cls_train = np.zeros(nimages, dtype=np.int)

		current_index = 0
		for image_filename in list_filenames:
			arr_filename = image_filename.split('_') # asuming that the format is class_imagenumber.jpg
			# obtain class
			cls = int(arr_filename[0]) - 1
			image_path = dataset_dir + image_filename
			pil_image = Image.open(image_path)
			#print image_path
			size_pil_image = pil_image.size
			#print "image_filename size_pil_image {}".format(size_pil_image)
			# set image information
			pil_std_image = standarize_image_size(pil_image)
			pil_pad_image = pad_with_zeros(pil_std_image)
			pad_image = np.asarray(pil_pad_image , dtype=np.float32)
			images_train[current_index, :, :, :] = pad_image
			# set class information
			size_pil_std_image = pil_std_image.size
			#print "size_pil_std_image {}".format(size_pil_std_image)
			cls_train[current_index] = cls
			current_index += 1
			#num_added = compute_color_transformations(pil_pad_image, cls, current_index, images_train, cls_train)
			#current_index += num_added
			# include rotated images in the dataset
			#num_added = compute_image_transformations(pil_std_image, cls, current_index, images_train, cls_train)
			#current_index += num_added

	# start time
	start_time = time.time()
	# compute transfer values using model transfer_values function
	if model == "vgg16":
		for i in xrange(nimages):
			transfer_values_train[i] = models_test.vgg16_transfer_values(image=images_train[i])
	elif model == "vgg19":
		for i in xrange(nimages):
			transfer_values_train[i] = models_test.vgg19_transfer_values(image=images_train[i])
	elif model == "inception":
		for i in xrange(nimages):
			transfer_values_train[i] = inception.Inception().transfer_values(image=images_train[i])
	elif model == "resnet":
		for i in xrange(nimages):
			transfer_values_train[i] = models_test.resnet50_transfer_values(image=images_train[i])
	else:
		for i in xrange(nimages):
			transfer_values_train[i] = models_test.xception_transfer_values(image=images_train[i])
	# end time
	end_time = time.time()
	time_dif = end_time - start_time
	print("Time usage: " + str(timedelta(seconds=int(round(time_dif)))))

	return transfer_values_train, cls_train

# compute test transfer values in batches
def compute_batch_test_transfer_values(dataset_dir, list_filenames):
	# compute number of original images
	nimages = len(list_filenames)
	# allocate memory
	transfer_values_test = np.zeros((nimages, transfer_len), dtype=np.float32)
	images_test = np.zeros((nimages, standard_size, standard_size, num_channels), dtype=np.float32)
	cls_test = np.zeros(nimages, dtype=np.int)

	current_index = 0
	for image_filename in list_filenames:
		arr_filename = image_filename.split('_') # asuming that the format is class_imagenumber.jpg
		# obtain class
		cls = int(arr_filename[0]) - 1
		image_path = dataset_dir + image_filename
		pil_image = Image.open(image_path)
		# set image information
		pil_std_image = standarize_image_size(pil_image)
		pil_pad_image = pad_with_zeros(pil_std_image)
		pad_image = np.asarray(pil_pad_image , dtype=np.float32)
		images_test[current_index, :, :, :] = pad_image
		# set class information
		cls_test[current_index] = cls
		current_index += 1

	# start time
	start_time = time.time()
	# compute transfer values using model transfer_values function
	if model == "vgg16":
		for i in xrange(nimages):
			transfer_values_test[i] = models_test.vgg16_transfer_values(image=images_test[i])
	elif model == "vgg19":
		for i in xrange(nimages):
			transfer_values_test[i] = models_test.vgg19_transfer_values(image=images_test[i])
	elif model == "inception":
		for i in xrange(nimages):
			transfer_values_test[i] = inception.Inception().transfer_values(image=images_test[i])
	elif model == "resnet":
		for i in xrange(nimages):
			transfer_values_test[i] = models_test.resnet50_transfer_values(image=images_test[i])
	else:
		for i in xrange(nimages):
			transfer_values_test[i] = models_test.xception_transfer_values(image=images_test[i])
	# end time
	end_time = time.time()
	time_dif = end_time - start_time
	print("Time usage: " + str(timedelta(seconds=int(round(time_dif)))))
	return transfer_values_test, cls_test


# ************************ Functions to compute transfer values ********************
def compute_train_transfer_values(dataset_dir):
	# read filenames
	image_filenames = [f for f in os.listdir(dataset_dir) if f.endswith(".jpg")]
	image_filenames.sort()
	nimages = len(image_filenames)
	if data_aug == "si":
		# compute number of original images and total number of images (after transformations)
		num_rot_transf = len(angles_array)
		num_total_transf = (num_rot_transf + 1) * (num_color_transf + 1)
		final_nimages = nimages * num_total_transf
		# allocate memory
		dataset_train = np.zeros((final_nimages, transfer_len), dtype=np.float32)
		cls_train = np.zeros(final_nimages, dtype=np.int)
		i = 0 
		current_index = 0
		while i < nimages:
			# define batch filenames
			j = min(i+batch_size_for_transfer_comp, final_nimages)
			batch_filenames = image_filenames[i:j]
			print "train transfer values {}".format(current_index)
			# compute transfervalues from batch of images
			batch_dataset, batch_cls = compute_batch_train_transfer_values(dataset_dir, batch_filenames)
			dataset_train[current_index: current_index + len(batch_cls)] = batch_dataset
			cls_train[current_index: current_index + len(batch_cls)] = batch_cls
			current_index += len(batch_cls)
			i = j
	else:
		# allocate memory
		dataset_train = np.zeros((nimages, transfer_len), dtype=np.float32)
		cls_train = np.zeros(nimages, dtype=np.int)
		i = 0 
		current_index = 0
		while i < nimages:
			# define batch filenames
			j = min(i+batch_size_for_transfer_comp, nimages)
			batch_filenames = image_filenames[i:j]
			print "train transfer values {}".format(current_index)
			# compute transfervalues from batch of images
			batch_dataset, batch_cls = compute_batch_train_transfer_values(dataset_dir, batch_filenames)
			dataset_train[current_index: current_index + len(batch_cls)] = batch_dataset
			cls_train[current_index: current_index + len(batch_cls)] = batch_cls
			current_index += len(batch_cls)
			i = j

	return dataset_train, cls_train

def compute_test_transfer_values(dataset_dir):
	# read filenames
	image_filenames = [f for f in os.listdir(dataset_dir) if f.endswith(".jpg")]
	image_filenames.sort()
	nimages = len(image_filenames)

	# allocate memory
	dataset_test = np.zeros((nimages, transfer_len), dtype=np.float32)
	cls_test = np.zeros(nimages, dtype=np.int)

	i = 0 
	current_index = 0
	while i < nimages:
		# define batch filenames
		j = min(i+batch_size_for_transfer_comp, nimages)
		batch_filenames = image_filenames[i:j]

		print "test transfer values {}".format(current_index)
		# compute transfervalues from batch of images
		batch_dataset, batch_cls = compute_batch_test_transfer_values(dataset_dir, batch_filenames)
		dataset_test[current_index: current_index + len(batch_cls)] = batch_dataset
		cls_test[current_index: current_index + len(batch_cls)] = batch_cls

		current_index += len(batch_cls)

		i = j

	return dataset_test, cls_test

def main():
	global session, image, wrap_pre_process_image, model, transfer_len, data_aug, standard_size

	parser = argparse.ArgumentParser()
	parser.add_argument("img_dir", type=str, help="Directory of images")
	parser.add_argument("dataset_type", type=str, help="dataset type", choices=['train', 'test'])
	parser.add_argument("model_type", type=str, help="model type", choices=['vgg16', 'vgg19', 'resnet', 'xception','inception'])
	parser.add_argument("data_augmentation", type=str, help="data augmentation", choices=['si', 'no'])
	parser.add_argument("output_data", type=str, help="Output transfer values (.npy)")
	parser.add_argument("output_cls", type=str, help="Output classes (.npy)")

	args = parser.parse_args()

	img_dir = args.img_dir
	dataset_type = args.dataset_type
	model_type = args.model_type
	data_augmentation = args.data_augmentation
	output_data = args.output_data
	output_cls = args.output_cls
	
	# start time
	start_time = time.time()
	# get model
	if model_type == "vgg16":
		transfer_len = models_test.vgg16_transfer_len()
		print "transfer_len {}".format(transfer_len)
		model = model_type
		standard_size = 224

	elif model_type == "vgg19":
		transfer_len = models_test.vgg19_transfer_len()
		print "transfer_len {}".format(transfer_len)
		model = model_type
		standard_size = 224

	elif model_type == "inception":
		model = inception.Inception()
		transfer_len = model.transfer_len
		print "transfer_len {}".format(transfer_len)
		model = model_type
		standard_size = 224

	elif model_type == "resnet":
		transfer_len = models_test.resnet50_transfer_len()
		print "transfer_len {}".format(transfer_len)
		model = model_type
		standard_size = 224

	else:
		transfer_len = models_test.xception_transfer_len()
		print "transfer_len {}".format(transfer_len)
		model = model_type
		standard_size = 299

	if dataset_type == "train":
		if data_augmentation == "si":
			# initialize session
			session = tf.Session()
			image = tf.placeholder(tf.float32, shape=[standard_size, standard_size, num_channels], name = 'image')
			wrap_pre_process_image = generate_random_color_transformation(image)
			data_aug = data_augmentation
			transfer_values_train, cls_train = compute_train_transfer_values(img_dir)
			print transfer_values_train.shape

			# save feature vector and class
			np.save(output_data , transfer_values_train)
			np.save(output_cls , cls_train)
			print  "train transfer values saved"
			session.close()
		else:
			# initialize session
			#session = tf.Session()
			#image = tf.placeholder(tf.float32, shape=[standard_size, standard_size, num_channels], name = 'image')
			#wrap_pre_process_image = generate_random_color_transformation(image)
			data_aug = data_augmentation
			transfer_values_train, cls_train = compute_train_transfer_values(img_dir)
			print transfer_values_train.shape

			np.save(output_data , transfer_values_train)
			np.save(output_cls , cls_train)
			print  "train transfer values saved"
			#session.close()

	else:
		transfer_values_test, cls_test = compute_test_transfer_values(img_dir)
		print transfer_values_test.shape
		np.save(output_data, transfer_values_test)
		np.save(output_cls , cls_test)
		print  "test transfer values saved"
	
	# end time
	end_time = time.time()
	time_dif = end_time - start_time
	print("Time usage: " + str(timedelta(seconds=int(round(time_dif)))))


if __name__ == "__main__":
	main()
