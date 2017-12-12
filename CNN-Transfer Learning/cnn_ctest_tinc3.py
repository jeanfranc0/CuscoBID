import argparse
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import time
from datetime import timedelta
import os

from PIL import Image

from utils import plot_2D_images, new_fc_layer

from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
from sklearn import metrics

from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix


# set global variables
num_classes = 0
dataset_train = None
cls_train = None
labels_train = None

dataset_test = None
cls_test = None
labels_test = None

session = None
x = None
y_true = None
y_pred_cls = None
optimizer = None
accuracy = None
global_step = None

cls_pred=None
# Configure the network
def configure_network2(dataset, input_dim):
	global x, y_true, num_classes

	num_features = int(input_dim)
	fc_size1 = int(1024)
	fc_size2 = int(1024)
	print "num_classes {} num_features {}".format(num_classes, num_features)
	layer_fc1 = new_fc_layer(input = dataset,
								num_inputs = num_features,
								num_outputs = fc_size1,
								use_relu = False)

	layer_fc2 = new_fc_layer(input = layer_fc1,
								num_inputs = fc_size1,
								num_outputs = fc_size2,
								use_relu = False)

	layer_fc3 = new_fc_layer(input = layer_fc2,
								num_inputs = fc_size2,
								num_outputs = num_classes,
								use_relu = False)

	y_pred = tf.nn.softmax(layer_fc3)

	print y_pred

	cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits = layer_fc3, labels = y_true)

	cost = tf.reduce_mean(cross_entropy)

	return y_pred, cost

def configure_network1(dataset, input_dim):
	global x, y_true, num_classes

	num_features = int(input_dim)
	fc_size1 = int(1024)
	print "num_classes {} num_features {}".format(num_classes, num_features)
	layer_fc1 = new_fc_layer(input = dataset,
								num_inputs = num_features,
								num_outputs = fc_size1,
								use_relu = False)

	layer_fc2 = new_fc_layer(input = layer_fc1,
								num_inputs = fc_size1,
								num_outputs = num_classes,
								use_relu = False)

	y_pred = tf.nn.softmax(layer_fc2)

	print y_pred

	cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits = layer_fc2, labels = y_true)

	cost = tf.reduce_mean(cross_entropy)

	return y_pred, cost


def random_batch(batch_size):
	global dataset_train, labels_train

	# Number of images (transfer-values) in the training-set.
	num_images = len(dataset_train)

	# Create a random index.
	idx = np.random.choice(num_images,
							size=batch_size,
							replace=False)

	# get a set of random transfer values
	x_batch = dataset_train[idx]
	y_batch = labels_train[idx]

	return x_batch, y_batch

# NN training
def optimize(num_iterations, batch_size = 50):
	global optimizer, accuracy, global_step

	start_time = time.time()
	for i in range(num_iterations):
		x_batch, y_true_batch = random_batch(batch_size)
		feed_dict_train = {x: x_batch, y_true: y_true_batch}
		i_global, _ = session.run([global_step, optimizer], feed_dict=feed_dict_train)

		# Print accuracy every 100 iterations
		if (i_global % 100 == 0) or (i == num_iterations - 1):
			batch_acc = session.run(accuracy, feed_dict=feed_dict_train)
			msg = "Global Step: {0:>6}, Training Batch Accuracy: {1:>6.1%}"
			print(msg.format(i_global, batch_acc))

	# Print time
	end_time = time.time()
	time_dif = end_time - start_time
	print("Time usage: " + str(timedelta(seconds=int(round(time_dif)))))

# classify in batches
def predict_cls(dataset, labels, cls_true):
	global y_pred_cls

	test_batch_size = 256
	num_images = len(dataset)
	cls_pred = np.zeros(shape=num_images, dtype=np.int)

	i = 0
	while i < num_images:
		# The ending index for the next batch is denoted j.
		j = min(i + test_batch_size, num_images)
		feed_dict = {x: dataset[i:j], y_true: labels[i:j]}
		#print feed_dict
		# computed predicted values
		cls_pred[i:j] = session.run(y_pred_cls, feed_dict=feed_dict)

		i = j

	# Create a boolean array whether each image is correctly classified.
	correct = (cls_true == cls_pred)

	return correct, cls_pred

def print_test_accuracy():
	global dataset_test, labels_test, cls_test,cls_pred
	# compute predicted classes
	correct, cls_pred = predict_cls(dataset = dataset_test,
						labels = labels_test, cls_true = cls_test)

	print "cls_pred"
	print cls_pred

	print "cls_test"
	print cls_test

	# get classification accuracy and the number of correct classifications.
	acc = correct.mean()
	num_correct = correct.sum()

	# Number of images being classified.
	num_images = len(correct)

	# Print the accuracy.
	msg = "Accuracy on Test-Set: {0:.1%} ({1} / {2})"
	print(msg.format(acc, num_correct, num_images))


def main():
	global session, image, wrap_pre_process_image, model, transfer_len, num_classes
	global dataset_test, labels_test, cls_test
	global dataset_train, labels_train, cls_train
	global x, y_true, y_pred_cls, optimizer, global_step, accuracy,cls_pred

	parser = argparse.ArgumentParser()
	parser.add_argument("file_path_train", type=str, help="File path train dataset")
	parser.add_argument("file_path_train_cls", type=str, help="File path train dataset")
	parser.add_argument("file_path_test", type=str, help="File path test dataset classes")
	parser.add_argument("file_path_test_cls", type=str, help="File path test dataset classes")
	parser.add_argument("file_path_save_model", type=str, help="Save model (.ckpt)")
	args = parser.parse_args()

	file_path_train = args.file_path_train
	file_path_train_cls = args.file_path_train_cls

	file_path_test = args.file_path_test
	file_path_test_cls = args.file_path_test_cls

	model_saved = args.file_path_save_model

	dataset_train = np.load(file_path_train)
	dataset_test = np.load(file_path_test)

	#print dataset_test[0]

	# load class

	cls_train = np.load(file_path_train_cls)
	cls_test = np.load(file_path_test_cls)

	num_train_samples = len(cls_train)
	num_test_samples = len(cls_test)

	num_classes = int(np.max(cls_train) + 1)

	print "num_classes {}".format(num_classes)
	print num_test_samples

	# get labels

	labels_train = np.zeros((num_train_samples, num_classes))
	labels_test = np.zeros((num_test_samples, num_classes))

	for i in xrange(num_train_samples):
		labels_train[i, cls_train[i]] = 1

	for i in xrange(num_test_samples):
		labels_test[i, cls_test[i]] = 1

	input_dim = dataset_train.shape[1]
	print "input_dim {}".format(input_dim)

	# create place holder variables
	x = tf.placeholder(tf.float32, shape=[None, input_dim], name='x')

	y_true = tf.placeholder(tf.float32, shape=[None, num_classes], name='y_true')

	y_true_cls = tf.argmax(y_true, dimension=1)

	y_pred, cost = configure_network2(x, input_dim)

	global_step = tf.Variable(initial_value=0, name='global_step', trainable=False)

	optimizer = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(cost, global_step)

	y_pred_cls = tf.argmax(y_pred, dimension = 1, name = 'argument')

	correct_prediction = tf.equal(y_pred_cls, y_true_cls)

	accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

	# Open Session
	session = tf.Session()	

	session.run(tf.global_variables_initializer())

	logs_path = '/tmp/'
	writer = tf.summary.FileWriter('',graph=tf.get_default_graph())

	#Create a saver object which will save all the variables
	saver = tf.train.Saver()
	
	# optimize and test
	train_batch_size = 16

	print_test_accuracy()

	# accuracy, recall, precision, and f1-score 
	# compute overall accuracy 
	acc_test = accuracy_score(cls_test, cls_pred)
	print "Overall Accuracy {}: ".format(acc_test)

	# compute overral recall
	recall = recall_score(cls_test, cls_pred, average="macro")  
	print "Overall recall {}: ".format(recall)
		
	# compute overral precision
	precision = precision_score(cls_test, cls_pred, average="macro")
	print "Overall precision {}: ".format(precision)
	
	# compute overral f1_score
	f1 = f1_score(cls_test, cls_pred, average="macro")
	print "Overall f1_score {}: ".format(f1)

	optimize(num_iterations =1000, batch_size = train_batch_size)

	print_test_accuracy()
	
	# accuracy, recall, precision, and f1-score 
	# compute overall accuracy 
	acc_test = accuracy_score(cls_test, cls_pred)
	print "Overall Accuracy {}: ".format(acc_test)

	# compute overral recall
	recall = recall_score(cls_test, cls_pred, average="macro")  
	print "Overall recall {}: ".format(recall)
		
	# compute overral precision
	precision = precision_score(cls_test, cls_pred, average="macro")
	print "Overall precision {}: ".format(precision)
	
	# compute overral f1_score
	f1 = f1_score(cls_test, cls_pred, average="macro")
	print "Overall f1_score {}: ".format(f1)

	
	optimize(num_iterations =1000, batch_size = train_batch_size)

	print_test_accuracy()

	# accuracy, recall, precision, and f1-score 
	# compute overall accuracy 
	acc_test = accuracy_score(cls_test, cls_pred)
	print "Overall Accuracy {}: ".format(acc_test)

	# compute overral recall
	recall = recall_score(cls_test, cls_pred, average="macro")  
	print "Overall recall {}: ".format(recall)
		
	# compute overral precision
	precision = precision_score(cls_test, cls_pred, average="macro")
	print "Overall precision {}: ".format(precision)
	
	# compute overral f1_score
	f1 = f1_score(cls_test, cls_pred, average="macro")
	print "Overall f1_score {}: ".format(f1)


	optimize(num_iterations =1000, batch_size = train_batch_size)

	print_test_accuracy()

	# accuracy, recall, precision, and f1-score 
	# compute overall accuracy 
	acc_test = accuracy_score(cls_test, cls_pred)
	print "Overall Accuracy {}: ".format(acc_test)

	# compute overral recall
	recall = recall_score(cls_test, cls_pred, average="macro")  
	print "Overall recall {}: ".format(recall)
		
	# compute overral precision
	precision = precision_score(cls_test, cls_pred, average="macro")
	print "Overall precision {}: ".format(precision)
	
	# compute overral f1_score
	f1 = f1_score(cls_test, cls_pred, average="macro")
	print "Overall f1_score {}: ".format(f1)

	optimize(num_iterations =1000, batch_size = train_batch_size)
	
	print_test_accuracy()

	# accuracy, recall, precision, and f1-score 
	# compute overall accuracy 
	acc_test = accuracy_score(cls_test, cls_pred)
	print "Overall Accuracy {}: ".format(acc_test)

	# compute overral recall
	recall = recall_score(cls_test, cls_pred, average="macro")  
	print "Overall recall {}: ".format(recall)
		
	# compute overral precision
	precision = precision_score(cls_test, cls_pred, average="macro")
	print "Overall precision {}: ".format(precision)
	
	# compute overral f1_score
	f1 = f1_score(cls_test, cls_pred, average="macro")
	print "Overall f1_score {}: ".format(f1)

	optimize(num_iterations =1000, batch_size = train_batch_size)
	
	print_test_accuracy()

	# accuracy, recall, precision, and f1-score 
	# compute overall accuracy 
	acc_test = accuracy_score(cls_test, cls_pred)
	print "Overall Accuracy {}: ".format(acc_test)

	# compute overral recall
	recall = recall_score(cls_test, cls_pred, average="macro")  
	print "Overall recall {}: ".format(recall)
		
	# compute overral precision
	precision = precision_score(cls_test, cls_pred, average="macro")
	print "Overall precision {}: ".format(precision)
	
	# compute overral f1_score
	f1 = f1_score(cls_test, cls_pred, average="macro")
	print "Overall f1_score {}: ".format(f1)

	optimize(num_iterations =1000, batch_size = train_batch_size)
	
	print_test_accuracy()

	# accuracy, recall, precision, and f1-score 
	# compute overall accuracy 
	acc_test = accuracy_score(cls_test, cls_pred)
	print "Overall Accuracy {}: ".format(acc_test)

	# compute overral recall
	recall = recall_score(cls_test, cls_pred, average="macro")  
	print "Overall recall {}: ".format(recall)
		
	# compute overral precision
	precision = precision_score(cls_test, cls_pred, average="macro")
	print "Overall precision {}: ".format(precision)
	
	# compute overral f1_score
	f1 = f1_score(cls_test, cls_pred, average="macro")
	print "Overall f1_score {}: ".format(f1)

	optimize(num_iterations =1000, batch_size = train_batch_size)
	
	print_test_accuracy()

	# accuracy, recall, precision, and f1-score 
	# compute overall accuracy 
	acc_test = accuracy_score(cls_test, cls_pred)
	print "Overall Accuracy {}: ".format(acc_test)

	# compute overral recall
	recall = recall_score(cls_test, cls_pred, average="macro")  
	print "Overall recall {}: ".format(recall)
		
	# compute overral precision
	precision = precision_score(cls_test, cls_pred, average="macro")
	print "Overall precision {}: ".format(precision)
	
	# compute overral f1_score
	f1 = f1_score(cls_test, cls_pred, average="macro")
	print "Overall f1_score {}: ".format(f1)

	optimize(num_iterations =1000, batch_size = train_batch_size)
	
	print_test_accuracy()

	# accuracy, recall, precision, and f1-score 
	# compute overall accuracy 
	acc_test = accuracy_score(cls_test, cls_pred)
	print "Overall Accuracy {}: ".format(acc_test)

	# compute overral recall
	recall = recall_score(cls_test, cls_pred, average="macro")  
	print "Overall recall {}: ".format(recall)
		
	# compute overral precision
	precision = precision_score(cls_test, cls_pred, average="macro")
	print "Overall precision {}: ".format(precision)
	
	# compute overral f1_score
	f1 = f1_score(cls_test, cls_pred, average="macro")
	print "Overall f1_score {}: ".format(f1)

	optimize(num_iterations =1000, batch_size = train_batch_size)
	
	print_test_accuracy()

	# accuracy, recall, precision, and f1-score 
	# compute overall accuracy 
	acc_test = accuracy_score(cls_test, cls_pred)
	print "Overall Accuracy {}: ".format(acc_test)

	# compute overral recall
	recall = recall_score(cls_test, cls_pred, average="macro")  
	print "Overall recall {}: ".format(recall)
		
	# compute overral precision
	precision = precision_score(cls_test, cls_pred, average="macro")
	print "Overall precision {}: ".format(precision)
	
	# compute overral f1_score
	f1 = f1_score(cls_test, cls_pred, average="macro")
	print "Overall f1_score {}: ".format(f1)

	optimize(num_iterations =1000, batch_size = train_batch_size)
	
	print_test_accuracy()

	# accuracy, recall, precision, and f1-score 
	# compute overall accuracy 
	acc_test = accuracy_score(cls_test, cls_pred)
	print "Overall Accuracy {}: ".format(acc_test)

	# compute overral recall
	recall = recall_score(cls_test, cls_pred, average="macro")  
	print "Overall recall {}: ".format(recall)
		
	# compute overral precision
	precision = precision_score(cls_test, cls_pred, average="macro")
	print "Overall precision {}: ".format(precision)
	
	# compute overral f1_score
	f1 = f1_score(cls_test, cls_pred, average="macro")
	print "Overall f1_score {}: ".format(f1)

	optimize(num_iterations =1000, batch_size = train_batch_size)
	
	print_test_accuracy()

	# accuracy, recall, precision, and f1-score 
	# compute overall accuracy 
	acc_test = accuracy_score(cls_test, cls_pred)
	print "Overall Accuracy {}: ".format(acc_test)

	# compute overral recall
	recall = recall_score(cls_test, cls_pred, average="macro")  
	print "Overall recall {}: ".format(recall)
		
	# compute overral precision
	precision = precision_score(cls_test, cls_pred, average="macro")
	print "Overall precision {}: ".format(precision)
	
	# compute overral f1_score
	f1 = f1_score(cls_test, cls_pred, average="macro")
	print "Overall f1_score {}: ".format(f1)

	optimize(num_iterations =1000, batch_size = train_batch_size)
	
	print_test_accuracy()

	# accuracy, recall, precision, and f1-score 
	# compute overall accuracy 
	acc_test = accuracy_score(cls_test, cls_pred)
	print "Overall Accuracy {}: ".format(acc_test)

	# compute overral recall
	recall = recall_score(cls_test, cls_pred, average="macro")  
	print "Overall recall {}: ".format(recall)
		
	# compute overral precision
	precision = precision_score(cls_test, cls_pred, average="macro")
	print "Overall precision {}: ".format(precision)
	
	# compute overral f1_score
	f1 = f1_score(cls_test, cls_pred, average="macro")
	print "Overall f1_score {}: ".format(f1)


	optimize(num_iterations =1000, batch_size = train_batch_size)
	
	print_test_accuracy()

	# accuracy, recall, precision, and f1-score 
	# compute overall accuracy 
	acc_test = accuracy_score(cls_test, cls_pred)
	print "Overall Accuracy {}: ".format(acc_test)

	# compute overral recall
	recall = recall_score(cls_test, cls_pred, average="macro")  
	print "Overall recall {}: ".format(recall)
		
	# compute overral precision
	precision = precision_score(cls_test, cls_pred, average="macro")
	print "Overall precision {}: ".format(precision)
	
	# compute overral f1_score
	f1 = f1_score(cls_test, cls_pred, average="macro")
	print "Overall f1_score {}: ".format(f1)


	optimize(num_iterations =1000, batch_size = train_batch_size)
	
	print_test_accuracy()

	# accuracy, recall, precision, and f1-score 
	# compute overall accuracy 
	acc_test = accuracy_score(cls_test, cls_pred)
	print "Overall Accuracy {}: ".format(acc_test)

	# compute overral recall
	recall = recall_score(cls_test, cls_pred, average="macro")  
	print "Overall recall {}: ".format(recall)
		
	# compute overral precision
	precision = precision_score(cls_test, cls_pred, average="macro")
	print "Overall precision {}: ".format(precision)
	
	# compute overral f1_score
	f1 = f1_score(cls_test, cls_pred, average="macro")
	print "Overall f1_score {}: ".format(f1)

	optimize(num_iterations =1000, batch_size = train_batch_size)
	
	print_test_accuracy()
	# Save the variables to disk.
  	save_path = saver.save(session, model_saved)
  	print("Model saved in file: %s" % save_path)

	#Now, save the graph
	saver.save(session, 'my_test_model',global_step=1000)
	
	session.close()

	# accuracy, recall, precision, and f1-score 
	# compute overall accuracy 
	acc_test = accuracy_score(cls_test, cls_pred)
	print "Overall Accuracy {}: ".format(acc_test)

	# compute overral recall
	recall = recall_score(cls_test, cls_pred, average="macro")  
	print "Overall recall {}: ".format(recall)
		
	# compute overral precision
	precision = precision_score(cls_test, cls_pred, average="macro")
	print "Overall precision {}: ".format(precision)
	
	# compute overral f1_score
	f1 = f1_score(cls_test, cls_pred, average="macro")
	print "Overall f1_score {}: ".format(f1)


	
if __name__ == "__main__":
	main()
