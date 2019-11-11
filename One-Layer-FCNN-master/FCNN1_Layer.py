# Modification And Bug Fixing by:
# Author: Ahmet T
#
#  Sudiro
#	[at] SudiroEEN@gmail.com

import tensorflow as tf
import numpy as np
from mnist import MNIST

 
def randomize(dataset, labels):
	permutation = np.random.permutation(labels.shape[0])
	shuffled_dataset = dataset[permutation, :, :]
	shuffled_labels = labels[permutation]
	return shuffled_dataset, shuffled_labels
 
def one_hot_encode(np_array):
	return (np.arange(10) == np_array[:,None]).astype(np.float32)
 
def reformat_data(dataset, labels, image_width, image_height, image_depth):
	np_dataset_ = np.array([np.array(image_data).reshape(image_width, image_height, image_depth) for image_data in dataset])
	np_labels_ = one_hot_encode(np.array(labels, dtype=np.float32))
	np_dataset, np_labels = randomize(np_dataset_, np_labels_)
	return np_dataset, np_labels
 
def flatten_tf_array(array):
	shape = array.get_shape().as_list()
	return tf.reshape(array, [shape[0], shape[1] * shape[2] * shape[3]])
 
def accuracy(predictions, labels):
	return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1)) / predictions[0].shape[0])

mnist_folder = '../dataMNIST/'
mnist_image_width = 28
mnist_image_height = 28
mnist_image_depth = 1
mnist_num_labels = 10
 
mndata = MNIST(mnist_folder)
mnist_train_dataset_, mnist_train_labels_ = mndata.load_training()
mnist_test_dataset_, mnist_test_labels_ = mndata.load_testing()
 
mnist_train_dataset, mnist_train_labels = reformat_data(mnist_train_dataset_, mnist_train_labels_, mnist_image_width, mnist_image_height, mnist_image_depth)
mnist_test_dataset, mnist_test_labels = reformat_data(mnist_test_dataset_, mnist_test_labels_, mnist_image_width, mnist_image_height, mnist_image_depth)
 
print("There are {} images, each of size {}".format(len(mnist_train_dataset), len(mnist_train_dataset[0])))
print("Meaning each image has the size of 28*28*1 = {}".format(mnist_image_width*mnist_image_height*1))
print("The training set contains the following {} labels: {}".format(len(np.unique(mnist_train_labels_)), np.unique(mnist_train_labels_)))
 
print('Training set shape', mnist_train_dataset.shape, mnist_train_labels.shape)
print('Test set shape', mnist_test_dataset.shape, mnist_test_labels.shape)



			### FCNN 1 Layer
image_width = mnist_image_width
image_height = mnist_image_height
image_depth = mnist_image_depth
num_labels = mnist_num_labels 
 
#the dataset
train_dataset = mnist_train_dataset
train_labels = mnist_train_labels 
test_dataset = mnist_test_dataset
test_labels = mnist_test_labels 
 
#number of iterations and learning rate
num_steps = 10001
display_step = 10
learning_rate = 0.5
batch_size = 60000
 
graph = tf.Graph()
with graph.as_default():
	#1) First we put the input data in a tensorflow friendly form. 
	tf_train_dataset = tf.placeholder(tf.float32, shape=(batch_size, image_width, image_height, image_depth), name='tf_train_dataset')
	tf_train_labels = tf.placeholder(tf.float32, shape=(batch_size, 10), name='tf_train_labels') #shape=train_labels.shape
	tf_test_dataset = tf.constant(test_dataset, tf.float32, name='tf_test_dataset')
	
  
	#2) Then, the weight matrices and bias vectors are initialized
	#as a default, tf.truncated_normal() is used for the weight matrix and tf.zeros() is used for the bias vector.
	weights = tf.Variable(tf.truncated_normal([image_width * image_height * image_depth, num_labels]), tf.float32, name='weights')
	bias = tf.Variable(tf.zeros([num_labels]), tf.float32, name='bias')
  
	#3) define the model:
	#A one layered fccd simply consists of a matrix multiplication

	def model(data, weights, bias):
		print("flatten: ", flatten_tf_array(data).shape)
		print("weights: ", weights.shape)
		return tf.matmul(flatten_tf_array(data), weights) + bias
	
	logits = model(tf_train_dataset, weights, bias)
	
	#4) calculate the loss, which will be used in the optimization of the weights
	loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=tf_train_labels))
 
	#5) Choose an optimizer. Many are available.
	optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)
 
	#6) The predicted values for the images in the train dataset and test dataset are assigned to the variables train_prediction and test_prediction. 
	#It is only necessary if you want to know the accuracy by comparing it with the actual values. 
	train_prediction = tf.nn.softmax(logits)
	test_prediction = tf.nn.softmax(model(tf_test_dataset, weights, bias))
 
 
with tf.Session(graph=graph) as session:
	tf.global_variables_initializer().run()
	print('Initialized')
	for step in range(num_steps):
		dict_logits = {tf_train_dataset: train_dataset}
		LogIts = session.run([logits], feed_dict=dict_logits) #(1, 60000, 10)

		print("LogIts: ", LogIts.shape)
		
		dict_loss = {logits: LogIts[0], tf_train_labels: train_labels}
		Loss = session.run([loss], feed_dict=dict_loss)
		
		dict_optimizer = {tf_train_dataset: train_dataset, tf_train_labels: train_labels, loss: Loss[0]}
		Optimizer = session.run([optimizer], feed_dict=dict_optimizer)

		dict_train_pred = {tf_train_dataset: train_dataset, tf_train_labels: train_labels, logits: LogIts[0]}
		predictions = session.run([train_prediction], feed_dict=dict_train_pred)

		train_accuracy = accuracy(predictions, train_labels[:, :])
		test_accuracy = accuracy(test_prediction.eval(), test_labels)
		# message = '''step {:04d} : loss is {:06.2f}, accuracy on training set {:02.2f} %, accuracy on test set {:02.2f} %'''.format(step, loss, train_accuracy, test_accuracy)
		message = "step {:04d}, loss:".format(step), Loss[0], " train_accuracy: ", train_accuracy, " test_accuracy: ", test_accuracy
		print(message)
