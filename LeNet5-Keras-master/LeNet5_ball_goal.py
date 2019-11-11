#
#
# Author:
#	Sudiro
#		[at] SudiroEEN@gmail.com
#

import tensorflow as tf
from keras.models import Sequential
from keras.utils import np_utils
from keras import layers

import numpy as np
import os
import cv2

in_img_train = []
in_label_train = []

in_img_test = []
in_label_test = []

path_train = 'classifier_data/train/'
path_test = 'classifier_data/test/'

print("Train Data")
for i in range(2):
	full_path_train = path_train + str(i) + '/'
	files = [f for f in os.listdir(full_path_train) if ((f.split('.')[-1] == 'png') or (f.split('.')[-1] == 'jpg') or (f.split('.')[-1] == 'jpeg'))]
	for f_ in files:
		temp_img = cv2.imread(full_path_train + f_)
		try:
			temp_img = cv2.resize(temp_img, (28,28))
			temp_img = cv2.cvtColor(temp_img, cv2.COLOR_BGR2GRAY)
			in_img_train.append(temp_img.tolist())
			in_label_train.append(i)
		except:
			pass
print('\n')
print("Test Data")
for i in range(2):
	full_path_test = path_test + str(i) + '/'
	files = [f for f in os.listdir(full_path_test) if ((f.split('.')[-1] == 'png') or (f.split('.')[-1] == 'jpg') or (f.split('.')[-1] == 'jpeg'))]
	for f_ in files:
		temp_img = cv2.imread(full_path_test + f_)
		try:
			temp_img = cv2.resize(temp_img, (28,28))
			temp_img = cv2.cvtColor(temp_img, cv2.COLOR_BGR2GRAY)
			in_img_test.append(temp_img.tolist())
			in_label_test.append(i)
		except:
			pass
print('\n')

print('get img_train, img_test, label_train, label_test')
img_train = np.array(in_img_train, dtype=np.uint8)
label_train = np.array(in_label_train, dtype=np.uint8)

img_test = np.array(in_img_test, dtype=np.uint8)
label_test = np.array(in_label_test, dtype=np.uint8)

print('convert to float ...')
img_train = img_train.astype('float32')
img_test = img_test.astype('float32')

print('Normalize ....')
img_train /= 255.0
img_test /= 255.0

print('one hot encoding ...')
label_train = np_utils.to_categorical(label_train, 2)
label_test = np_utils.to_categorical(label_test, 2)

print('reshaping img ...')
img_train = img_train.reshape(img_train.shape[0], 28, 28, 1)
img_test = img_test.reshape(img_test.shape[0], 28, 28, 1)

print(' model Sequential ...')
model = Sequential()

# Layer 1
# Terdiri dari Convolutional Layer dengan 
# 6 jenis filter dengan ukuran 5x5
# dengan stride (1,1)
# activation func: tanh
# input_shape: 28x28x1
# padding: SAME -> with zero padding
print('1st Layer')
model.add(layers.Conv2D(6, kernel_size=(5,5), strides=(1,1),activation='tanh',input_shape=(28,28,1),padding='SAME'))


# Layer 2
# Terdiri dari Average Pooling Layer
# dengan 6 jenis filter
# kernel filter = 2x2
# strides = (2,2)
# padding: 'VALID' -> tanpa padding
# input_shape: 14x14
print('2nd Layer')
model.add(layers.AveragePooling2D(pool_size=(2,2),strides=(1,1),padding='VALID'))


# Layer 3
# Terdiri dari Convolutional Layer
# dengan 16 filter dengan ukuran kernel 5x5
# strides: (1,1)
# activation function: tanh
# padding: 'VALID' -> tanpa padding
# input_shape: 10x10
print('3rd Layer')
model.add(layers.Conv2D(16, kernel_size=(5,5), strides=(1,1), activation='tanh', padding='VALID'))


# Layer 4
# Terdiri dari Average Pooling
# dengan 16 jenis filter 
# ukuran kernel 2x2
# strides: (2,2)
# activation function tanh
# input_shape: 5x5
print('4th Layer')
model.add(layers.AveragePooling2D(pool_size=(2,2), strides=(2,2), padding='VALID'))

# Layer 5
# Terdiri dari Convolutional Layer
# dengan 120 jenis filter
# kernel size: 1x1
# strides: (1,1)
# activation function: tanh
# input_shape: 5x5
print('5th Layer')
model.add(layers.Conv2D(120, kernel_size=(5,5), strides=(1,1), padding='VALID'))
model.add(layers.Flatten())

# Layer 6
# Terdiri dari Fully Connected Layer
# size: 84
# activation function: tanh
print('6th Layer')
model.add(layers.Dense(84, activation='tanh'))


# Layer Output
# Terdiri dari Fully Connected Layer
# Size: 10
# activation function: softmax
print('Output Layer')
model.add(layers.Dense(2, activation='softmax'))

print('compile model')
model.compile(loss=tf.keras.losses.categorical_crossentropy, optimizer='SGD', metrics=["accuracy"])

print('\n\n Training ....\n')

hist = model.fit(x=img_train, y=label_train, epochs=15, batch_size=128, validation_data=(img_test, label_test), verbose=1)

test_score = model.evaluate(img_test, label_test)

print("Test loss {:.7f}, accuracy {:.3f}%".format(test_score[0], test_score[1]*100))

print('\n\n Saving model')
model.save("LeNet5_ball_goal.h5")
print('model has been saved !!!')

