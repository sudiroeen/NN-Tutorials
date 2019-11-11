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


model = tf.keras.models.load_model("LeNet5_ball_goal.h5")
print("model has been loaded !!!")

# pathImg = "classifier_data/test/1/46.jpg"

# pathImg = "classifier_data/test/0/image_51.jpeg"
# pathImg = "classifier_data/test/0/image_84.jpeg"
# pathImg = "classifier_data/test/0/image_107.jpeg"
# pathImg = "classifier_data/test/0/image_4286.jpeg"
# pathImg = "classifier_data/test/0/image_4402.jpeg"
# pathImg = "classifier_data/test/0/image_5919.jpeg"
pathImg = "/home/udiro/Pictures/bll_crop.png"
temp_img = cv2.imread(pathImg)
img = cv2.resize(temp_img, (28,28))
img  = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

list_img = [img.tolist()]
raw_img = np.array(list_img, dtype=np.uint8)

raw_img = raw_img.astype('float32')

raw_img /= 255.0

new_raw_img = raw_img.reshape(raw_img.shape[0], 28, 28, 1)

predictions = model.predict(new_raw_img)

listPred = predictions.tolist()
print("listPred: ", listPred)
# max_label = max(listPred[0])
# index_label = listPred.index(max_label)
maks = 0.0
for n in range(len(listPred[0])):
	if listPred[0][n] > maks:
		index_label = n
		maks = listPred[0][n]
if index_label:
	print("predictions: goalpost")
else:
	print("predictions: ball")

