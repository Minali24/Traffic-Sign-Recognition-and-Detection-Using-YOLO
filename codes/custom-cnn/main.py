#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 28 20:10:22 2017
@author: ankit
"""
# reference: http://machinelearningmastery.com/object-recognition-convolutional-neural-networks-keras-deep-learning-library/
# Plot ad hoc CIFAR10 instances
import sys
from keras.datasets import cifar10
from matplotlib import pyplot
from scipy.misc import toimage
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.constraints import maxnorm
from keras.optimizers import SGD
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.utils import np_utils
from keras import backend as K
from PIL import Image
from os import listdir
import os
import numpy as np
from sklearn.model_selection import train_test_split

K.set_image_dim_ordering('th')
seed = 10
np.random.seed(seed)
num_classes = 43
#load data and split into train and validation
classes = [ name for name in os.listdir(sys.argv[1]) if os.path.isdir(os.path.join(sys.argv[1], name)) ]
labels = []
class_map = {}
train_images = []

for c in classes:
	one_hot_enc_c = [0]*num_classes
	one_hot_enc_c[int(c)] = 1
	class_map[c] = one_hot_enc_c
	# print(class_map[c], c)

for c in classes:
	path = sys.argv[1] + c
	onlyfiles = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]
	for f in onlyfiles:
		img = np.array(Image.open(path+'/'+f).resize((32, 32))).astype('float32')
		img = np.reshape(img, (3, 32, 32))
		train_images.append(img)
		labels.append(class_map[c])
		# print(path+f)


X_train, X_test, y_train, y_test = train_test_split(train_images, labels, test_size=0.2, random_state=42)

X_train = np.array(X_train)
X_test = np.array(X_test)
y_train = np.array(y_train)
y_test = np.array(y_test)

# print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

# X_train = X_test = np.array([img]).astype('float32')
# print(X_train.shape)
# y_train = y_test = np.array(np.reshape([1, 0, 0, 0], (1, 4)))

# Create the model
model = Sequential()
model.add(Conv2D(32, (3, 3), input_shape=(3, 32, 32), activation='relu', padding='same'))
model.add(Dropout(0.2))
model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
model.add(Dropout(0.2))
model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
model.add(Dropout(0.2))
model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dropout(0.2))
model.add(Dense(1024, activation='relu', kernel_constraint=maxnorm(3)))
model.add(Dropout(0.2))
model.add(Dense(512, activation='relu', kernel_constraint=maxnorm(3)))
model.add(Dropout(0.2))
model.add(Dense(num_classes, activation='softmax'))

#second model
#model.add(Conv2D(32, (3, 3), input_shape=(3, 32, 32), padding='same', activation='relu', kernel_constraint=maxnorm(3)))
#model.add(Dropout(0.2))
#model.add(Conv2D(32, (3, 3), activation='relu', padding='same', kernel_constraint=maxnorm(3)))
#model.add(MaxPooling2D(pool_size=(2, 2)))
#model.add(Flatten())
#model.add(Dense(512, activation='relu', kernel_constraint=maxnorm(3)))
#model.add(Dropout(0.5))
#model.add(Dense(num_classes, activation='softmax'))
# Compile model
epochs = 25
lrate = 0.01
decay = lrate/epochs
sgd = SGD(lr=lrate, momentum=0.9, decay=decay, nesterov=False)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
print(model.summary())
# Fit the model
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=epochs, batch_size=32)
# Final evaluation of the model
scores = model.evaluate(X_test, y_test, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))
