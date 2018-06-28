#!/usr/bin/python
# -*- coding: utf-8 -*-

RANDOM_SEED = 1234

import numpy as np
np.random.seed(RANDOM_SEED)

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, Conv2D
from keras.layers.pooling import MaxPool2D
from keras.utils import np_utils

from keras.datasets import cifar10

from utils.graphox import Graphox

(X_train, y_train), (X_test, y_test) = cifar10.load_data()

X_train = X_train.reshape(X_train.shape[0], 32, 32, 3)
X_test = X_test.reshape(X_test.shape[0], 32, 32, 3)

X_train = X_train.astype('float32') / 255
X_test = X_test.astype('float32') / 255

NUM_LABELS = 10

Y_train = np_utils.to_categorical(y_train, NUM_LABELS)
Y_test = np_utils.to_categorical(y_test, NUM_LABELS)


# model
model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
model.add(MaxPool2D(pool_size=(2, 2)))
#model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
#model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(NUM_LABELS, activation='softmax'))
print(model.output_shape)

# compile
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# fit
history = model.fit(X_train, Y_train, batch_size=32, epochs=20, verbose=1)

# evaluation
score = model.evaluate(X_test, Y_test, verbose=0)
print(model.metrics_names)
print(score)

Graphox(history)