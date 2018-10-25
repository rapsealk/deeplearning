#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
os.environ["KERAS_BACKEND"] = "tensorflow"
import numpy as np

from keras.models import Sequential
from keras.layers import Dense

import cv2

#from tqdm import tqdm

class Model():

	def __init__(self, n_input=647*643, n_hidden=1024, n_output=2):
		self.n_input = n_input
		self.n_hidden = n_hidden
		self.n_output = n_output

	def train(self):
		data = cv2.imread('./spectrogram/1540492360.png')
		print(data)
	

if __name__ == '__main__':
	print('Emotiongram!')
	model = Model()
	model.train()