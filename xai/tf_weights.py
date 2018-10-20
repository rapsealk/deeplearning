#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
from random import random

n_input = 128
n_hidden = 1024
n_output = 1

X = tf.placeholder(tf.float32, [None, n_input])
Y = tf.placeholder(tf.float32, [None, n_output])

W = tf.Variable(tf.random_normal([n_input, n_hidden], stddev=0.01))
b = tf.Variable(tf.zeros([n_hidden]))

W_out = tf.Variable(tf.random_normal([n_hidden, n_output], stddev=0.01))
b_out = tf.Variable(tf.zeros([n_output]))

task = tf.matmul(X, W) + b
task = tf.nn.relu(task)

task = tf.matmul(task, W_out) + b_out
task = tf.nn.sigmoid(task)
#task = tf.nn.softmax(task)
#task = tf.nn.dropout(task, keep_prob=0.5)

# cross entropy
loss = -tf.reduce_sum(task * tf.log(Y), reduction_indices=[1])
loss = tf.reduce_mean(loss)

# optimizer
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
optimizer = optimizer.minimize(loss)


if __name__ == '__main__':
    x_size = 10
    x = np.array([[random() for i in range(n_input)] for j in range(x_size)])
    y = np.array([[random()] for j in range(x_size)])

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        print('before W:', sess.run(W))
        
        for i in range(x_size):
            sess.run(optimizer, feed_dict={
                X: x, Y: y
            })

        print('after W:', sess.run(W))