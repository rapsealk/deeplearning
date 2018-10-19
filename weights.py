#!/usr/bin/python3
import tensorflow as tf

a = tf.placeholder(tf.float32)
b = tf.placeholder(tf.float32)

y = tf.multiply(a, b)

# lazy
with tf.Session() as sess:
	result, weights = sess.run(y, feed_dict={ a: 3, b: 3 })
	print('Result:', result)
	print('Weights:', weights)