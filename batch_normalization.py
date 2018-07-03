import tensorflow as tf
#from keras.layers.normalization import BatchNormalization

def batch_normalization(shape, x):
	# tf.nn.batch_normalization()
	epsilon = 1e-8
	beta = tf.Variable(tf.zeros(shape))
	gamma = tf.Variable(tf.ones(shape))
	mean, var = tf.nn.moments(x, [0])
	return gamma * (x - mean) / tf.sqrt(var + epsilon) + beta