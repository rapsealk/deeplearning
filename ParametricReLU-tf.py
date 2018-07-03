import tensorflow as tf

def prelu(x, alpha):
	return tf.maximum(tf.zeros(tf.shape(x)), x) + alpha * tf.minimum(tf.zeros(tf.shape(x)), x)

n_in = 28 * 28
n_hidden = 1024
n_out = 10

if __name__ == '__main__':

	x = tf.zeros([n_in])

	# Input - Hidden
	W0 = tf.Variable(tf.truncated_normal([n_in, n_hidden], stddev=0.01))
	b0 = tf.Variable(tf.zeros([n_hidden]))
	alpha0 = tf.Variable(tf.zeros([n_hidden]))
	h0 = prelu(tf.matmul(x, W0) + b0, alpha0)

	# Hidden - Hidden
	W1 = tf.Variable(tf.truncated_normal([n_hidden, n_hidden], stddev=0.01))
	b1 = tf.Variable(tf.zeros([n_hidden]))
	alpha1 = tf.Variable(tf.zeros([n_hidden]))
	h1 = prelu(tf.matmul(h0, W1) + b1, alpha1)

	W2 = tf.Variable(tf.truncated_normal([n_hidden, n_hidden], stddev=0.01))
	b2 = tf.Variable(tf.zeros([n_hidden]))
	alpha2 = tf.Variable(tf.zeros([n_hidden]))
	h2 = prelu(tf.matmul(h1, W2) + b2, alpha2)

	W3 = tf.Variable(tf.truncated_normal([n_hidden, n_hidden], stddev=0.01))
	b3 = tf.Variable(tf.zeros([n_hidden]))
	alpha3 = tf.Variable(tf.zeros([n_hidden]))
	h3 = prelu(tf.matmul(h2, W3) + b3, alpha3)

	# Hidden - Output
	W4 = tf.Variable(tf.truncated_normal([n_hidden, n_out], stddev=0.01))
	b4 = tf.Variable(tf.zeros([n_out]))
	y = tf.nn.softmax(tf.matmul(h3, W4) + b4)