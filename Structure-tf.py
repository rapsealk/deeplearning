import tensorflow as tf

def inference(x, keep_prob, n_in, n_hiddens, n_out):
	# Define model.

	def weight_variable(shape):
		initial = tf.truncated_normal(shape, stddev=0.01)
		return tf.Variable(initial)

	def bias_variable(shape):
		initial = tf.zeros(shape)
		return tf.Variable(initial)

	# Input - Hidden, Hidden - Hidden
	for i, n_hidden in enumerate(n_hiddens):
		if i == 0:
			input = x
			input_dim = n_in
		else:
			input = output
			input_dim = n_hiddens[i-1]

		W = weight_variable([input_dim, n_hidden])
		b = bias_variable([n_hidden])

		h = tf.nn.relu(tf.matmul(input, W) + b)
		output = tf.nn.dropout(h, keep_prob=keep_prob)

	# Hidden - Output
	W_out = weight_variable([n_hiddens[-1], n_out])
	b_out = bias_variable([n_out])
	y = tf.nn.softmax(tf.matmul(output, W_out) + b_out)
	return y

def loss(y, t):
	# Define loss function.
	cross_entropy = tf.reduce_mean(-tf.reduce_sum(t * tf.log(y), reduction_indicies=[1]))
	return cross_entropy

def training(loss):
	# Define training algorithm.
	optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
	train_step = optimizer.minimize(loss)
	return train_step

if __name__ == '__main__':
	# 1. Set data
	# 2. Build model

	n_in = 28 * 28
	n_hiddens = [200, 200, 200]
	n_out = 10

	x = tf.placeholder(tf.float32, shape=[None, n_in])
	keep_prob = tf.placeholder(tf.float32)

	y = inference(x, keep_prob, n_in=n_in, n_hiddens=n_hiddens, n_out=n_out)
	loss = loss(y, t)
	train_step = training(loss)
	# 3. Train model
	# 4. Evaluate model