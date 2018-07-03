import tensorflow as tf
from keras.layers.advanced_activations import LeakyReLU

def lrelu(x, alpha=0.01):
    return tf.maximum(alpha * x, x)