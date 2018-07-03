from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.optimizers import SGD
from keras.layers.advanced_activations import ParametricReLU

n_in = 28 * 28
n_hidden = 1024
n_out = 10

if __name__ == '__main__':

    model = Sequential()
    model.add(Dense(n_hidden, input_dim=n_in))
    model.add(ParametricReLU())

    model.add(Dense(n_hidden))
    model.add(ParametricReLU())

    model.add(Dense(n_hidden))
    model.add(ParametricReLU())

    model.add(Dense(n_hidden))
    model.add(ParametricReLU())

    model.add(Dense(n_out))
    model.add(Activation('softmax'))

    model.compile(loss='categorical_crossentropy', optimizer=SGD(lr=0.01), metrics=['accuracy'])    # precision, recall