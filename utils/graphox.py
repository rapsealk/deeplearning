import matplotlib.pyplot as plt

class Graphox():

    def __init__(self, history):
        self.fig, self.loss_ax = plt.subplots()
        self.acc_ax = self.loss_ax.twinx()
        self.loss_ax.plot(history.history['loss'], 'y', label='train loss')
        self.acc_ax.plot(history.history['acc'], 'b', label='train acc')

        self.loss_ax.set_xlabel('epoch')
        self.loss_ax.set_ylabel('loss')
        self.acc_ax.set_ylabel('accuracy')

        self.loss_ax.legend(loc='upper left')
        self.acc_ax.legend(loc='lower left')

        plt.show()