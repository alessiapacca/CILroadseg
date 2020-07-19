
import numpy as np

from util.model_base import ModelBase

#
# Rotate and Mean decorator. For each classification, the classification is done for the eight
# possible rotations/flip combinations and then the mean of predictions is returned.
#
class RotAndMean(ModelBase):

    #
    # model - model used for classification
    #
    def __init__(self, model):
        self.model = model

    def initialize(self):
        self.model.initialize()

    def train(self, Y, X):
        self.model.train(Y, X)

    def train_online(self, generator, val_generator = None):
        self.model.train_online(generator, val_generator)

    def classify(self, X):
        model = self.model
        num_samples = X.shape[0]

        X_rots = np.empty((8*num_samples, X.shape[1], X.shape[2], X.shape[3]))

        for i in range(4):
            X_rots[i*num_samples:(i+1)*num_samples] = np.rot90(X, i+1, axes=(1, 2))

        for i in range(4):
            X_rots[(i+4)*num_samples:(i+5)*num_samples] = np.fliplr(np.rot90(X, i+1, axes=(1, 2)))

        Y_pred = model.classify(X_rots)
        Y_pred = Y_pred.reshape((48, num_samples))

        Y_pred = np.mean(Y_pred, axis=0)
        Y_pred = (Y_pred >= 0.5) * 1
        return Y_pred

    def save(self, filename):
        self.model.save(filename)

    def load(self, filename):
        self.model.load(filename)