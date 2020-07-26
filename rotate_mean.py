
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
            X_rots[i*num_samples:(i+1)*num_samples] = np.rot90(X, i, axes=(1, 2))

            for j in range(i*num_samples, (i+1)*num_samples): # copy the array but flipped
                X_rots[j + 4*num_samples] = np.fliplr(X_rots[j])

        Y_pred = model.classify(X_rots)
        if Y_pred.ndim > 2:  # self.model returns full masks

            for j in range(4*num_samples, 8*num_samples): # reflip
                Y_pred[j] = np.fliplr(Y_pred[j])

            for i in range(4): # rerotate
                Y_pred[i*num_samples:(i+1)*num_samples] = np.rot90(Y_pred[i*num_samples:(i+1)*num_samples], 4-i, axes=(1, 2))

            Y_pred = Y_pred.reshape((8, num_samples, Y_pred.shape[1], Y_pred.shape[2]))
        else:  # self.model returns patchwise classifications
            Y_pred = Y_pred.reshape((8, num_samples))

        return np.mean(Y_pred, axis=0)

    def save(self, filename):
        self.model.save(filename)

    def load(self, filename):
        self.model.load(filename)