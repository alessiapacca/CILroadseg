
import numpy as np

from util.model_base import ModelBase

#
# Rotate and Mean. For each classification, the classification is done for the eight
# possible rotations/flip combinations and then the mean of predictions is returned.
#
class RotAndMean(ModelBase):

    #
    # model - model used for classification
    # include_flips - whether also flips should be included
    #
    def __init__(self, model, include_flips=True):
        self.model = model
        self.include_flips = include_flips

    def initialize(self):
        self.model.initialize()

    def train(self, Y, X):
        self.model.train(Y, X)

    def train_online(self, generator, val_generator = None):
        self.model.train_online(generator, val_generator)

    def classify(self, X):
        model = self.model
        num_samples = X.shape[0]
        preds_per_sample = 8 if self.include_flips else 4

        X_rots = np.empty((preds_per_sample*num_samples, X.shape[1], X.shape[2], X.shape[3]))

        for i in range(4):
            X_rots[i*num_samples:(i+1)*num_samples] = np.rot90(X, i, axes=(1, 2))

        if self.include_flips: # copy flipped images on the second part of array
            for i in range(4*num_samples):
                X_rots[4*num_samples + i] = np.fliplr(X_rots[i])

        Y_pred = model.classify(X_rots)

        if Y_pred.ndim == 1: # patchwise predictions
            Y_pred = Y_pred.reshape((preds_per_sample, num_samples))
        else: # imagewise prediction
            Y_pred = Y_pred.reshape((preds_per_sample, num_samples, X.shape[1], X.shape[2], X.shape[3]))

        return np.mean(Y_pred, axis=0)

    def save(self, filename):
        self.model.save(filename)

    def load(self, filename):
        self.model.load(filename)