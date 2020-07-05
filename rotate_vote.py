
import numpy as np

from util.model_base import ModelBase

#
# Rotate and Vote. For each patch passed for classification, the classification is done for the four
# possible rotations and then the majority of predictions is returned.
#
class RotAndVote(ModelBase):

    #
    # model - model used for classification
    # method - 'maj' (Majority) or 'avg' (1 iff average >= 0.5)
    #
    def __init__(self, model, method = 'maj'):
        self.model = model
        self.method = method

    def initialize(self):
        self.model.initialize()

    def train(self, Y, X):
        self.model.train(Y, X)

    def train_online(self, generator, val_generator = None):
        self.model.train_online(generator)

    def classify(self, X):
        model = self.model
        num_samples = X.shape[0]

        X_rots = np.empty((4*num_samples, X.shape[1], X.shape[2], X.shape[3]))

        X_rots[0:num_samples] = X
        for i in range(3):
            X_rots[i*num_samples:(i+1)*num_samples] = np.rot90(X, i+1, axes=(1, 2))

        Y_pred = model.classify(X_rots)
        Y_pred = Y_pred.reshape((4, num_samples))

        Y_pred = Y_pred.T
        if self.method == 'avg':
            Y_pred = np.mean(Y_pred, axis=1)
            Y_pred = (Y_pred >= 0.5) * 1
            return Y_pred
        else:
            return np.array([1 - np.bincount(1 - Y_pred[i]).argmax() for i in range(num_samples)])

    def save(self, filename):
        self.model.save(filename)

    def load(self, filename):
        self.model.load(filename)