
import numpy as np

from util.model_base import ModelBase

#
# Rotate and Vote. For each patch passed for classification, the classification is done for the four
# possible rotations and then the majority of predictions is returned.
#
class RotAndVote(ModelBase):

    #
    # model - model used for classification
    #
    def __init__(self, model):
        self.model = model

    def initialize(self):
        self.model.initialize()

    def train(self, Y, X):
        self.model.train(Y, X)

    def train_online(self, generator):
        self.model.train_online(generator)

    def classify(self, X):
        model = self.model
        num_samples = X.shape[0]

        Y_pred = np.empty((4, num_samples), dtype=int)

        for i in range(4):
            Y_pred[i] = model.classify(np.rot90(X, i, axes=(1, 2)))

        Y_pred = Y_pred.T
        return np.array([np.bincount(Y_pred[i]).argmax() for i in range(num_samples)])

    def save(self, filename):
        self.model.save(filename)

    def load(self, filename):
        self.model.load(filename)