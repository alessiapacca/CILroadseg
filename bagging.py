
import numpy as np

from util.model_base import ModelBase

#
# Voting classifier. It takes a number of models and does a classification based on the majority.
#
class ClassVoting(ModelBase):

    #
    # num_voters - number of voters
    # vgen - generator that yields models. The models should just be constructed.
    #        Make sure that the generator yields enough objects
    #
    def __init__(self, num_voters, vgen):
        self.num_voters = num_voters
        self.voters = []

        for i in range(num_voters):
            self.voters.append(vgen())

    def initialize(self):
        for v in self.voters:
            v.initialize()

    def train(self, Y, X):
        for i in range(len(self.voters)):
            print('Training voter #' + str(i+1) + '...')
            self.voters[i].train(Y, X)

    def train_online(self, generator, val_generator = None):
        for i in range(len(self.voters)):
            print('Training voter #' + str(i+1) + '...')
            self.voters[i].train_online(generator)

    def classify(self, X):
        num_voters = self.num_voters
        num_samples = X.shape[0]

        Y_pred = np.empty((num_voters, num_samples), dtype=int)

        for i in range(num_voters):
            Y_pred[i] = self.voters[i].classify(X)

        Y_pred = Y_pred.T
        return np.array([np.bincount(Y_pred[i]).argmax() for i in range(num_samples)])

    def save(self, filename): # TODO directory
        pass

    def load(self, filename): # TODO directory
        pass