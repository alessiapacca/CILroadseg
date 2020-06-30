
import numpy as np

from util.model_base import ModelBase

class ZeroClassifier(ModelBase):

    def __init__(self):
        pass

    def initialize(self):
        pass

    def train(self, Y, X):
        pass

    def train_online(self, generator, val_generator = None):
        pass

    def classify(self, X):
        return np.zeros((X.shape[0], X.shape[1], X.shape[2]))