
from util.model_base import ModelBase


class Discretizer(ModelBase):

    def __init__(self, model):
        self.model = model

    def initialize(self):
        self.model.initialize()

    def train(self, Y, X):
        self.model.train(Y, X)

    def train_online(self, generator, val_generator = None):
        self.model.train_online(generator, val_generator)

    def classify(self, X):
        Z = self.model.classify(X)
        return (Z >= 0.5) * 1

    def save(self, filename):
        self.model.save(filename)

    def load(self, filename):
        self.model.load(filename)