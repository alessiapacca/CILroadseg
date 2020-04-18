

class ModelBase:

    #
    # Resets the model and prepares it for a new training.
    #
    def initialize(self):
        pass

    #
    # Trains the model with the given training data.
    #
    def train(self, Y, X):
        pass

    #
    # Uses the model to classify the given data.
    #
    def classify(self, X):
        pass