

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
        raise NotImplementedError('This model does not support offline training. Use train_online() instead')

    #
    # Trains the model with a generator that yields the data.
    # The caller should also provide an independent generator that can be used for validation.
    #
    def train_online(self, generator, val_generator = None):
        raise NotImplementedError('This model does not support online training. Use train() instead')

    #
    # Uses the model to classify the given data.
    #
    def classify(self, X):
        pass

    #
    # Saves the model to the given filename
    #
    def save(self, filename):
        pass

    #
    # Loads the model from the given filename
    #
    def load(self, filename):
        pass