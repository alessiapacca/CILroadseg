from keras import Sequential
from keras.applications.xception import Xception
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.layers import Dense, Dropout, Flatten
from keras.models import Model
from keras.optimizers import Adam

from decomposer import *
from util.visualize import *

batch_size = 50
val_batch_size = 1000
window_size = 200

steps_per_epoch = 200
epochs = 20

# let's say we want just 1 output
def batch_generator(bootstrap):
    while 1:
        # create one batch
        X_batch = np.empty((batch_size, window_size, window_size, 3))
        Y_batch = np.empty(batch_size)

        for i in range(batch_size):
            Y_batch[i], X_batch[i] = next(bootstrap)

        yield (X_batch, Y_batch)

def val_batch_generator(bootstrap):
    while 1:
        # create one batch
        X_batch = np.empty((val_batch_size, window_size, window_size, 3))
        Y_batch = np.empty(val_batch_size)

        for i in range(val_batch_size):
            Y_batch[i], X_batch[i] = next(bootstrap)

        yield (X_batch, Y_batch)

class XceptionModel(ModelBase):

    def __init__(self):
        self.window_size = window_size
        self.model = None

    def initialize(self):
        input_shape = (window_size, window_size, 3)

        xcept = Xception(weights="imagenet", include_top=False, input_shape=input_shape, classes=1)
        output = xcept.layers[-1].output
        output = Flatten()(output)
        xcept = Model(xcept.input, output)

        model = Sequential()
        model.add(xcept)
        model.add(Dense(64, activation='relu', input_dim=input_shape))
        model.add(Dropout(rate=0.3))
        model.add(Dense(64, activation='relu'))
        model.add(Dropout(rate=0.3))
        model.add(Dense(1, activation='sigmoid'))

        self.model = model
        self.model.summary()

        for layer in xcept.layers[:75]:
            layer.trainable = False

        for layer in xcept.layers[0:]:
            layer.trainable = True

    def load(self, filename):
        self.model.load_weights(filename)

    def save(self, filename):
        self.model.save_weights(filename)

    def train_online(self, generator, val_generator = None):
        # this generator does the bootstrap of a single sample.
        # batch_generator will create batches of these samples

        self.model.compile(optimizer=Adam(0.001), loss='binary_crossentropy', metrics=['accuracy'])

        callbacks = [
            ReduceLROnPlateau(monitor='val_accuracy', min_delta=0.0001, patience=4, verbose=1, factor=0.5),
            EarlyStopping(monitor='val_accuracy', min_delta=0.0001, patience=10, verbose=1)
        ]

        X_val, Y_val = next(val_batch_generator(val_generator))

        np.random.seed(3)  # fix randomness
        history = self.model.fit(batch_generator(generator),
                                 steps_per_epoch=steps_per_epoch,
                                 epochs=epochs,
                                 verbose=1,
                                 callbacks=callbacks,
                                 validation_data=(X_val, Y_val))

        plot_history(history)

    def classify(self, X):
        return self.model.predict(X)