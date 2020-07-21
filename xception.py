from keras import Sequential
from keras.applications.xception import Xception
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.layers import Dense, Dropout, Flatten
from keras.optimizers import Adam

from decomposer import *
from util.visualize import *


def batch_generator(bootstrap):
    batch_size = 50

    while 1:
        # create one batch
        X_batch = np.empty((batch_size, window_size, window_size, 3))
        Y_batch = np.empty(batch_size)

        for i in range(batch_size):
            Y_batch[i], X_batch[i] = next(bootstrap)

        yield (X_batch, Y_batch)


def val_batch_generator(bootstrap):
    val_batch_size = 1000

    while 1:
        # create one batch
        X_batch = np.empty((val_batch_size, window_size, window_size, 3))
        Y_batch = np.empty(val_batch_size)

        for i in range(val_batch_size):
            Y_batch[i], X_batch[i] = next(bootstrap)

        yield (X_batch, Y_batch)

class XceptionModel(ModelBase):

    def __init__(self):
        self.model = None
        self.window_size = 200

        self.epochs = 20
        self.steps_per_epoch = 200

    def initialize(self):
        input_shape = (self.window_size, self.window_size, 3)

        xcept = Xception(weights='imagenet', include_top=False, input_shape=input_shape)

        for layer in xcept.layers[:20]:
            layer.trainable = False

        for layer in xcept.layers[20:]:
            layer.trainable = True

        layers = [
            xcept,

            Flatten(),
            Dense(64, activation='relu', input_dim=input_shape),
            Dropout(0.15),
            Dense(64, activation='relu'),
            Dropout(0.15),
            Dense(1, activation='sigmoid')
        ]

        self.model = Sequential(layers)

    def load(self, filename):
        self.model.load_weights(filename)

    def save(self, filename):
        self.model.save_weights(filename)

    def train_online(self, generator, val_generator=None):
        # this generator does the bootstrap of a single sample.
        # batch_generator will create batches of these samples

        self.model.summary()
        self.model.compile(optimizer=Adam(lr=0.001), loss='binary_crossentropy', metrics=['accuracy'])

        callbacks = [
            ReduceLROnPlateau(monitor='val_accuracy', min_delta=0.0001, patience=4, verbose=1, factor=0.5),
            EarlyStopping(monitor='val_accuracy', min_delta=0.0001, patience=10, verbose=1)
        ]

        X_val, Y_val = next(val_batch_generator(val_generator))

        np.random.seed(3)
        history = self.model.fit(batch_generator(generator),
                                 steps_per_epoch=self.steps_per_epoch,
                                 epochs=self.epochs,
                                 verbose=1,
                                 callbacks=callbacks,
                                 validation_data=(X_val, Y_val))

        plot_history(history)

    def classify(self, X):
        return self.model.predict(X)