
from keras import Sequential
from keras.callbacks import ReduceLROnPlateau, EarlyStopping
from keras.layers import Conv2D, MaxPooling2D, ReLU, Dropout, Flatten, Dense
from keras.optimizers import RMSprop, Adam
from keras.regularizers import l2

from decomposer import *
from util.config import *
from util.visualize import plot_history


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

class ConvNetModel(ModelBase):

    def __init__(self):
        self.window_size = window_size
        self.model = None

    def initialize(self):
        input_s = (self.window_size, self.window_size, 3)

        print('d')
        layers = [
            # First convolution
            Conv2D(filters=48, kernel_size=5, padding='same', input_shape=input_s),
            MaxPooling2D(pool_size=2, padding='same'),
            Dropout(0.3),
            ReLU(),

            # Second convolution
            Conv2D(filters=48, kernel_size=3, padding='same'),
            MaxPooling2D(pool_size=2, padding='same'),
            Dropout(0.3),
            ReLU(),

            # Third convolution
            Conv2D(filters=128, kernel_size=3, padding='same'),
            MaxPooling2D(pool_size=2, padding='same'),
            Dropout(0.3),
            ReLU(),

            # Fourth convolution
            Conv2D(filters=256, kernel_size=3, padding='same'),
            MaxPooling2D(pool_size=2, padding='same'),
            Dropout(0.3),
            ReLU(),

            # Fifth convolution
            Conv2D(filters=256, kernel_size=3, padding='same'),
            MaxPooling2D(pool_size=2, padding='same'),
            Dropout(0.3),
            ReLU(),

            # Classification layers
            Flatten(),
            Dense(64), #, kernel_regularizer=l2(1e-6)
            Dropout(0.5),
            ReLU(),

            Dense(64),
            Dropout(0.5),
            ReLU(),

            # Output
            Dense(1, activation='sigmoid') #, activity_regularizer=l2(1e-6)
        ]

        self.model = Sequential(layers)

    def load(self, filename):
        self.model.load_weights(filename)

    def save(self, filename):
        self.model.save_weights(filename)

    def train_online(self, generator):
        # this generator does the bootstrap of a single sample.
        # batch_generator will create batches of these samples

        self.model.compile(optimizer=Adam(0.001), loss='binary_crossentropy', metrics=['accuracy'])

        callbacks = [
            ReduceLROnPlateau(monitor='val_accuracy', min_delta=0.0001, patience=6,
                              verbose=1, mode='auto', factor=0.5, cooldown=0, min_lr=0),
            #EarlyStopping(monitor='val_accuracy', min_delta=0.0001, patience=5,
            #              verbose=1, mode='auto')
        ]

        X_val, Y_val = next(val_batch_generator(generator))

        history = self.model.fit(batch_generator(generator),
                                 steps_per_epoch=steps_per_epoch,
                                 epochs=epochs,
                                 verbose=1,
                                 callbacks=callbacks,
                                 validation_data=(X_val, Y_val))

        plot_history(history)

    def classify(self, X):
        Z = self.model.predict(X)

        return (Z > 0.5).astype(int).ravel()