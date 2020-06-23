
from keras import Sequential
from keras.callbacks import ReduceLROnPlateau, EarlyStopping
from keras.layers import Conv2D, MaxPooling2D, ReLU, Dropout, Flatten, Dense, LeakyReLU, BatchNormalization
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

        print('u')
        layers = [
            # First convolution
            Conv2D(filters=16, kernel_size=5, padding='same', input_shape=input_s),
            LeakyReLU(alpha=0.1),
            BatchNormalization(),
            MaxPooling2D(pool_size=2, padding='same'),
            Dropout(0.1),

            # Second convolution
            Conv2D(filters=32, kernel_size=5, padding='same'),
            LeakyReLU(alpha=0.1),
            BatchNormalization(),
            MaxPooling2D(pool_size=2, padding='same'),
            Dropout(0.1),

            # Third convolution
            Conv2D(filters=64, kernel_size=3, padding='same'),
            LeakyReLU(alpha=0.1),
            BatchNormalization(),
            MaxPooling2D(pool_size=2, padding='same'),
            Dropout(0.2),

            # Fourth convolution
            Conv2D(filters=128, kernel_size=3, padding='same'),
            LeakyReLU(alpha=0.1),
            BatchNormalization(),
            MaxPooling2D(pool_size=2, padding='same'),
            Dropout(0.2),

            # Fifth convolution
            Conv2D(filters=256, kernel_size=3, padding='same'),
            LeakyReLU(alpha=0.1),
            BatchNormalization(),
            MaxPooling2D(pool_size=2, padding='same'),
            Dropout(0.3),

            # Classification layer
            Flatten(),
            Dense(128), #kernel_regularizer=l2(1e-6)
            LeakyReLU(alpha=0.1),
            Dropout(0.4),

            # Output
            Dense(1, activation='sigmoid')
        ]

        self.model = Sequential(layers)

    def load(self, filename):
        self.model.load_weights(filename)

    def save(self, filename):
        self.model.save_weights(filename)

    def train_online(self, generator):
        # this generator does the bootstrap of a single sample.
        # batch_generator will create batches of these samples

        # self.model.summary()

        opt = Adam(0.001)
        # opt = RMSprop(lr=2e-4)

        self.model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])

        callbacks = [
            ReduceLROnPlateau(monitor='val_loss', min_delta=0.0001, patience=5, verbose=1, factor=0.1),
            EarlyStopping(monitor='val_loss', min_delta=0.0001, patience=6, verbose=1)
        ]

        val_batch = None
        if val_batch_size > 0:
            val_batch = next(val_batch_generator(generator))

        history = self.model.fit(batch_generator(generator),
                                 steps_per_epoch=steps_per_epoch,
                                 epochs=epochs,
                                 verbose=1,
                                 callbacks=callbacks,
                                 validation_data=val_batch)

        if val_batch_size > 0:
            plot_history(history)

    def classify(self, X):
        Z = self.model.predict(X)

        return (Z > 0.5).astype(int).ravel()