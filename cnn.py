
from keras import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Flatten
from decomposer import *
from util.config import *
from keras.optimizers import Adam
from keras.utils import np_utils
from keras.callbacks import EarlyStopping


def batch_generator(bootstrap):
    while 1:
        # create one batch
        X_batch = np.empty((batch_size, window_size, window_size, 3))
        Y_batch = np.empty((batch_size))

        for i in range(batch_size):
            Y_batch[i], X_batch[i] = next(bootstrap)

        yield (X_batch, Y_batch)


class CNNModel(ModelBase):

    def __init__(self, patch_size = 16, window_size = 72):
        self.model = None

        self.patch_size = patch_size
        self.window_size = window_size

    def initialize(self):
        input_shape = (self.window_size, self.window_size, 3)

        self.cnn = Sequential()
        self.cnn.add(Conv2D(filters=32, kernel_size=(5, 5), activation='relu', input_shape=input_shape))

        self.cnn.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
        self.cnn.add(Dropout(0.25))
        self.cnn.add(Conv2D(filters=64, kernel_size=(5, 5), activation='relu'))
        self.cnn.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
        self.cnn.add(Dropout(0.25))
        self.cnn.add(Flatten())

        self.cnn.add(Dense(128, activation='relu'))
        self.cnn.add(Dropout(0.5))
        self.cnn.add(Dense(1, activation='sigmoid'))

        opt = Adam(lr=0.001)
        self.cnn.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])

    def load(self, filename):
        self.model.load_weights(filename)

    def save(self, filename):
        self.model.save_weights(filename)

    def train_online(self, generator):
        # this generator does the bootstrap of a single sample.
        # batch_generator will create batches of these samples

        callbacks = [
            EarlyStopping(monitor='loss', patience=2, verbose=1, mode='auto')
        ]

        np.random.seed(3) # fix randomness
        self.model.fit_generator(batch_generator(generator),
                                 steps_per_epoch=steps_per_epoch,
                                 epochs=epochs,
                                 verbose=1,
                                 callbacks=callbacks)

    def classify(self, X):
        Z = self.model.predict(X)
        Z = (Z > 0.5) * 1

        return Z