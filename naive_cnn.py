
import numpy as np

from keras import Sequential
from keras.layers import Conv2D, MaxPooling2D, ReLU, Dense, Flatten
from keras.optimizers import Adam

from util.model_base import ModelBase

def patchify(X):
    patches = []

    width = X.shape[0]
    height = X.shape[1]

    for y in range(0, height, 16):
        for x in range(0, width, 16):
            patches.append(X[x:x+16, y:y+16, :])

    return patches


def patchify_gt(X):
    patches = []

    width = X.shape[0]
    height = X.shape[1]

    for y in range(0, height, 16):
        for x in range(0, width, 16):
            patches.append((np.mean(X[x:x+16, y:y+16]) >= 0.25) * 1)

    return patches


def decompose(Y, X):
    X_patches = []
    Y_patches = []

    for i in range(X.shape[0]):
        X_patches += patchify(X[i])
        Y_patches += patchify_gt(Y[i])

    X_patches = np.array(X_patches)
    Y_patches = np.array(Y_patches)

    return Y_patches, X_patches


class NaiveConvModel(ModelBase):

    def __init__(self):
        self.model = None

    def initialize(self):

        layers = [
            Conv2D(filters=32, kernel_size=5, input_shape=(16, 16, 3)),
            ReLU(),
            MaxPooling2D(pool_size=2),

            Conv2D(filters=64, kernel_size=5),
            ReLU(),
            MaxPooling2D(pool_size=2),

            Flatten(),
            Dense(64),
            ReLU(),

            Dense(1, activation='sigmoid')
        ]

        self.model = Sequential(layers)

    def load(self, filename):
        self.model.load_weights(filename)

    def save(self, filename):
        self.model.save_weights(filename)

    def train(self, Y, X):
        Y_f, X_f = decompose(Y, X)

        self.model.summary()
        self.model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])

        self.model.fit(X_f, Y_f, batch_size=150, epochs=100)

    def classify(self, X):
        X_patches = []

        for i in range(X.shape[0]):
            X_patches += patchify(X[i])

        Z = self.model.predict(np.array(X_patches))
        return (Z >= 0.5) * 1