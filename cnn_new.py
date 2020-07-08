
import tensorflow as tf

from keras import Sequential, Input, Model
from keras.callbacks import ReduceLROnPlateau, EarlyStopping
from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense, LeakyReLU, BatchNormalization, UpSampling2D, \
    Add, Concatenate, ReLU, Conv2DTranspose, concatenate
from keras.optimizers import Adam
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

def conv_block(filters, dropout=0.5):
    return Sequential([
        Conv2D(filters=filters, kernel_size=3, padding='same', kernel_initializer='he_normal'),
        LeakyReLU(alpha=0.1),
        Dropout(dropout)
    ])

class ConvNetModel(ModelBase):

    def __init__(self):
        self.window_size = window_size
        self.model = None

    def initialize(self):
        input_s = (self.window_size, self.window_size, 3)

        print('a')

        input_layer = Input(shape=input_s)

        classifier = Sequential([
            Conv2D(filters=16, kernel_size=3, padding='same'),
            #BatchNormalization(),
            ReLU(),
            Dropout(0.5),
            MaxPooling2D(pool_size=2, padding='same'),

            Conv2D(filters=32, kernel_size=3, padding='same'),
            #BatchNormalization(),
            ReLU(),
            Dropout(0.5),
            MaxPooling2D(pool_size=2, padding='same'),

            Conv2D(filters=64, kernel_size=3, padding='same'),
            #BatchNormalization(),
            ReLU(),
            Dropout(0.5),
            MaxPooling2D(pool_size=2, padding='same'),

            Conv2D(filters=128, kernel_size=3, padding='same'),
            BatchNormalization(),
            ReLU(),
            Dropout(0.5),
            MaxPooling2D(pool_size=2, padding='same'),

            # Classification layer
            Flatten(),
            Dense(128, kernel_regularizer=l2(1e-6)),
            ReLU(),
            Dropout(0.5),

            # Output
            Dense(1, kernel_regularizer=l2(1e-6), activation='sigmoid')
        ]) (input_layer)

        self.model = Model(inputs=input_layer, outputs=classifier)

    def load(self, filename):
        self.model.load_weights(filename)

    def save(self, filename):
        self.model.save_weights(filename)

    def train_online(self, generator, val_generator = None):
        # this generator does the bootstrap of a single sample.
        # batch_generator will create batches of these samples

        self.model.summary()

        opt = Adam(0.001)
        # opt = RMSprop(lr=2e-4)

        self.model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])

        callbacks = [
            ReduceLROnPlateau(monitor='accuracy', min_delta=0.0001, patience=5, verbose=1, factor=0.5),
            #EarlyStopping(monitor='accuracy', min_delta=0.0001, patience=11, verbose=1)
        ]

        tf.random.set_seed(3)

        history = self.model.fit_generator(
                        batch_generator(generator),
                        steps_per_epoch=steps_per_epoch,
                        epochs=epochs,
                        verbose=2,
                        callbacks=callbacks,
                        validation_data=batch_generator(val_generator),
                        validation_steps=100
        )

        if val_batch_size > 0:
            plot_history(history)

    def classify(self, X):
        return self.model.predict(X).ravel()