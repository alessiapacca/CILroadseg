
from keras import Sequential, Input, Model
from keras.callbacks import ReduceLROnPlateau, EarlyStopping
from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense, LeakyReLU, BatchNormalization, UpSampling2D, Add
from keras.optimizers import Adam

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


class ConvNetModel(ModelBase):

    def __init__(self):
        self.window_size = window_size
        self.model = None

    def initialize(self):
        input_s = (self.window_size, self.window_size, 3)

        print('r')

        input_layer = Input(shape=input_s)

        cleaner_layers = [
            Conv2D(filters=64, kernel_size=5, padding='same'),
            MaxPooling2D(pool_size=2, padding='same'),
            BatchNormalization(),
            Dropout(0.3),
            LeakyReLU(alpha=0.1),

            UpSampling2D(size=2, interpolation='nearest'),
            Conv2D(filters=3, kernel_size=5, padding='same'),
            Dropout(0.3),
            LeakyReLU(alpha=0.1)
        ]

        cleaner = Sequential(cleaner_layers)(input_layer)

        cleaned_image = Add()([input_layer, cleaner])

        layers = [
            # First convolution
            Conv2D(filters=64, kernel_size=5, padding='same'),
            #BatchNormalization(),
            MaxPooling2D(pool_size=2, padding='same'),
            Dropout(0.3),
            LeakyReLU(alpha=0.1),

            # Second convolution
            Conv2D(filters=96, kernel_size=5, padding='same'),
            #BatchNormalization(),
            MaxPooling2D(pool_size=2, padding='same'),
            Dropout(0.3),
            LeakyReLU(alpha=0.1),

            # Third convolution
            Conv2D(filters=128, kernel_size=3, padding='same'),
            #BatchNormalization(),
            MaxPooling2D(pool_size=2, padding='same'),
            Dropout(0.3),
            LeakyReLU(alpha=0.1),

            # Fourth convolution
            Conv2D(filters=256, kernel_size=3, padding='same'),
            #BatchNormalization(),
            MaxPooling2D(pool_size=2, padding='same'),
            Dropout(0.3),
            LeakyReLU(alpha=0.1),

            # Fifth convolution
            Conv2D(filters=256, kernel_size=3, padding='same'),
            #BatchNormalization(),
            MaxPooling2D(pool_size=2, padding='same'),
            Dropout(0.3),
            LeakyReLU(alpha=0.1),

            # Classification layer
            Flatten(),
            Dense(64),#, kernel_regularizer=l2(1e-6)),
            Dropout(0.4),
            LeakyReLU(alpha=0.1),

            Dense(64),#, kernel_regularizer=l2(1e-6)),
            Dropout(0.4),
            LeakyReLU(alpha=0.1),

            # Output
            Dense(1, activation='sigmoid')
        ]

        output_layer = Sequential(layers)(cleaned_image)

        self.model = Model(inputs=input_layer, outputs=output_layer)

    def load(self, filename):
        self.model.load_weights(filename)

    def save(self, filename):
        self.model.save_weights(filename)

    def train_online(self, generator, val_generator = None):
        # this generator does the bootstrap of a single sample.
        # batch_generator will create batches of these samples

        # self.model.summary()

        opt = Adam(0.001)
        # opt = RMSprop(lr=2e-4)

        self.model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])

        callbacks = [
            ReduceLROnPlateau(monitor='accuracy', min_delta=0.0001, patience=5, verbose=1, factor=0.5),
            EarlyStopping(monitor='accuracy', min_delta=0.0001, patience=11, verbose=1)
        ]

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
        Z = self.model.predict(X)

        return (Z > 0.5).astype(int).ravel()