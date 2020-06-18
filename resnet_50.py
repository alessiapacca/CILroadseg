
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.models import Model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, GlobalMaxPooling2D, Dropout,Activation,Flatten
from tensorflow.keras.callbacks import EarlyStopping,ReduceLROnPlateau
from tensorflow.keras import optimizers
from decomposer import *
from util.config import *


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

class ResnetModel(ModelBase):

    def __init__(self):
        self.window_size = window_size
        self.model = None

    def initialize(self):
        input_shape = (self.window_size, self.window_size, 3)

        restnet = ResNet50(weights = "imagenet", include_top = False, input_shape = input_shape)

        output = restnet.layers[-1].output
        output = Flatten()(output)
        restnet = Model(restnet.input, output)

        # for i, layer in enumerate(restnet.layers):
        #     print(i, layer.name)

        for layer in restnet.layers[:75]:
            layer.trainable = False

        for layer in restnet.layers[0:]:
            layer.trainable = True
        
        model = Sequential()
        model.add(restnet)
        model.add(Dense(64, activation='relu'))
        model.add(Dropout(rate = 0.3))
        model.add(Dense(64, activation='relu'))
        model.add(Dropout(rate = 0.3))
        model.add(Dense(1, activation='sigmoid'))

        self.model = model

    def load(self, filename):
        self.model.load_weights(filename)

    def save(self, filename):
        self.model.save_weights(filename)

    def train_online(self, generator):
        # this generator does the bootstrap of a single sample.
        # batch_generator will create batches of these samples

        self.model.compile(optimizer=optimizers.RMSprop(lr=2e-5), loss='binary_crossentropy', metrics=['accuracy'])

        callbacks = [
            ReduceLROnPlateau(monitor='accuracy', factor=0.5, patience=3,
                              verbose=1, mode='auto', min_delta=0.0001, cooldown=0, min_lr=0),
            EarlyStopping(monitor='accuracy', min_delta=0.0001, patience=5, verbose=1, mode='auto')
        ]

        X_val, Y_val = next(val_batch_generator(generator))

        self.model.fit(batch_generator(generator),
                                 steps_per_epoch=steps_per_epoch,
                                 epochs=epochs,
                                 verbose=1,
                                 callbacks=callbacks,
                                 validation_data=(X_val, Y_val))

    def classify(self, X):
        Z = self.model.predict(X)

        return (Z > 0.5).astype(int).ravel()