from keras.models import Model
from keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Flatten
from tensorflow.keras.layers import Convolution2D, MaxPooling2D
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from decomposer import *
from util.config import *
from util.visualize import * 
from keras.optimizers import Adam
from keras import optimizers
from keras.utils import np_utils
from keras.callbacks import EarlyStopping
from keras import Sequential
from keras.applications.xception import Xception


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

class Xception_Model(ModelBase):

    def __init__(self):
        self.model = None

    def initialize(self):
        input_shape = (200, 200, 3)
        xcept = Xception(weights="imagenet", include_top=False, input_shape=input_shape, classes=1)
        print('create xcept model')
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
        print('created xcept model successfully')
        self.model = model
        self.model.summary()

        for i, layer in enumerate(xcept.layers):
            print(i, layer.name)

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
        adam = Adam(lr=0.001)  # Adam optimizer with default initial learning rate

        self.model.compile(optimizer=adam, loss='binary_crossentropy', metrics=['accuracy'])

        
        callbacks = [
            ReduceLROnPlateau(monitor='val_accuracy', min_delta=0.0001, patience=4,
                              verbose=1, mode='auto', factor=0.5, cooldown=0, min_lr=0),
            EarlyStopping(monitor='val_accuracy', min_delta=0.0001, patience=10,
                          verbose=1, mode='auto')
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
        Z = self.model.predict(X)
        return Z