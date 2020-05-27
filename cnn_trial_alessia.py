from googlenet import create_googlenet
from keras.models import Model
from keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Flatten
from tensorflow.keras.layers import Convolution2D, MaxPooling2D
from tensorflow.keras.callbacks import EarlyStopping,ReduceLROnPlateau
from decomposer import *
from util.config import *
from keras.optimizers import Adam
from keras.utils import np_utils
from keras.callbacks import EarlyStopping
from keras import Sequential
from keras.applications.vgg16 import VGG16

# let's say we want just 1 output
def batch_generator(bootstrap):
    while 1:
        # create one batch
        X_batch = np.empty((batch_size, window_size, window_size, 3))
        Y_batch = np.empty(batch_size)

        for i in range(batch_size):
            Y_batch[i], X_batch[i] = next(bootstrap)

        yield (X_batch, Y_batch)


class cnn_trial(ModelBase):

    def __init__(self):
        self.model = None

    def initialize(self):
        input_shape=(72,72,3)
        vgg = VGG16(weights="imagenet",include_top = False, input_shape = input_shape)
        print('create vgg model')
        output = vgg.layers[-1].output
        output = Flatten()(output)
        vgg = Model(vgg.input, output)

    
        model = Sequential()
        model.add(vgg)
        model.add(Dense(64, activation='relu', input_dim=input_shape))
        model.add(Dropout(rate = 0.3))
        model.add(Dense(64, activation='relu'))
        model.add(Dropout(rate = 0.3))
        model.add(Dense(1, activation='sigmoid'))
        print('created vgg model successfully')
        self.model = model
        self.model.summary()

                                                 
        for i, layer in enumerate(vgg.layers):
            print(i, layer.name)

        for layer in vgg.layers[:75]:
            layer.trainable = False

        for layer in vgg.layers[0:]:
            layer.trainable = True
        
        
    

    def load(self, filename):
        self.model.load_weights(filename)

    def save(self, filename):
        self.model.save_weights(filename)

    def train_online(self, generator):
        # this generator does the bootstrap of a single sample.
        # batch_generator will create batches of these samples

        # TODO choose an optimizer, and a loss function
        adam = Adam(lr=0.001)  # Adam optimizer with default initial learning rate

        self.model.compile(optimizer=adam, loss='binary_crossentropy', metrics=['accuracy'])

        lr_callback = ReduceLROnPlateau(monitor='accuracy', factor=0.5, patience=5,
                                        verbose=1, mode='auto', epsilon=0.0001, cooldown=0, min_lr=0)

        # Stops the training process upon convergence
        stop_callback = EarlyStopping(monitor='accuracy', min_delta=0.0001, patience=11, verbose=1, mode='auto')
        callbacks = [
            EarlyStopping(monitor='loss', patience=2, verbose=1, mode='auto'),
            ReduceLROnPlateau(monitor='loss', factor=0.5, patience=5,
                                    verbose=1, mode='auto', epsilon=0.0001, cooldown=0, min_lr=0)
            
        ]

        np.random.seed(3)  # fix randomness
        self.model.fit_generator(batch_generator(generator),
                                 steps_per_epoch=steps_per_epoch,
                                 epochs=epochs,
                                 verbose=1,
                                 callbacks=[lr_callback, stop_callback])

    def classify(self, X):
        Z = self.model.predict(X)
        Z = ((Z) > 0.5).astype(int)

        return Z