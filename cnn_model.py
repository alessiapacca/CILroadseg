
import numpy as np
from util.helpers import *
from util.model_base import ModelBase
import tensorflow as tf
from tensorflow import keras
from keras.applications.resnet50 import ResNet50
from keras.models import Sequential,Model,load_model
from keras.layers import Activation, Dropout, Flatten, Dense, GlobalMaxPooling2D
from keras.applications.imagenet_utils import decode_predictions
from decomposer import *
from util.config import *
from keras.optimizers import Adam
from keras.utils import np_utils
from keras.callbacks import EarlyStopping

class Cnn_Model(ModelBase):
    def __init__(self):
        print("initialize model")
        self.model = None


    def initialize(self):
        #resnet initialization
        model = ResNet50(weights = "imagenet", include_top = False)
        x = self.model.output
        x = GlobalMaxPooling2D()(x)
        x = Dense(1024, activation='relu')(x)
        x = Dense(1024, activation='relu')(x)
        x = Dense(512, activation='relu')(x)
        #last layer specifies the number of outputs. We have a binary output, but it's better to use a single unit in the last dense layer.
        #https://stackoverflow.com/questions/54797065/resnet-for-binary-classification-just-2-values-of-cross-validation-accuracy
        x = Dense(2, activation='sigmoid')(x)
        self.model = Model(inputs=self.model.input, outputs=x)
        self.model.summary()
        for i, layer in enumerate(self.model.layers):
            print(i, layer.name)

        for layer in self.model.layers[:75]:
            layer.trainable = False
        for layer in self.model.layers[75:]:
            layer.trainable = True
            

    def load(self, filename):
        self.model.load_weights(filename)

    def save(self, filename):
        self.model.save_weights(filename)

    def generator_batch(self, generator):
        while 1:
            # create one batch
            Xbatch = np.empty((batch_size, window_size, window_size, 3))
            Ybatch = np.empty((batch_size, 2))
            for i in range(batch_size):
                #print((next(generator)))
                label, Xbatch[i] = next(generator)
                Ybatch[i] = np_utils.to_categorical(label, 2)
            yield (Xbatch, Ybatch)


    def train_online(self, generator):
        #train will call the bootstrap method and the fit method
        #generate batches. this methoed
        #TODO choose an optimizer, and a loss function
        adam = Adam(lr=0.001)  # Adam optimizer with default initial learning rate
        self.model.compile(optimizer=adam, loss='binary_crossentropy', metrics=['accuracy'])

        early_stopping = EarlyStopping(monitor='loss', min_delta=0.0001, patience=11, verbose=1, mode='auto')
        #np.random.seed(3), needed?
        self.model.fit_generator(self.generator_batch(generator),
                                 steps_per_epoch=steps_per_epoch,
                                 epochs=epochs,
                                 verbose=1,
                                 callbacks=[early_stopping])

    #TODO write "classify" method
    def classify(self, X):
        # Run prediction
        Z = self.model.predict(X)
        Z = (Z[:, 0] < Z[:, 1]) * 1

        # Regroup patches into images
        return Z


