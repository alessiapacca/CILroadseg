# -*- coding: utf-8 -*-

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import Adam
from keras.utils import np_utils
from keras import backend as K
from keras.regularizers import l2
from keras.layers import LeakyReLU
from keras.callbacks import ReduceLROnPlateau, EarlyStopping
from helpers import *

class NaiveCnnModel:
    
    def __init__(self):
        """ Construct a CNN classifier. """
        
        self.patch_size = 16
  
        
    def initialize(self):
        
        print('Initialize cnn')
        patch_size = self.patch_size
        pool_size = (2, 2)
        input_shape = (patch_size, patch_size, 3)
        self.cnn = Sequential()
        self.cnn.add(Conv2D(filters=32, kernel_size=(5,5) ,activation='relu',input_shape=input_shape))

        self.cnn.add(MaxPooling2D(pool_size=pool_size, padding='same'))
        self.cnn.add(Dropout(0.25))
        self.cnn.add(Conv2D(filters=64, kernel_size=(5,5) ,activation='relu'))
        self.cnn.add(MaxPooling2D(pool_size=pool_size, padding='same'))
        self.cnn.add(Dropout(0.25))
        self.cnn.add(Flatten())

        self.cnn.add(Dense(128, activation='relu'))
        self.cnn.add(Dropout(0.5))
        self.cnn.add(Dense(1, activation='sigmoid'))

        opt = Adam(lr=0.001) # Adam optimizer with default initial learning rate
        self.cnn.compile(optimizer=opt, loss='binary_crossentropy',metrics=['accuracy'])   
        
    def train(self, Y, X):
        """
        Train this model.
        """
        
        foreground_threshold = 0.25 # percentage of pixels > 1 required to assign a foreground label to a patch

        def value_to_class(v):
            df = np.sum(v)
            if df > foreground_threshold:
                return 1
            else:
                return 0
        
        # Extract patches from input images
        patch_size = self.patch_size
        img_patches = [img_crop(X[i], patch_size, patch_size, patch_size, 0) for i in range(X.shape[0])]
        gt_patches = [img_crop_gt(Y[i], patch_size, patch_size, patch_size) for i in range(X.shape[0])]

        # Linearize list of patches
        img_patches = np.asarray([img_patches[i][j] for i in range(len(img_patches)) for j in range(len(img_patches[i]))])
        gt_patches =  np.asarray([gt_patches[i][j] for i in range(len(gt_patches)) for j in range(len(gt_patches[i]))])
        
        X = img_patches
        Y = np.asarray([value_to_class(np.mean(gt_patches[i])) for i in range(len(gt_patches))])
           
        stop_callback = EarlyStopping(monitor='accuracy', min_delta=0.0001, patience=11, verbose=1, mode='auto')
        self.history = self.cnn.fit(X, Y, verbose=True, epochs = 10, callbacks=[stop_callback])
        
        print('Training completed')
        
    def save(self, filename):
        """ Save the weights of this model. """
        self.model.save_weights(filename)
        
    def load(self, filename):
        """ Load the weights for this model from a file. """
        self.model.load_weights(filename)
        
    def classify(self, X):
        """
        Classify an unseen set of samples.
        This method must be called after "train".
        Returns a list of predictions.
        """
        # Subdivide the images into blocks
        patch_size = self.patch_size
        img_patches = [img_crop(X[i], patch_size, patch_size, patch_size, 0) for i in range(X.shape[0])]
        img_patches = np.asarray([img_patches[i][j] for i in range(len(img_patches)) for j in range(len(img_patches[i]))])
             
        X = img_patches
        
        # Run prediction
        Z = self.cnn.predict(img_patches)
        # Label points with probability greater than 0.5 as 1 and 0 otherwise
        Z = ((Z)> 0.5).astype(int)
        
        return  (Z.reshape(X.shape[0], -1))
    
        