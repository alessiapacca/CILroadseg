
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.models import Model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, GlobalMaxPooling2D, Dropout,Activation,Flatten
from tensorflow.keras.layers import Convolution2D, MaxPooling2D
from tensorflow.keras.callbacks import EarlyStopping,ReduceLROnPlateau
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import optimizers
from decomposer import *
from util.config import *

#from keras.utils import np_utils

def batch_generator(bootstrap):
    #print('\n\n In batch_generator')
    while 1:
        # create one batch
        X_batch = np.empty((batch_size, window_size, window_size, 3))
        #print(X_batch.shape)
        Y_batch = np.empty(batch_size)
         
        for i in range(batch_size):
            Y_batch[i], X_batch[i] = next(bootstrap)
            # Y_batch[i] = np_utils.to_categorical(label, 2)

        yield (X_batch, Y_batch)
        
def pad_image(data, padding):
    """
    Extend the canvas of an image. Mirror boundary conditions are applied.
    """
    if len(data.shape) < 3:
        # Greyscale image (ground truth)
        data = np.lib.pad(data, ((padding, padding), (padding, padding)), 'reflect')
    else:
        # RGB image
        data = np.lib.pad(data, ((padding, padding), (padding, padding), (0,0)), 'reflect')
    
    return data


foreground_threshold = 0.25 # percentage of pixels > 1 required to assign a foreground label to a patch

def value_to_class(v):
    df = np.sum(v)
    if df > foreground_threshold:
        return 1
    else:
        return 0


class ResnetModel(ModelBase):

    def __init__(self):
        print('initialize resnet 50 model')
        self.model = None

    def initialize(self):
        print('create resnet 50 model')
        input_shape=(72,72,3)
        restnet = ResNet50(weights = "imagenet", include_top = False,input_shape = input_shape)
        output = restnet.layers[-1].output
        output = Flatten()(output)
        restnet = Model(restnet.input, output)

                       
        for i, layer in enumerate(restnet.layers):
            print(i, layer.name)

        for layer in restnet.layers[:75]:
            layer.trainable = False

        for layer in restnet.layers[0:]:
            layer.trainable = True
        
        model = Sequential()
        model.add(restnet)
        model.add(Dense(64, activation='relu', input_dim=input_shape))
        model.add(Dropout(rate = 0.3))
        model.add(Dense(64, activation='relu'))
        model.add(Dropout(rate = 0.3))
        model.add(Dense(1, activation='sigmoid'))
        '''
        model.compile(loss='binary_crossentropy',
              optimizer=optimizers.RMSprop(lr=2e-5),
              metrics=['accuracy'])
        '''
        print('created resnet 50 model successfully')
        self.model = model
        self.model.summary()

    def load(self, filename):
        print(filename)
        self.model.load_weights(filename)

    def save(self, filename):
        self.model.save_weights(filename)

    def train_online(self, generator):
        # this generator does the bootstrap of a single sample.
        # batch_generator will create batches of these samples

        # TODO choose an optimizer, and a loss function
        print('\n\n In train online')
        adam = Adam(lr=0.001)  # Adam optimizer with default initial learning rate

        self.model.compile(optimizer=adam, loss='binary_crossentropy', metrics=['accuracy'])

        callbacks = [
            EarlyStopping(monitor='loss', patience=2, verbose=1, mode='auto'),
            ReduceLROnPlateau(monitor='loss', factor=0.5, patience=5,
                                    verbose=1, mode='auto', epsilon=0.0001, cooldown=0, min_lr=0)
            
        ]
            # This callback reduces the learning rate when the training accuracy does not improve any more
        lr_callback = ReduceLROnPlateau(monitor='accuracy', factor=0.5, patience=5,
                                    verbose=1, mode='auto', epsilon=0.0001, cooldown=0, min_lr=0)

        # Stops the training process upon convergence
        stop_callback = EarlyStopping(monitor='accuracy', min_delta=0.0001, patience=11, verbose=1, mode='auto')

        np.random.seed(3) # fix randomness
        self.model.fit(batch_generator(generator),
                                 steps_per_epoch=steps_per_epoch,
                                 epochs=epochs,
                                 verbose=1,
                                callbacks=[lr_callback, stop_callback])
        
    
    def train(self,Y,X):
        
        print('Training set shape: ', X.shape)
        padding = (window_size - patch_size) // 2
        print(padding)
        samples_per_epoch = X.shape[0]*X.shape[1]*X.shape[2]//256 # Arbitrary value

        # Pad training set images (by appling mirror boundary conditions)
        X_new = np.empty((X.shape[0],
                     X.shape[1] + 2*padding, X.shape[2] + 2*padding,
                     X.shape[3]))
        Y_new = np.empty((Y.shape[0],
                     Y.shape[1] + 2*padding, Y.shape[2] + 2*padding))
        for i in range(X.shape[0]):
            X_new[i] = pad_image(X[i], padding)
            Y_new[i] = pad_image(Y[i], padding)
        X = X_new
        Y = Y_new
        print(X.shape)
        print(Y.shape)

    
      
        samples_per_epoch = 200
        
        def generate_minibatch():
            """
            Procedure for real-time minibatch creation and image augmentation.
            This runs in a parallel thread while the model is being trained.
            """
            k = 0
            while 1:
                # Generate one minibatch
                X_batch = np.empty((batch_size, window_size, window_size, 3))
                Y_batch = np.empty((batch_size, 2))
                k = k+1;
                #print('\n passing batch nu {} with size {}'.format(k, batch_size))    
                for i in range(batch_size):
                    # Select a random image
                    idx = np.random.choice(X.shape[0])
                    shape = X[idx].shape

                    # Sample a random window from the image
                    center = np.random.randint(window_size//2, shape[0] - window_size//2, 2)
                    sub_image = X[idx][center[0]-window_size//2:center[0]+window_size//2,
                                       center[1]-window_size//2:center[1]+window_size//2]
                    gt_sub_image = Y[idx][center[0]-patch_size//2:center[0]+patch_size//2,
                                          center[1]-patch_size//2:center[1]+patch_size//2]

                    # The label does not depend on the image rotation/flip (provided that the rotation is in steps of 90°)
                    threshold = 0.25
                    label = (np.array([np.mean(gt_sub_image)]) > threshold) * 1

                    # Image augmentation
                    # Random flip
                    if np.random.choice(2) == 0:
                        # Flip vertically
                        sub_image = np.flipud(sub_image)
                    if np.random.choice(2) == 0:
                        # Flip horizontally
                        sub_image = np.fliplr(sub_image)

                    # Random rotation in steps of 90°
                    num_rot = np.random.choice(4)
                    sub_image = np.rot90(sub_image, num_rot)

                    #label = np_utils.to_categorical(label, nb_classes)
                    X_batch[i] = sub_image
                    Y_batch[i] = label
                    
                yield (X_batch, Y_batch)
        
        #epochs = 2
        # This callback reduces the learning rate when the training accuracy does not improve any more
        lr_callback = ReduceLROnPlateau(monitor='accuracy', factor=0.5, patience=5,
                                    verbose=1, mode='auto', epsilon=0.0001, cooldown=0, min_lr=0)

        # Stops the training process upon convergence
        stop_callback = EarlyStopping(monitor='accuracy', min_delta=0.0005, patience=3, verbose=1, mode='auto')
        
        self.model.compile(loss='binary_crossentropy',
              optimizer=optimizers.RMSprop(lr=2e-5),
              metrics=['accuracy'])

        self.model.fit(generate_minibatch(),
                        steps_per_epoch=samples_per_epoch,
                        epochs=epochs,
                        verbose=1,
                        callbacks=[lr_callback, stop_callback])

    def classify(self, X):
        Z = self.model.predict(X)
        #Z = (Z[:, 0] < Z[:, 1]) * 1
        Z = ((Z)> 0.5).astype(int)

        return Z