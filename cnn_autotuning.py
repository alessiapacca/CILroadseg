from decomposer import *
from util.config import *
from util.helpers import view_image_and_mask
import tensorflow as tf
from tensorflow import keras
import IPython
from sklearn.model_selection import train_test_split

import IPython
import kerastuner as kt

glb_window_size = 72
glb_patch_size = 16

def model_builder(hp):
    input_shape = (glb_window_size, glb_window_size, 3)
    first_set_cnn = hp.Int('cnn_set_0', min_value = 1, max_value = 2, step = 1)
    second_set_cnn = first_set_cnn + hp.Int('cnn_set_1', min_value = 1, max_value = 4, step = 1)
    first_set_dense = hp.Int('dense_set_0', min_value = 1, max_value = 2, step = 1)
    second_set_dense = first_set_dense + hp.Int('dense_set_1', min_value = 1, max_value = 4, step = 1)
    model = keras.Sequential()

    hp_filter = hp.Int('filter_input', min_value = 32, max_value = 256, step = 32)
    hp_activation = hp.Choice('act_input', values = ['relu', 'selu', 'elu'])
    hp_dropout = hp.Float('drop_input', min_value = 0.1, max_value = 0.9, step = 0.1)
    model.add(keras.layers.Conv2D(  filters = hp_filter,
                                    activation = hp_activation,
                                    kernel_size = (3, 3),
                                    input_shape = input_shape,
                                    padding = 'same'
                                ))
    model.add(keras.layers.MaxPooling2D(    pool_size = (2, 2),
                                            padding = 'same'
                                        ))
    model.add(keras.layers.Dropout(hp_dropout))

    for i in range(first_set_cnn):
        hp_filter = hp.Int('filter' + str(i), min_value = 32, max_value = 256, step = 32)
        hp_activation = hp.Choice('act' + str(i), values = ['relu', 'selu', 'elu'])
        hp_kernel_size = hp.Int('size' + str(i), min_value = 2, max_value = 5, step = 1)
        hp_dropout = hp.Float('drop' + str(i), min_value = 0.1, max_value = 0.9, step = 0.1)
        model.add(keras.layers.Conv2D(  filters = hp_filter,
                                        activation = hp_activation,
                                        kernel_size = (hp_kernel_size, hp_kernel_size),
                                        padding = 'same'
                                    ))
        model.add(keras.layers.MaxPooling2D(    pool_size = (2, 2),
                                                padding='same'
                                            ))
        model.add(keras.layers.Dropout(hp_dropout))

    for i in range(first_set_cnn, second_set_cnn):
        hp_filter = hp.Int('filter' + str(i), min_value = 16, max_value = 128, step = 16)
        hp_activation = hp.Choice('act' + str(i), values = ['relu', 'selu', 'elu'])
        hp_kernel_size = hp.Int('size' + str(i), min_value = 2, max_value = 5, step = 1)
        hp_dropout = hp.Float('drop' + str(i), min_value = 0.1, max_value = 0.9, step = 0.1)
        model.add(keras.layers.Conv2D(  filters = hp_filter,
                                        activation = hp_activation,
                                        kernel_size = (hp_kernel_size, hp_kernel_size),
                                        padding = 'same'
                                    ))
        model.add(keras.layers.MaxPooling2D(    pool_size = (2, 2),
                                                padding = 'same'
                                            ))
        model.add(keras.layers.Dropout(hp_dropout))

    model.add(keras.layers.Flatten())
    
    for i in range(first_set_dense):
        hp_units = hp.Int('units_dense' + str(i), min_value = 32, max_value = 256, step = 32)
        hp_activation = hp.Choice('act_dense' + str(i), values = ['relu', 'selu', 'elu'])
        hp_dropout = hp.Float('drop_dense' + str(i), min_value = 0.1, max_value = 0.9, step = 0.1)
        model.add(keras.layers.Dense(units = hp_units,
                        activation = hp_activation
                        ))
        model.add(keras.layers.Dropout(hp_dropout))
    
    for i in range(first_set_dense, second_set_dense):
        hp_units = hp.Int('units_dense' + str(i), min_value = 16, max_value = 128, step = 16)
        hp_activation = hp.Choice('act_dense' + str(i), values = ['relu', 'selu', 'elu'])
        hp_dropout = hp.Float('drop_dense' + str(i), min_value = 0.1, max_value = 0.9, step = 0.1)
        model.add(keras.layers.Dense(units = hp_units,
                        activation = hp_activation
                        ))
        model.add(keras.layers.Dropout(hp_dropout))

    model.add(keras.layers.Dense(1, activation='sigmoid'))

    # model.summary()
    
    # hp_optimizer = hp.Choice('opt', values = ['rmsprop', 'adam', 'sgd'])
    # model.compile(optimizer=hp_optimizer, loss='binary_crossentropy', metrics=['accuracy'])

    hp_learning_rate = hp.Float('learning_rate', min_value = 0.000001, max_value = 0.0001, step = 0.000001)
    opt = tf.keras.optimizers.RMSprop(learning_rate=hp_learning_rate)
    model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])
    return model

def batch_generator(bootstrap):
    while 1:
        X_batch = np.empty((batch_size, window_size, window_size, 3))
        Y_batch = np.empty((batch_size))

        for i in range(batch_size):
            Y_batch[i], X_batch[i] = next(bootstrap)

        yield (X_batch, Y_batch)

def cv_generator(bootstrap):
    while 1:
        X_batch = np.empty((batch_size, window_size, window_size, 3))
        Y_batch = np.empty((batch_size))

        for i in range(batch_size):
            Y_batch[i], X_batch[i] = next(bootstrap)

        X_train, X_test, Y_train, Y_test = train_test_split(X_batch, Y_batch, test_size=0.33)

        yield X_train, Y_train, X_test, Y_test


class CNNModel(ModelBase):

    def __init__(self, patch_size=16, window_size=72):
        self.model = None
        self.patch_size = patch_size
        self.window_size = window_size
        global glb_patch_size
        global glb_window_size
        glb_patch_size = patch_size
        glb_window_size = window_size

    def initialize(self):
        self.tuner = kt.Hyperband(model_builder,
                     objective = 'val_accuracy', 
                     max_epochs = epochs,
                     executions_per_trial = 3,
                     hyperband_iterations = 8,
                     factor = 3,
                     project_name = 'cnn_v2_2424_val_a_tr2'
                     #overwrite = True
                     )

    def load(self, filename):
        self.model.load(filename)

    def save(self, filename):
        self.model.save(filename)

    def train_online(self, generator, val_generator = None):


        #if self.model is not assigned then we tune parameters. This is done for the cross validation folds
        if not isinstance(self.model, keras.Sequential):
            img_train, label_train, img_test, label_test = next(cv_generator(generator))
            print(label_train)
            self.tuner.search(img_train, 
                label_train, 
                epochs = 10, 
                steps_per_epoch=20,
                #  validation_split=0.2, 
                validation_data = (img_test, label_test),
                callbacks = [ClearTrainingOutput()]
            )

            # Get the optimal hyperparameters and store them
            self.best_hps = self.tuner.get_best_hyperparameters(num_trials = 1)[0]

        # Build the model with the optimal hyperparameters
        self.model = self.tuner.hypermodel.build(self.best_hps)
        
        self.model.summary()
        # self.model.fit(img_train, label_train, epochs = epochs, steps_per_epoch=steps_per_epoch, validation_data = (img_test, label_test))
        callbacks = [
            tf.keras.callbacks.EarlyStopping(monitor='accuracy', patience=2, verbose=1, mode='auto')
        ]
        self.model.fit(batch_generator(generator),
                                 steps_per_epoch=steps_per_epoch,
                                 epochs=epochs,
                                 verbose=1,
                                 callbacks=callbacks)


    def classify(self, X):
        Z = self.model.predict(X)
        Z = (Z > 0.5) * 1

        return Z

class ClearTrainingOutput(tf.keras.callbacks.Callback):
  def on_train_end(*args, **kwargs):
    IPython.display.clear_output(wait = True)