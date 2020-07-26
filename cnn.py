from keras import Input, Model
from keras.callbacks import ReduceLROnPlateau, EarlyStopping
from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense, ReLU
from keras.optimizers import Adam
from keras.regularizers import l2

from decomposer import *
from util.visualize import plot_history

WINDOW_SIZE = 200
BATCH_SIZE = 150

EPOCHS = 200
STEPS_PER_EPOCH = 500

def batch_generator(bootstrap):
    while 1:
        # create one batch
        X_batch = np.empty((BATCH_SIZE, WINDOW_SIZE, WINDOW_SIZE, 3))
        Y_batch = np.empty(BATCH_SIZE)

        for i in range(BATCH_SIZE):
            Y_batch[i], X_batch[i] = next(bootstrap)

        yield (X_batch, Y_batch)

def max_pool(input, pool_size=2):
    return MaxPooling2D(pool_size=pool_size)(input)

def conv_block(input, filters, kernel_size=3, dropout=0.0):
    conv = Conv2D(filters=filters, kernel_size=kernel_size, kernel_initializer='normal')(input)
    #conv = BatchNormalization()(conv)
    conv = ReLU()(conv)
    if dropout > 0.0: conv = Dropout(dropout)(conv)

    return max_pool(conv)

class ConvNetModel(ModelBase):

    def __init__(self):
        self.window_size = WINDOW_SIZE
        self.model = None

    def initialize(self):
        input_layer = Input(shape=(self.window_size, self.window_size, 3))

        conv = conv_block(input_layer, filters=32, kernel_size=5, dropout=0.3)
        conv = conv_block(conv, filters=64,  dropout=0.3)
        conv = conv_block(conv, filters=128, dropout=0.3)
        conv = conv_block(conv, filters=256, dropout=0.3)

        cfy = Flatten()(conv)

        cfy = Dense(64, activation='relu', kernel_regularizer=l2(1e-6))(cfy)
        cfy = Dropout(0.5)(cfy)

        cfy = Dense(64, activation='relu', kernel_regularizer=l2(1e-6))(cfy)
        cfy = Dropout(0.5)(cfy)

        output = Dense(1, activation='sigmoid', kernel_regularizer=l2(1e-6))(cfy)

        self.model = Model(inputs=input_layer, outputs=output)

    def load(self, filename):
        self.model.load_weights(filename)

    def save(self, filename):
        self.model.save_weights(filename)

    def train_online(self, generator, val_generator = None):
        # this generator does the bootstrap of a single sample.
        # batch_generator will create batches of these samples

        self.model.summary()
        self.model.compile(optimizer=Adam(0.001), loss='binary_crossentropy', metrics=['accuracy'])

        callbacks = [
            ReduceLROnPlateau(monitor='accuracy', min_delta=0.0001, patience=5, verbose=1, factor=0.5),
            EarlyStopping(monitor='accuracy', min_delta=0.0001, patience=11, verbose=1)
        ]

        history = self.model.fit_generator(
                        batch_generator(generator),
                        steps_per_epoch=STEPS_PER_EPOCH,
                        epochs=EPOCHS,
                        verbose=2,
                        callbacks=callbacks,
                        validation_data=batch_generator(val_generator),
                        validation_steps=100
        )

        plot_history(history)

    def classify(self, X):
        return self.model.predict(X)