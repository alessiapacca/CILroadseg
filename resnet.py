
from keras.applications.resnet50 import ResNet50
from keras.models import Model
from keras.layers import Dense, GlobalMaxPooling2D
from decomposer import *
from util.config import *
from keras.optimizers import Adam
from keras.utils import np_utils
from keras.callbacks import EarlyStopping


def batch_generator(bootstrap):
    while 1:
        # create one batch
        X_batch = np.empty((batch_size, window_size, window_size, 3))
        Y_batch = np.empty((batch_size, 2))

        for i in range(batch_size):
            label, X_batch[i] = next(bootstrap)
            Y_batch[i] = np_utils.to_categorical(label, 2)

        yield (X_batch, Y_batch)


class ResnetModel(ModelBase):

    def __init__(self):
        self.model = None

    def initialize(self):
        self.model = ResNet50(weights = "imagenet", include_top = False)

        x = self.model.output
        x = GlobalMaxPooling2D()(x)
        x = Dense(1024, activation='relu')(x)
        x = Dense(1024, activation='relu')(x)
        x = Dense(512, activation='relu')(x)

        # last layer specifies the number of outputs. We have a binary output, but it's better to use a single unit in the last dense layer.
        # https://stackoverflow.com/questions/54797065/resnet-for-binary-classification-just-2-values-of-cross-validation-accuracy
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

    def train_online(self, generator):
        # this generator does the bootstrap of a single sample.
        # batch_generator will create batches of these samples

        # TODO choose an optimizer, and a loss function
        adam = Adam(lr=0.001)  # Adam optimizer with default initial learning rate

        self.model.compile(optimizer=adam, loss='binary_crossentropy', metrics=['accuracy'])

        callbacks = [
            EarlyStopping(monitor='loss', min_delta=0.0001, patience=11, verbose=1, mode='auto')
        ]

        np.random.seed(3) # fix randomness
        self.model.fit_generator(batch_generator(generator),
                                 steps_per_epoch=steps_per_epoch,
                                 epochs=epochs,
                                 verbose=1,
                                 callbacks=callbacks)

    def classify(self, X):
        Z = self.model.predict(X)
        Z = (Z[:, 0] < Z[:, 1]) * 1

        return Z