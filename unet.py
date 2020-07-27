
from keras import Input, Model
from keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
from keras.layers import Conv2D, MaxPooling2D, Dropout, ReLU, UpSampling2D
from keras.layers.merge import concatenate
from keras.optimizers import Adam
from keras_preprocessing.image import ImageDataGenerator

from decomposer import *
from util.config import *


SEED = 4
WINDOW_SIZE = 304

BATCH_SIZE = 8
STEPS_PER_EPOCH = 500
EPOCHS = 50

def convolve(input, filters, kernel_size=3):
    conv = Conv2D(filters=filters, kernel_size=kernel_size, padding='same', kernel_initializer='normal')(input)
    #conv = BatchNormalization()(conv)
    conv = ReLU()(conv)

    conv = Conv2D(filters=filters, kernel_size=kernel_size, padding='same', kernel_initializer='normal')(conv)
    #conv = BatchNormalization()(conv)
    conv = ReLU()(conv)

    return conv

def transpose_convolve(input, filters):
    up = UpSampling2D(size=2)(input)
    up = Conv2D(filters=filters, kernel_size=3, padding='same', kernel_initializer="normal")(up)
    return ReLU()(up)
    #return Conv2DTranspose(filters=filters, strides=2, kernel_size=3, padding='same')(input)

def pool(input):
    return MaxPooling2D(pool_size=2)(input)


class UNetModel(ModelBase):

    #
    # dense_prediction - If True, 9 predictions per image are done, as explained in the report
    #                  - If False, only 4 predictions are done
    # augment_colors   - If True, also brightness and contrast factors are augmented.
    #
    def __init__(self, dense_prediction=True, augment_colors=True):
        self.model = None

        self.dense_prediction = dense_prediction
        self.augment_colors = augment_colors

    def initialize(self):
        inputs = Input((WINDOW_SIZE, WINDOW_SIZE, 3))

        base_filters = 32
        unet_depth = 4

        down_socket = []
        dropouts = [0, 0, 0, 0]
        kernel_sizes = [5, 3, 3, 3]

        # Downsampling construction
        down = inputs
        for i in range(unet_depth):
            conv = convolve(down, base_filters, kernel_size=kernel_sizes[i])

            if dropouts[i] > 0: conv = Dropout(dropouts[i])(conv)
            down = pool(conv)

            down_socket.append(conv)
            base_filters *= 2

        # Bottom Layers
        deconv = convolve(down, base_filters)
        #deconv = Dropout(0.2)(deconv)

        for i in reversed(range(unet_depth)):
            base_filters //= 2
            up = transpose_convolve(deconv, base_filters)

            merge = concatenate([down_socket[i], up], axis=3)
            deconv = convolve(merge, base_filters)

        output = Conv2D(1, 1, activation='sigmoid')(deconv)

        self.model = Model(inputs=inputs, outputs=output)

    def load(self, filename):
        self.model.load_weights(filename)

    def save(self, filename):
        self.model.save_weights(filename)

    def train(self, Y, X):
        Y = Y.reshape((-1, 400, 400, 1))

        self.model.summary()

        opt = Adam()
        #opt = SGD()

        self.model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])

        callbacks = [
            ReduceLROnPlateau(monitor='accuracy', min_delta=0.0001, patience=5, verbose=1, factor=0.5),
            EarlyStopping(monitor='accuracy', min_delta=0.0001, patience=11, verbose=1),
            ModelCheckpoint(filepath='saves/checkpoints/cp-{epoch}.h5',
                save_weights_only=True,
                monitor='accuracy')
        ]

        def datagen(X, Y):
            datagen = ImageDataGenerator(rotation_range=360.,
                                         horizontal_flip=True,
                                         vertical_flip=True,
                                         zoom_range=0.2,
                                         fill_mode='reflect')

            Xgen = datagen.flow(X, batch_size=BATCH_SIZE, seed=SEED)
            Ygen = datagen.flow(Y, batch_size=BATCH_SIZE, seed=SEED)

            for x, y in zip(Xgen, Ygen):
                yield x, y

        def random_cropper(generator):
            while 1:
                X_batch = np.empty((BATCH_SIZE, WINDOW_SIZE, WINDOW_SIZE, 3))
                Y_batch = np.empty((BATCH_SIZE, WINDOW_SIZE, WINDOW_SIZE, 1))

                X_batch_gen, Y_batch_gen = next(generator)
                for i in range(X_batch_gen.shape[0]):
                    cur_img, cur_lbl = X_batch_gen[i], Y_batch_gen[i]

                    window_center = (np.random.randint(WINDOW_SIZE // 2, cur_img.shape[0] - WINDOW_SIZE // 2),
                                     np.random.randint(WINDOW_SIZE // 2, cur_img.shape[1] - WINDOW_SIZE // 2))

                    X_sample = cur_img[
                        window_center[0] - WINDOW_SIZE // 2: window_center[0] + WINDOW_SIZE // 2,
                        window_center[1] - WINDOW_SIZE // 2: window_center[1] + WINDOW_SIZE // 2
                    ]

                    Y_sample = cur_lbl[
                        window_center[0] - WINDOW_SIZE // 2: window_center[0] + WINDOW_SIZE // 2,
                        window_center[1] - WINDOW_SIZE // 2: window_center[1] + WINDOW_SIZE // 2
                    ]

                    if self.augment_colors:
                        contrast_factor = 1 + (np.random.randint(0, 100) / 100)
                        brightness_factor = 1 + (np.random.randint(0, 100) / 100)

                        X_sample = np.clip(X_sample * brightness_factor, 0, 1)
                        X_sample = np.clip(0.5 + contrast_factor * (X_sample - 0.5), 0, 1)

                    X_batch[i] = X_sample
                    Y_batch[i] = Y_sample

                yield (X_batch, Y_batch)

        self.model.fit_generator(
            random_cropper(datagen(X, Y)),
            steps_per_epoch=STEPS_PER_EPOCH,
            epochs=EPOCHS,
            verbose=2,
            callbacks=callbacks
        )

    def classify(self, X):
        Z = np.empty((X.shape[0], X.shape[1], X.shape[2], 1))
        for i in range(X.shape[0]):
            Z[i] = self.segment_image(X[i])

        return Z.reshape((X.shape[0], X.shape[1], X.shape[2]))

    def segment_image(self, X):
        fragments = []

        w = X.shape[0]
        h = X.shape[1]
        WS = WINDOW_SIZE

        fragments.append(X[:WS, :WS])
        fragments.append(X[:WS, -WS:])
        fragments.append(X[-WS:, :WS])
        fragments.append(X[-WS:, -WS:])

        if self.dense_prediction:
            fragments.append(X[w//2-WS//2: w//2+WS//2, h//2-WS//2:h//2+WS//2])
            fragments.append(X[w//2-WS//2: w//2+WS//2, :WS])
            fragments.append(X[w//2-WS//2: w//2+WS//2, -WS:])
            fragments.append(X[:WS, h//2-WS//2:h//2+WS//2])
            fragments.append(X[-WS:, h//2-WS//2:h//2+WS//2])

        Y = self.model.predict(np.array(fragments))

        Z = np.empty((w, h, 1))
        Z[-WS:, -WS:] = Y[3]
        Z[-WS:, :WS] = Y[2]
        Z[:WS, -WS:] = Y[1]
        Z[:WS, :WS] = Y[0]

        if self.dense_prediction:
            Z[w//2-100: w//2+100, :WS]  = Y[5][WS//2-100: WS//2+100, :WS]
            Z[w//2-100: w//2+100, -WS:] = Y[6][WS//2-100: WS//2+100, -WS:]
            Z[:WS, h//2-100: h//2+100]  = Y[7][:WS, WS//2-100: WS//2+100]
            Z[-WS:, h//2-100: h//2+100] = Y[8][-WS:, WS//2-100: WS//2+100]

            Z[w//2-100: w//2+100, h//2-100: h//2+100] = Y[4][WS//2-100: WS//2+100, WS//2-100: WS//2+100]

        return Z




