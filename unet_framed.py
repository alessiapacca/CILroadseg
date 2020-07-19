import tensorflow as tf

from keras import Input, Model
from keras.callbacks import ReduceLROnPlateau, EarlyStopping
from keras.layers import Conv2D, MaxPooling2D, Dropout, ReLU, UpSampling2D
from keras.layers.merge import concatenate
from keras.optimizers import Adam
from keras_preprocessing.image import ImageDataGenerator

from decomposer import *
from util.config import *

SEED = 4
WINDOW_SIZE = 304

BATCH_SIZE = 16
STEPS_PER_EPOCH = 400
EPOCHS = 75

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


class FramedUNetModel(ModelBase):

    def __init__(self):
        self.window_size = window_size
        self.model = None

    def initialize(self):
        print('ft10')

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
        #deconv = Dropout(0.5)(deconv)

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
            EarlyStopping(monitor='accuracy', min_delta=0.0001, patience=11, verbose=1, restore_best_weights=True)
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
                temp = X_batch_gen.shape[0]
                for i in range(temp):
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

                    brightness_factor = 1 + (np.random.randint(-100, 200) / 100)
                    X_sample = np.clip(X_sample * brightness_factor, 0, 1)

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
            Z[i] = self.predict_img_rotandmean(X[i])

        return Z.reshape((X.shape[0], X.shape[1], X.shape[2]))

    def predict_img_rotandmean(self, X):
        Z = np.empty((8, X.shape[0], X.shape[1], 1))

        for i in range(4):
            Z[i] = np.rot90(self.predict_img(np.rot90(X, i)), 4 - i)

        for i in range(4):
            Z[4+i] = np.rot90(np.fliplr(self.predict_img(np.fliplr(np.rot90(X, i)))), 4 - i)

        return np.mean(Z, axis=0)

    def predict_img(self, img):
        img1 = img[:WINDOW_SIZE, :WINDOW_SIZE]
        img2 = img[:WINDOW_SIZE, -WINDOW_SIZE:]
        img3 = img[-WINDOW_SIZE:, :WINDOW_SIZE]
        img4 = img[-WINDOW_SIZE:, -WINDOW_SIZE:]

        img5 = img[WINDOW_SIZE//2: -WINDOW_SIZE//2, WINDOW_SIZE//2: -WINDOW_SIZE//2]
        img6 = img[WINDOW_SIZE//2: -WINDOW_SIZE//2, :WINDOW_SIZE]
        img7 = img[WINDOW_SIZE//2: -WINDOW_SIZE//2, -WINDOW_SIZE:]
        img8 = img[:WINDOW_SIZE, WINDOW_SIZE//2: -WINDOW_SIZE//2]
        img9 = img[-WINDOW_SIZE:, WINDOW_SIZE//2: -WINDOW_SIZE//2]

        imgs = np.array([img1, img2, img3, img4, img5, img6, img7, img8, img9])
        labels = self.model.predict(imgs)
        img_label = np.empty((img.shape[0], img.shape[1], 1))
        img_label[-WINDOW_SIZE:, -WINDOW_SIZE:] = labels[3]
        img_label[-WINDOW_SIZE:, :WINDOW_SIZE] = labels[2]
        img_label[:WINDOW_SIZE, -WINDOW_SIZE:] = labels[1]
        img_label[:WINDOW_SIZE, :WINDOW_SIZE] = labels[0]

        img_label[img.shape[0]//2-100: img.shape[0]//2+100, :WINDOW_SIZE] \
            = labels[5][WINDOW_SIZE//2-100: WINDOW_SIZE//2+100, :WINDOW_SIZE]
        img_label[img.shape[0]//2-100: img.shape[0]//2+100, -WINDOW_SIZE:] \
            = labels[6][WINDOW_SIZE//2-100: WINDOW_SIZE//2+100, -WINDOW_SIZE:]
        img_label[:WINDOW_SIZE, img.shape[1]//2-100: img.shape[1]//2+100] \
            = labels[7][:WINDOW_SIZE, WINDOW_SIZE//2-100: WINDOW_SIZE//2+100]
        img_label[-WINDOW_SIZE:, img.shape[1]//2-100: img.shape[1]//2+100] \
            = labels[8][-WINDOW_SIZE:, WINDOW_SIZE//2-100: WINDOW_SIZE//2+100]

        img_label[img.shape[0] // 2 - 100: img.shape[0] // 2 + 100, img.shape[1] // 2 - 100: img.shape[1] // 2 + 100] \
            = labels[4][WINDOW_SIZE // 2 - 100: WINDOW_SIZE // 2 + 100, WINDOW_SIZE // 2 - 100: WINDOW_SIZE // 2 + 100]

        return img_label




