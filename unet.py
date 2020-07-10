import tensorflow as tf

from keras import Input, Model, Sequential
from keras.callbacks import ReduceLROnPlateau, EarlyStopping
from keras.layers import Conv2D, MaxPooling2D, Dropout, UpSampling2D, ReLU, Conv2DTranspose, LeakyReLU, \
    BatchNormalization
from keras.layers.merge import concatenate
from keras.optimizers import Adam, SGD
from keras import backend as K
from keras_preprocessing.image import ImageDataGenerator

from decomposer import *
from util.config import *

SEED = 1998
BATCH_SIZE = 8
STEPS_PER_EPOCH = 300


def jaccard_distance_loss(y_true, y_pred, smooth=100):
    intersection = K.sum(K.abs(y_true * y_pred), axis=-1)
    sum_ = K.sum(K.abs(y_true) + K.abs(y_pred), axis=-1)
    jac = (intersection + smooth) / (sum_ - intersection + smooth)
    return (1 - jac) * smooth

def convolve(filters):
    convolve.id += 1
    return Sequential([
        Conv2D(filters, 3, padding='same', kernel_initializer='random_uniform'),
        ReLU(),
        Dropout(0.25),
        Conv2D(filters, 3, padding='same', kernel_initializer='random_uniform'),
        ReLU(),
        Dropout(0.25)
    ], name='conv_block_'+ str(convolve.id))
convolve.id = -1

def transpose_convolve(filters):
    transpose_convolve.id += 1
    return Conv2DTranspose(filters=filters, strides=2, kernel_size=3, padding='same', name='tconv_block_'+ str(convolve.id))
transpose_convolve.id = -1

def pool():
    return MaxPooling2D(pool_size=2)



def trainGenerator(X, Y):

    data_gen_args = dict(rotation_range=0.2,
                         width_shift_range=0.05,
                         height_shift_range=0.05,
                         shear_range=0.05,
                         zoom_range=0.05,
                         horizontal_flip=True,
                         fill_mode='nearest')

    image_datagen = ImageDataGenerator(**data_gen_args)
    mask_datagen = ImageDataGenerator(**data_gen_args)
    image_generator = image_datagen.flow(X, batch_size=BATCH_SIZE, seed=SEED)
    mask_generator = mask_datagen.flow(Y, batch_size=BATCH_SIZE, seed=SEED)

    for img, mask in zip(image_generator, mask_generator):
        yield (img, mask)


class UNetModel(ModelBase):

    def __init__(self):
        self.window_size = window_size
        self.model = None

    def initialize(self):
        print('y')

        inputs = Input((400, 400, 3))

        base_filters = 64

        # Downsampling (400x400x3 -> 25x25x1024)
        conv1 = convolve(base_filters)(inputs)
        down1 = pool()(conv1)

        conv2 = convolve(base_filters * 2)(down1)
        down2 = pool()(conv2)

        conv3 = convolve(base_filters * 4)(down2)
        down3 = pool()(conv3)

        conv4 = convolve(base_filters * 8)(down3)
        down4 = pool()(conv4)

        # Bottom Layers
        deconv5 = convolve(base_filters * 16)(down4)
        up4 = transpose_convolve(base_filters * 8)(deconv5)

        # Upsampling (25x25x1024 -> 400x400x1)
        merge4 = concatenate([conv4, up4], axis=3)
        deconv4 = convolve(base_filters * 8)(merge4)
        up3 = transpose_convolve(base_filters * 4)(deconv4)

        merge3 = concatenate([conv3, up3], axis=3)
        deconv3 = convolve(base_filters * 4)(merge3)
        up2 = transpose_convolve(base_filters * 2)(deconv3)

        merge2 = concatenate([conv2, up2], axis=3)
        deconv2 = convolve(base_filters * 2)(merge2)
        up1 = transpose_convolve(base_filters)(deconv2)

        merge1 = concatenate([conv1, up1], axis=3)
        deconv1 = convolve(base_filters)(merge1)

        #logits = Conv2D(2, 3, activation='relu', padding='same', data_format='channels_last')(deconv1)
        output = Conv2D(1, 1, activation='sigmoid')(deconv1)

        self.model = Model(inputs=inputs, outputs=output)

    def load(self, filename):
        self.model.load_weights(filename)

    def save(self, filename):
        self.model.save_weights(filename)

    def train(self, Y, X):
        Y = Y.reshape((-1, 400, 400, 1))

        self.model.summary()

        opt = Adam(lr=0.001)
        #opt = SGD()

        self.model.compile(optimizer=opt, loss=jaccard_distance_loss, metrics=['accuracy'])

        callbacks = [
            ReduceLROnPlateau(monitor='loss', min_delta=0.0001, patience=5, verbose=1, factor=0.5),
            EarlyStopping(monitor='accuracy', min_delta=0.0001, patience=11, verbose=1)
        ]

        tf.random.set_seed(3)

        def datagen(X, Y):
            while 1:
                X_batch = np.empty((BATCH_SIZE, 400, 400, 3))
                Y_batch = np.empty((BATCH_SIZE, 400, 400, 1))

                for i in range(BATCH_SIZE):
                    img_id = np.random.choice(X.shape[0])
                    x = X[img_id]
                    y = Y[img_id]

                    flip = np.random.choice(2)
                    rot_step = np.random.choice(4)

                    if flip:
                        x = np.fliplr(x)
                        y = np.fliplr(y)

                    X_batch[i] = np.rot90(x, rot_step)
                    Y_batch[i] = np.rot90(y, rot_step)

                yield (X_batch, Y_batch)

        self.model.fit_generator(
            trainGenerator(X, Y),
            steps_per_epoch=STEPS_PER_EPOCH,
            epochs=epochs,
            verbose=2,
            callbacks=callbacks
        )

    # This method is used to adapt the 400x400 size of the CNN to the 608x608 size of test data.
    # It just predicts four 400x400 images to compose the mask, overwriting overlapping pixels.
    def predict_img(self, img):
        img1 = img[:400,:400]
        img2 = img[:400,-400:]
        img3 = img[-400:,:400]
        img4 = img[-400:,-400:]
        imgs = np.array([img1,img2,img3,img4])
        labels = self.model.predict(imgs)
        img_label = np.empty((608,608,1))
        img_label[-400:,-400:] = labels[3]
        img_label[-400:,:400] = labels[2]
        img_label[:400,-400:] = labels[1]
        img_label[:400,:400] = labels[0]
        return img_label

    def classify(self, X):
        # FIXME this takes a great amount of memory
        # model.predict() should be called once.

        Z = np.empty((X.shape[0], X.shape[1], X.shape[2], 1))
        for i in range(X.shape[0]):
            Z[i] = self.predict_img(X[i])

        return Z.reshape((X.shape[0],X.shape[1],X.shape[2]))



