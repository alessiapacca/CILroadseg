
import math
import numpy as np

from util.helpers import img_crop
from util.model_base import ModelBase


# Pad given image applying reflect on borders
def pad_image(X, padding):
    return np.lib.pad(X, ((padding, padding), (padding, padding), (0, 0)), 'reflect')

def pad_gt(Y, padding):
    return np.lib.pad(Y, ((padding, padding), (padding, padding)), 'reflect')


#
# Decorator class: decomposes images into fixed-size patches, does a patch-wise classification and then
#                  recomposes the patches into an image
#
class Decomposer(ModelBase):

    #
    # model - model that takes a window (window_size * window_size) and returns
    #         a patch (patch_size * patch_size)
    # window_size - size of the image window the inner model will take into account in order to do a classification
    # focus_size - size of the focus of a window (i.e. the part in the center of the window that will be classified
    #              by looking at the whole window)
    #
    def __init__(self, model, focus_size = 16, window_size = 72):
        self.model = model # (window) -> (patch)

        self.focus_size = focus_size
        self.window_size = window_size

    def initialize(self):
        self.model.initialize()

    def train(self, Y, X):

        padding = (self.window_size - self.focus_size) // 2

        X_pad = np.empty((X.shape[0], X.shape[1] + 2*padding, X.shape[2] + 2*padding, X.shape[3]))
        Y_pad = np.empty((Y.shape[0], Y.shape[1] + 2*padding, Y.shape[2] + 2*padding))

        for i in range(X.shape[0]):
            X_pad[i] = pad_image(X[i], padding)
            Y_pad[i] = pad_gt(Y[i], padding)

        def bootstrap(Y, X):
            window_size = self.window_size
            focus_size = self.focus_size
            print('in decomposer bootstrap code')
            while 1:
                img_id = np.random.choice(X.shape[0])
                img_size = X[img_id].shape

                # window has to fit in the image
                window_center = (np.random.randint(window_size // 2, img_size[0] - window_size // 2),
                                 np.random.randint(window_size // 2, img_size[1] - window_size // 2))

                X_sample = X[img_id][
                    window_center[0] - window_size // 2 : window_center[0] + window_size // 2,
                    window_center[1] - window_size // 2 : window_center[1] + window_size // 2
                ]

                Y_sample = np.mean(Y[img_id][
                    window_center[0] - focus_size // 2 : window_center[0] + focus_size // 2,
                    window_center[1] - focus_size // 2 : window_center[1] + focus_size // 2
                ])

                Y_sample = 1 * (Y_sample > 0.25)
                # 0.79 is the accuracy of the zero classifier on the data we have

                # data augmentation: random flip and rotation (in steps of 90°)
                # TODO: arbitrary rotation
                flip = np.random.choice(2)
                rot_step = np.random.choice(4)

                if flip: X_sample = np.fliplr(X_sample)
                X_sample = np.rot90(X_sample, rot_step)

                yield Y_sample, X_sample

        #self.train_online(bootstrap(Y_pad, X_pad))
        self.model.train(Y,X)

    def train_online(self, generator):
        self.model.train_online(generator)

    #
    # Divides the given images into focus_size x focus_size squares, in order to classify them.
    # Also the surrounding pixels are taken, up to a window_size x window_size matrix.
    # X - the list of images. This should be a numpy array with shape (num_images, width, height, channels)
    #
    def create_windows(self, X):
        padding = (self.window_size - self.focus_size) // 2

        windows = np.asarray([img_crop(X[i], self.focus_size, self.focus_size, self.focus_size, padding) for i in range(X.shape[0])])
        return windows.reshape((-1, windows.shape[2], windows.shape[3], windows.shape[4]))

    #
    # Recomposes the images from the classified patches.
    # Y - numpy array of shape (num_of_img, ) containing the list of classes for each patch.
    #     The classification is assumed to be patch-wise, with only one class for all the pixels of the path
    # num_of_img - the number of images
    # img_size - tuple (width, height) indicating the size of the images to reconstruct
    #
    def recompose(self, Y, num_of_img, img_size):
        focus_size = self.focus_size

        Y = Y.reshape((num_of_img, math.ceil(img_size[0] / focus_size), math.ceil(img_size[1] / focus_size)))
        Y = np.transpose(Y, axes=[0, 2, 1])

        Y = np.repeat(Y, focus_size, axis=1)
        Y = np.repeat(Y, focus_size, axis=2)

        return Y[:, 0:img_size[0], 0:img_size[1]]

    def classify(self, X):
        num_of_img = X.shape[0]
        img_size = (X.shape[1], X.shape[2])

        X_windows = self.create_windows(X)
        Y_pred = self.model.classify(X_windows)

        return self.recompose(Y_pred, num_of_img, img_size)
        #return Y_pred

    def summary(self):
        self.model.summary()

    def save(self, filename):
        self.model.save(filename)

    def load(self, filename):
        self.model.load(filename)