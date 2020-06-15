
import math
import numpy as np

from util.helpers import img_crop
from util.model_base import ModelBase


#
# Decorator class: recomposes the patch classifications into an image (mask)
#
class Recomposer(ModelBase):

    #
    # model - model that takes a window (window_size * window_size) and returns
    #         a patch (patch_size * patch_size)
    # focus_size - size of the focus of a window (i.e. the part in the center of the window that will be classified
    #              by looking at the whole window)
    #
    def __init__(self, model, focus_size = 16):
        self.model = model

        self.focus_size = focus_size

    def initialize(self):
        self.model.initialize()

    def train(self, Y, X):
        self.model.train(Y, X)

    def train_online(self, generator):
        self.model.train_online(generator)

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

        Y = np.repeat(Y, focus_size, axis=1)
        Y = np.repeat(Y, focus_size, axis=2)

        return Y[:, 0:img_size[0], 0:img_size[1]]

    def classify(self, X):
        num_of_img = X.shape[0]
        img_size = (X.shape[1], X.shape[2])

        Y_pred = self.model.classify(X)

        return self.recompose(Y_pred, num_of_img, img_size)

    def summary(self):
        self.model.summary()

    def save(self, filename):
        self.model.save(filename)

    def load(self, filename):
        self.model.load(filename)