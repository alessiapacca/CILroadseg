
import numpy as np

import matplotlib.image as mpimg
from PIL import Image

from util.model_base import ModelBase


def img_crop_gt(X):
    patches = []

    for i in range(0, X.shape[1], 16):
        for j in range(0, X.shape[0], 16):
            im_patch = X[j:j+16, i:i+16]
            patches.append(im_patch)

    return patches

# Divides gt images into 16x16 patches
def patchify_gt(X):
    X_patches = np.asarray([img_crop_gt(X[i]) for i in range(X.shape[0])])

    return X_patches.reshape((-1, X_patches.shape[2], X_patches.shape[3]))

class Patchifier(ModelBase):

    def __init__(self, model, threshold=0.25):
        self.model = model
        self.threshold = threshold

    def classify(self, X):
        return np.mean(patchify_gt(self.model.classify(X)), axis=(1, 2)) > self.threshold


def load_image(filename):
    return mpimg.imread(filename)

def save_image(x, filename):
    Image.fromarray(np.uint8(x * 255)).save(filename)