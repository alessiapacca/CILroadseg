
from util.config import *
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
    def __init__(self, model):
        self.model = model # (window) -> (patch)

        self.focus_size = focus_size
        self.window_size = window_size
        self.padding = (self.window_size - self.focus_size) // 2

    def initialize(self):
        self.model.initialize()

    def sample_window(self, Y, X):
        window_size = self.window_size
        focus_size = self.focus_size

        img_size = X.shape

        # window has to fit in the image
        window_center = (np.random.randint(window_size // 2, img_size[0] - window_size // 2),
                         np.random.randint(window_size // 2, img_size[1] - window_size // 2))

        X_sample = X[
                   window_center[0] - window_size // 2: window_center[0] + window_size // 2,
                   window_center[1] - window_size // 2: window_center[1] + window_size // 2
                   ]

        Y_sample = np.mean(Y[
                           window_center[0] - focus_size // 2: window_center[0] + focus_size // 2,
                           window_center[1] - focus_size // 2: window_center[1] + focus_size // 2
                           ])

        Y_sample = (np.array([Y_sample]) > 0.25) * 1

        # data augmentation: random flip and rotation (in steps of 90°)
        flip = np.random.choice(2)
        rot_step = np.random.choice(4)

        if flip: X_sample = np.fliplr(X_sample)
        X_sample = np.rot90(X_sample, rot_step)
        return Y_sample, X_sample

    def train(self, Y, X):
        padding = self.padding

        X_pad = np.empty((X.shape[0], X.shape[1] + 2*padding, X.shape[2] + 2*padding, X.shape[3]))
        Y_pad = np.empty((Y.shape[0], Y.shape[1] + 2*padding, Y.shape[2] + 2*padding))

        for i in range(X.shape[0]):
            X_pad[i] = pad_image(X[i], padding)
            Y_pad[i] = pad_gt(Y[i], padding)

        def bootstrap(Y, X):
            while 1:
                img_id = np.random.choice(X.shape[0])
                yield self.sample_window(Y[img_id], X[img_id])

        print('valsplit')
        perm = np.random.permutation(X.shape[0])

        val_split = int(0.9 * X.shape[0])
        Itr = perm[0: val_split]
        Ival = perm[val_split: X.shape[0]]

        self.train_online(bootstrap(Y_pad[Itr], X_pad[Itr]), bootstrap(Y_pad[Ival], X_pad[Ival]))

    def train_online(self, generator, val_generator = None):
        self.model.train_online(generator, val_generator)

    #
    # Divides the image in windows.
    # A window should be of shape (window_size, window_size, 3) and there
    # should be (width / focus_size * height / focus_size) windows.
    #
    def crop_windows(self, X):
        focus_size = self.focus_size
        stride = self.focus_size
        padding = self.padding

        img_size = (X.shape[0], X.shape[1])
        list_patches = []

        X = pad_image(X, padding)
        for i in range(padding, img_size[1] + padding, stride):
            for j in range(padding, img_size[0] + padding, stride):
                list_patches.append(X[j-padding:j+focus_size+padding, i-padding:i+focus_size+padding, :])

        return list_patches

    #
    # Divides the given images into focus_size x focus_size squares, in order to classify them.
    # Also the surrounding pixels are taken, up to a window_size x window_size matrix.
    # X - the list of images. This should be a numpy array with shape (num_images, width, height, channels)
    #
    def create_windows(self, X):
        windows = np.asarray([self.crop_windows(X[i]) for i in range(X.shape[0])])
        return windows.reshape((-1, windows.shape[2], windows.shape[3], windows.shape[4]))

    def classify(self, X):
        X_windows = self.create_windows(X)
        return self.model.classify(X_windows)

    def summary(self):
        self.model.summary()

    def save(self, filename):
        self.model.save(filename)

    def load(self, filename):
        self.model.load(filename)