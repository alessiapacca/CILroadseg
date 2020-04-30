
import numpy as np


# seed used for pseudo-random number generators
RND_SEED = 1
patch_size = 16
batch_size = 250
steps_per_epoch = 200
epochs = 20
window_size = 72


# shape of output: (number of images, width, height)


# score function
def score(Y, Y_star):
    Y = Y.reshape(-1)           # vectorize
    Y_star = Y_star.reshape(-1) # vectorize

    return np.sum(Y == Y_star) / Y.size