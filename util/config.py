
import numpy as np
from sklearn.metrics import f1_score


# seed used for pseudo-random number generators
RND_SEED = 1

focus_size = 16
window_size = 100

batch_size = 250
steps_per_epoch = 200
epochs = 100


# shape of output: (number of images, width, height)


# score functions
def accuracy(Y, Y_star):
    Y = Y.reshape(-1)           # vectorize
    Y_star = Y_star.reshape(-1) # vectorize

    return np.sum(Y == Y_star) / Y.size

def mean_f_score(Y, Y_star):
    Y = Y.reshape(-1)           # vectorize
    Y_star = Y_star.reshape(-1) # vectorize

    return f1_score(Y_star, Y)