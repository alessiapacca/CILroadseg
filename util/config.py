
import numpy as np


# seed used for pseudo-random number generators
RND_SEED = 1


# score function
def score(Y, Y_star):
    Y = Y.reshape(-1)           # vectorize
    Y_star = Y_star.reshape(-1) # vectorize

    return np.sum(Y == Y_star) / Y.size