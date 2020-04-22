
from util.config import *
from util.helpers import patchify

import numpy as np


#
# Validates the given model using the given sample indices as validation set.
#
# fold - array of indices of the samples in X that should be used as validation set
# non_fold - array of indices of the samples in X that should be used as training set
#
def validate_fold(model, fold, non_fold, X, Y):

    X_tr = X[non_fold]
    Y_tr = Y[non_fold]

    X_te = X[fold]
    Y_te = Y[fold]

    model.initialize() # reset the model
    model.train(Y_tr, X_tr)

    Z = model.classify(X_te)

    #Y_te = patchify(Y_te, 16)

    if Z.shape != Y_te.shape:
        raise ValueError('The model returned data with different shape: (' + str(Z.shape) + ' vs ' + str(Y_te.shape) + ')')

    # NOTE: this assumes all the data to be already vectorized and with values in {0, 1}.
    return score(Z, Y_te)


#
# Performs a K-fold cross validation on the given model.
#
# model - Keras model or equivalent interface
# X, Y - numpy array containing training data (input, labels)
#
def cross_validate(model, K, X, Y):
    np.random.seed(RND_SEED) # fix randomness

    perm = np.random.permutation(Y.shape[0]) # randomize folds
    # Y.shape[0] is the number of samples

    fold_size = int(Y.shape[0] / K)
    # range of i-th fold is [i*fold_size, (i+1)*fold_size]

    results = np.zeros(K)

    for i in range(K):
        fold_indices = perm[i * fold_size : (i + 1) * fold_size]
        non_fold_indices = perm[np.arange(perm.shape[0]) != i].ravel()

        results[i] = validate_fold(model, fold_indices, non_fold_indices, X, Y)
        print("Fold #" + str(i+1) + ": " + str(results[i]))

    print()
    print("Cross Validation done:")
    print(results)

    print("AVG: "+ str(np.mean(results)))
    print("STD: "+ str(np.std(results)))