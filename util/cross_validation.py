
from util.config import *

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

    if Z.shape != Y_te.shape:
        raise ValueError('The model returned data with different shape: (' + str(Z.shape) + ' vs ' + str(Y_te.shape) + ')')

    # NOTE: this assumes all the data to be already vectorized and with values in {0, 1}.
    return accuracy(Z, Y_te), mean_f_score(Z, Y_te)


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

    accuracy_v = np.empty(K)
    fscore_v = np.empty(K)

    for i in range(K):
        fold_indices = perm[i * fold_size : (i + 1) * fold_size]
        non_fold_indices = perm[np.arange(perm.shape[0]) != i].ravel()

        accuracy_v[i], fscore_v[i] = validate_fold(model, fold_indices, non_fold_indices, X, Y)
        print("Fold #" + str(i+1) + ": ")
        print("    Accuracy: " + str(accuracy_v[i]))
        print("Mean F Score: " + str(fscore_v[i]))

    print()
    print("Cross Validation done:")
    print()

    print("Accuracy: "+ accuracy_v)
    print("AVG: "+ str(np.mean(accuracy_v)))
    print("STD: "+ str(np.std(accuracy_v)))
    print()

    print("Mean F Score: "+ fscore_v)
    print("AVG: "+ str(np.mean(fscore_v)))
    print("STD: "+ str(np.std(fscore_v)))
    print()