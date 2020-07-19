
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

from util.model_base import ModelBase

def patchify(X):
    patches = []

    width = X.shape[0]
    height = X.shape[1]

    for y in range(0, height, 16):
        for x in range(0, width, 16):
            patches.append(X[x:x+16, y:y+16, :])

    return patches


def patchify_gt(X):
    patches = []

    width = X.shape[0]
    height = X.shape[1]

    for y in range(0, height, 16):
        for x in range(0, width, 16):
            patches.append((np.mean(X[x:x+16, y:y+16]) >= 0.25) * 1)

    return patches


def patches_to_features(X_patches):
    # Basic features as suggested in project statement
    X_mean = np.mean(X_patches, axis=(1,2))
    X_var = np.var(X_patches, axis=(1,2))
    X = np.empty((X_mean.shape[0], 6))

    X[:, 0:3] = X_mean
    X[:, 3:6] = X_var

    # Polynomial expansion of features
    poly_expansion = PolynomialFeatures(5, interaction_only=False)
    return poly_expansion.fit_transform(X)


def decompose(Y, X):
    X_patches = []
    Y_patches = []

    for i in range(X.shape[0]):
        X_patches += patchify(X[i])
        Y_patches += patchify_gt(Y[i])

    X_patches = np.array(X_patches)
    Y_patches = np.array(Y_patches)

    return Y_patches, patches_to_features(X_patches)


class LogisticModel(ModelBase):

    def __init__(self):
        self.model = None
        self.scaler = None

    def initialize(self):
        self.model = LogisticRegression(C=1e5, class_weight='balanced', max_iter=500)
        self.scaler = StandardScaler()

    def train(self, Y, X):
        Y_f, X_f = decompose(Y, X)

        print('Training Logistic Regression... ', end = '')
        X_f = self.scaler.fit_transform(X_f)
        self.model.fit(X_f, Y_f)
        print('done.')

    def classify(self, X):
        X_patches = []

        for i in range(X.shape[0]):
            X_patches += patchify(X[i])

        X_features = patches_to_features(np.array(X_patches))

        X_features = self.scaler.transform(X_features)
        return self.model.predict(X_features)