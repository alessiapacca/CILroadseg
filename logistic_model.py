
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import PolynomialFeatures

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

class LogisticModel(ModelBase):

    def __init__(self):
        self.model = None

    def initialize(self):
        self.model = LogisticRegression(C=1e5, max_iter = 500)

    def patches_to_features(self, X_patches):
        # Basic features as suggested in project statement
        X_mean = np.mean(X_patches, axis=(1,2))
        X_std = np.std(X_patches, axis=(1,2))
        X = np.append(X_mean, X_std)

        # Polynomial expansion of features
        poly_expansion = PolynomialFeatures(5, interaction_only=False)
        return poly_expansion.fit_transform(X)

    def decompose(self, Y, X):
        X_patches = []
        Y_patches = []

        for i in range(X.shape[0]):
            X_patches += patchify(X[i])
            Y_patches += patchify_gt(Y[i])

        X_patches = np.array(X_patches)
        Y_patches = np.array(Y_patches)

        return Y_patches, self.patches_to_features(X_patches)

    def train(self, Y, X):
        Y_f, X_f = self.decompose(Y, X)

        self.model.fit(Y_f, X_f)

    def classify(self, X):
        X_patches = []

        for i in range(X.shape[0]):
            X_patches += patchify(X[i])

        X_features = self.patches_to_features(np.array(X_patches))

        return self.model.predict(X_features)